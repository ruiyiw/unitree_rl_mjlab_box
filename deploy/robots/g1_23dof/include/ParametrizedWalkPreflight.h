#pragma once
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <yaml-cpp/yaml.h>
#include "param.h"
#include "unitree_joystick_dsl.hpp"
#include "ParametrizedWalkPolicy.h"

namespace parametrized_walk {
inline bool is_profile(const YAML::Node& config) {
    for(const auto& state:config["FSM"]["_"])
        if(state.second["type"] && state.second["type"].as<std::string>()=="ParametrizedWalk") return true;
    return false;
}
// A gamepad expression must parse and name real keys: an unknown key would otherwise only
// throw when the expression is first evaluated, inside the control loop.
inline void check_expression(const std::string& text) {
    unitree::common::dsl::Parser parser(text);
    const auto ast=parser.Parse();
    const unitree::common::UnitreeJoystick joystick;
    std::function<void(const unitree::common::dsl::Node&)> visit=[&](const unitree::common::dsl::Node& node) {
        if(node.kind==unitree::common::dsl::Node::kAtom) unitree::common::dsl::GetKey(joystick,node.atom.name);
        if(node.lhs) visit(*node.lhs);
        if(node.rhs) visit(*node.rhs);
    };
    visit(*ast);
}
// No DDS objects are constructed here.  Check the profile, load the walk library and both
// actors with the launch's own binary and paths, and run one step of each.
inline void preflight(const YAML::Node& config,const std::filesystem::path& project) {
    auto require=[](bool ok,const std::string& why) {
        if(!ok) throw std::runtime_error("Parametrized-walk preflight: "+why);
    };
    const auto fsm=config["FSM"],states=fsm["_"];
    require(states.IsMap() && states.size()==5,"expected Passive, FixStand, Velocity and the two walk states");
    const std::array<std::string,5> names{"Passive","FixStand","Velocity","WalkGoal","WalkAdapter"};
    size_t index=0;
    for(const auto& entry:states) {
        require(entry.first.as<std::string>()==names[index],"unexpected state order/name");
        require(entry.second["id"].as<int>()==int(index+1),"unexpected state ID");
        const auto type=entry.second["type"]?entry.second["type"].as<std::string>():entry.first.as<std::string>();
        require(type==(index<2?names[index]:index==2?"RLBase":"ParametrizedWalk"),"unexpected state type");
        ++index;
    }
    // A walk starts from Velocity with the sticks centred, or from the other walk policy
    // once the robot stands still; every active state has the Passive exit.
    const std::array<std::map<std::string,std::string>,5> edges{{
        {{"FixStand","LT + up.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Velocity","RB + X.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"WalkGoal","RB + Y.on_pressed"},{"WalkAdapter","RB + A.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Velocity","RB + X.on_pressed"},{"WalkAdapter","RB + A.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Velocity","RB + X.on_pressed"},{"WalkGoal","RB + Y.on_pressed"}}
    }};
    for(size_t i=0;i<names.size();++i)
        require(fsm[names[i]]["transitions"].as<std::map<std::string,std::string>>()==edges[i],"unexpected transitions for "+names[i]);
    const auto mode=fsm["Passive"]["mode"].as<std::vector<int>>();
    const auto damping=fsm["Passive"]["kd"].as<std::vector<float>>();
    const auto kp=fsm["FixStand"]["kp"].as<std::vector<float>>(),kd=fsm["FixStand"]["kd"].as<std::vector<float>>();
    const auto poses=fsm["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
    require(mode.size()==29 && damping.size()==29 && kp.size()==29 && kd.size()==29,"expected 29 SDK slots");
    require(fsm["FixStand"]["ts"].as<std::vector<float>>()==std::vector<float>{0,3} &&
            poses.size()==2 && poses[0].empty() && poses[1].size()==29,"invalid FixStand interpolation");
    auto near=[](float a,float b){return std::isfinite(a) && std::abs(a-b)<1e-5f;};
    for(int id=0;id<29;++id) {
        const auto it=std::find(kMotorIds.begin(),kMotorIds.end(),id);
        if(it==kMotorIds.end()) require(mode[id]==0 && damping[id]==0 && kp[id]==0 && kd[id]==0,"absent joint enabled");
        else {
            const auto i=it-kMotorIds.begin();
            require(mode[id]==1 && near(damping[id],3) && near(kp[id],kKp[i]) && near(kd[id],kKd[i]),"incorrect gains");
            require(near(poses[1][id],kStart[i]),"FixStand does not end on the walks' starting pose");
        }
    }
    auto resolve=[&](const std::string& value) {
        std::filesystem::path path=value;
        return path.is_absolute()?path:project/path;
    };
    // Velocity: the same 23 motors, and an actor that loads and runs.
    const auto velocity=param::parser_policy_dir(resolve(fsm["Velocity"]["policy_dir"].as<std::string>()));
    const auto deploy=YAML::LoadFile((velocity/"params/deploy.yaml").string());
    require(deploy["joint_ids_map"].as<std::vector<int>>()==std::vector<int>(kMotorIds.begin(),kMotorIds.end()),"velocity motor mapping differs");
    require(std::filesystem::is_regular_file(velocity/"exported/policy.onnx"),"missing velocity actor");
    {
        Ort::Env runtime(ORT_LOGGING_LEVEL_WARNING,"walk_velocity_check");
        Ort::SessionOptions options; options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        Ort::Session session(runtime,(velocity/"exported/policy.onnx").c_str(),options);
        require(session.GetInputCount()==1 && session.GetOutputCount()==1,"invalid velocity actor I/O");
        auto in_type=session.GetInputTypeInfo(0),out_type=session.GetOutputTypeInfo(0);
        const auto shape=in_type.GetTensorTypeAndShapeInfo().GetShape();
        require(shape.size()==2 && shape[0]==1 && shape[1]>0 && out_type.GetTensorTypeAndShapeInfo().GetShape()==std::vector<int64_t>{1,23},
                "unexpected velocity actor shape");
        std::vector<float> observation(shape[1],0.f);
        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto input=Ort::Value::CreateTensor<float>(mem,observation.data(),observation.size(),shape.data(),shape.size());
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name=session.GetInputNameAllocated(0,alloc),out_name=session.GetOutputNameAllocated(0,alloc);
        const char* ins[]{in_name.get()}; const char* outs[]{out_name.get()};
        auto result=session.Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
        for(int i=0;i<23;++i) require(std::isfinite(result[0].GetTensorData<float>()[i]),"non-finite velocity action");
    }
    // The operator's distance: a default inside the library, and buttons that parse.
    const auto distance=config["WalkDistance"];
    require(distance && distance["default_m"],"missing WalkDistance.default_m");
    const auto directory=resolve(fsm["WalkGoal"]["policy_dir"].as<std::string>()).string();
    const auto library=Library::shared(directory);
    const float first=distance["default_m"].as<float>();
    require(std::isfinite(first) && first>=library->shortest() && first<=library->longest(),"default distance outside the library");
    if(distance["buttons"]) for(const auto& button:distance["buttons"]) {
        check_expression(button.first.as<std::string>());
        const float step=button.second.as<float>();
        require(std::isfinite(step) && step!=0 && std::abs(step)<=5,"distance button step must be within 5 m");
    }
    for(int policy=0;policy<2;++policy) {
        const auto name=names[policy+3];
        const auto node=fsm[name];
        require(node["policy"].as<std::string>()==kNames[policy],"wrong policy for "+name);
        require(resolve(node["policy_dir"].as<std::string>()).string()==directory,"the walk states must share one policy directory");
        require(near(node["blend_s"].as<float>(),.5f),"expected 0.5 second entry blend");
        const float hold=node["hold_s"].as<float>(),scale=node["odometry_scale"].as<float>();
        require(hold>=.5f && hold<=5,"hold_s must cover the entry blend (0.5 to 5 s)");
        require(scale>=.9f && scale<=1.1f,"odometry_scale is a calibration near 1");
        const float shortest=node["shortest_m"].as<float>();
        require(shortest>=library->shortest() && shortest<=1,"shortest_m outside the library's short end");
        check_expression(node["again"].as<std::string>());
        Controller controller(directory,policy);
        Sensors sensors{}; sensors.quaternion[0]=1; sensors.accelerometer[2]=9.81f;
        for(size_t i=0;i<23;++i) sensors.position[kMotorIds[i]]=kStart[i];
        for(float metres:{library->shortest(),first,library->longest()}) {
            require(near(controller.start(metres,hold),metres),"distance missing from the library");
            const auto a=controller.step(sensors),b=controller.step(sensors);
            require(a.clock==0 && b.clock==0 && !b.ended,"the plan's first frame is not held on entry");
            for(float v:b.target) require(std::isfinite(v),"non-finite joint target");
        }
        std::cout<<"Parametrized-walk preflight passed: "<<kNames[policy]<<" policy; 109 inputs, 23 outputs.\n";
    }
    std::cout<<"Walk library: "<<library->size()<<" distances, "<<library->shortest()<<" to "<<library->longest()
             <<" m. Velocity and both walk policies passed; no DDS connection or motor commands.\n";
}
} // namespace parametrized_walk
