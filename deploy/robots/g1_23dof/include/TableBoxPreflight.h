#pragma once
#include "BoxPreflight.h"
#include "TableBoxPolicy.h"

namespace table_box {
inline bool is_profile(const YAML::Node& config) {
    for(const auto& state:config["FSM"]["_"])
        if(state.second["type"] && state.second["type"].as<std::string>()=="TableBoxPolicy") return true;
    return false;
}
inline void preflight(const YAML::Node& config,const std::filesystem::path& project) {
    auto require=[](bool ok,const std::string& why) {
        if(!ok) throw std::runtime_error("Table-box preflight: "+why);
    };
    const auto fsm=config["FSM"],states=fsm["_"];
    require(states.IsMap() && states.size()==6,"expected Passive, FixStand, Velocity and three box states");
    const std::array<std::string,6> names{"Passive","FixStand","Velocity","GoTo","Take","PutDown"};
    size_t index=0;
    for(const auto& entry:states) {
        require(entry.first.as<std::string>()==names[index],"unexpected state order/name");
        require(entry.second["id"].as<int>()==int(index+1),"unexpected state ID");
        const auto type=entry.second["type"]?entry.second["type"].as<std::string>():entry.first.as<std::string>();
        require(type==(index<2?names[index]:index==2?"RLBase":"TableBoxPolicy"),"unexpected state type");
        ++index;
    }
    // Explicit transitions avoid accidentally walking with a held box or
    // entering a manipulation policy directly from a moving joystick command.
    const std::array<std::map<std::string,std::string>,6> edges{{
        {{"FixStand","LT + up.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Velocity","RB + X.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"GoTo","RB + Y.on_pressed"},{"Take","RB + A.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Take","RB + A.on_pressed"},{"Velocity","RB + X.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"PutDown","RB + A.on_pressed"}},
        {{"Passive","LT + B.on_pressed"},{"Velocity","RB + X.on_pressed"}}
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
        const auto it=std::find(box_policy::kMotorIds.begin(),box_policy::kMotorIds.end(),id);
        if(it==box_policy::kMotorIds.end()) require(mode[id]==0 && damping[id]==0 && kp[id]==0 && kd[id]==0,"absent joint enabled");
        else {
            const auto i=it-box_policy::kMotorIds.begin();
            require(mode[id]==1 && near(damping[id],3) && near(kp[id],box_policy::kKp[i]) && near(kd[id],box_policy::kKd[i]),"incorrect gains");
            require(near(poses[1][id],kStarts[0][i]),"incorrect FixStand pose");
        }
    }
    auto resolve=[&](const std::string& value) {
        std::filesystem::path path=value;
        return path.is_absolute()?path:project/path;
    };
    const auto velocity=param::parser_policy_dir(resolve(fsm["Velocity"]["policy_dir"].as<std::string>()));
    const auto deploy=YAML::LoadFile((velocity/"params/deploy.yaml").string());
    require(deploy["joint_ids_map"].as<std::vector<int>>()==std::vector<int>(box_policy::kMotorIds.begin(),box_policy::kMotorIds.end()),"velocity motor mapping differs");
    require(std::filesystem::is_regular_file(velocity/"exported/policy.onnx"),"missing velocity actor");
    // Load/infer the actual velocity actor as well (without constructing DDS).
    Ort::Env runtime(ORT_LOGGING_LEVEL_WARNING,"table_velocity_check");
    Ort::SessionOptions options; options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
    Ort::Session session(runtime,(velocity/"exported/policy.onnx").c_str(),options);
    require(session.GetInputCount()==1 && session.GetOutputCount()==1,"invalid velocity actor I/O");
    auto in_type=session.GetInputTypeInfo(0),out_type=session.GetOutputTypeInfo(0);
    const auto in=in_type.GetTensorTypeAndShapeInfo(),out=out_type.GetTensorTypeAndShapeInfo();
    const auto shape=in.GetShape();
    require(shape==std::vector<int64_t>{1,80} && out.GetShape()==std::vector<int64_t>{1,23} &&
            in.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && out.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"unexpected velocity actor shape/type");
    std::array<float,80> observation{};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(mem,observation.data(),observation.size(),shape.data(),shape.size());
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name=session.GetInputNameAllocated(0,alloc),out_name=session.GetOutputNameAllocated(0,alloc);
    const char* ins[]{in_name.get()}; const char* outs[]{out_name.get()};
    auto result=session.Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
    for(int i=0;i<23;++i) require(std::isfinite(result[0].GetTensorData<float>()[i]),"non-finite velocity action");
    for(int stage=0;stage<3;++stage) {
        const auto name=names[stage+3];
        require(fsm[name]["stage"].as<std::string>()==kNames[stage],"wrong stage for "+name);
        require(near(fsm[name]["blend_s"].as<float>(),.5f),"expected 0.5 second entry blend");
        Controller controller(resolve(fsm[name]["policy_dir"].as<std::string>()).string(),stage);
        Sensors sensors{}; sensors.quaternion[0]=1;
        for(size_t i=0;i<23;++i) sensors.position[box_policy::kMotorIds[i]]=kStarts[stage][i];
        const auto step=controller.step(sensors,kStarts[stage]);
        require(step.clock==0,"invalid initial clock");
        for(float v:step.torque) require(std::isfinite(v),"non-finite arm gravity torque");
        std::cout<<"Table-box preflight passed: "<<name<<"; 376 inputs, 23 outputs.\n";
    }
    std::cout<<"Velocity and three box policies passed; no DDS connection or motor commands.\n";
}
} // namespace table_box
