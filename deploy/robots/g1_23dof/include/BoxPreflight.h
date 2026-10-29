#pragma once
#include <filesystem>
#include <iostream>
#include <map>
#include <yaml-cpp/yaml.h>
#include "param.h"
#include "BoxPolicy.h"

namespace box_policy {
// No DDS objects are constructed here. Validate the exact profile and run its
// selected actor(s), using the same binary and paths as the real launch.
inline void preflight(const YAML::Node& config, const std::filesystem::path& project) {
    const auto fsm=config["FSM"];
    auto require=[](bool ok,const std::string& message) {
        if (!ok) throw std::runtime_error("Box preflight: "+message);
    };
    auto resolve=[&](const std::string& path) {
        std::filesystem::path directory=path;
        return directory.is_relative()?project/directory:directory;
    };
    const auto states=fsm["_"];
    require(states.IsMap(),"missing FSM state list");
    require(states.begin()->first.as<std::string>()=="Passive","first state must be Passive");
    require(states["Passive"]["id"].as<int>()==1 && states["FixStand"]["id"].as<int>()==2,"unexpected FSM IDs");
    std::map<std::string,int> stages; // box state -> selection (-1: sequence)
    std::map<int,std::string> ids;
    for (const auto& entry:states) {
        const auto name=entry.first.as<std::string>();
        require(ids.emplace(entry.second["id"].as<int>(),name).second,"duplicate FSM ID for "+name);
        const auto type=entry.second["type"]?entry.second["type"].as<std::string>():name;
        if (type=="BoxPolicy") {
            const auto stage=fsm[name]["stage"].as<std::string>();
            int selected=-2;
            if (stage=="sequence") selected=-1;
            for (int i=0;i<5;++i) if (stage==kNames[i]) selected=i;
            require(selected>=-1,"unknown stage for "+name);
            stages[name]=selected;
        } else if (type=="RLBase") {
            // Velocity: its actuated motors must be the box's (never the absent ones).
            const auto dir=param::parser_policy_dir(resolve(fsm[name]["policy_dir"].as<std::string>())); // as RLBase does
            require(std::filesystem::exists(dir/"exported/policy.onnx"),name+": missing exported/policy.onnx");
            const auto deploy=YAML::LoadFile((dir/"params/deploy.yaml").string());
            const auto map=deploy["joint_ids_map"].as<std::vector<int>>();
            require(std::vector<int>(kMotorIds.begin(),kMotorIds.end())==map,name+": joint_ids_map is not the 23-DoF motor set");
        } else {
            require(type=="Passive" || type=="FixStand","unsupported state type "+type+" in a box profile");
        }
    }
    require(!stages.empty(),"no BoxPolicy state");
    for (const auto& entry:states) {
        const auto name=entry.first.as<std::string>();
        const auto transitions=fsm[name]["transitions"];
        require(transitions && transitions.IsMap(),name+": missing transitions");
        for (const auto& t:transitions)
            require(static_cast<bool>(states[t.first.as<std::string>()]),name+": transition to unknown state "+t.first.as<std::string>());
        if (name!="Passive")
            require(transitions["Passive"] && transitions["Passive"].as<std::string>()=="LT + B.on_pressed",name+": missing LT + B Passive exit");
    }
    require(fsm["Passive"]["transitions"]["FixStand"].as<std::string>()=="LT + up.on_pressed","unexpected operator transition");
    auto modes=fsm["Passive"]["mode"].as<std::vector<int>>();
    auto damping=fsm["Passive"]["kd"].as<std::vector<float>>();
    auto kp=fsm["FixStand"]["kp"].as<std::vector<float>>();
    auto kd=fsm["FixStand"]["kd"].as<std::vector<float>>();
    auto times=fsm["FixStand"]["ts"].as<std::vector<float>>();
    auto poses=fsm["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
    require(modes.size()==29 && damping.size()==29 && kp.size()==29 && kd.size()==29,"expected 29 SDK slots");
    require(times==std::vector<float>{0,3} && poses.size()==2 && poses[0].empty() && poses[1].size()==29,"incorrect FixStand interpolation");
    // FixStand ends on the start pose of every box state it leads to (one with
    // no box button ends on go-to's: the pose Velocity is entered from).
    std::vector<int> stand_targets;
    for (const auto& t:fsm["FixStand"]["transitions"]) {
        const auto it=stages.find(t.first.as<std::string>());
        if (it!=stages.end()) stand_targets.push_back(std::max(it->second,0));
    }
    if (stand_targets.empty()) stand_targets.push_back(0);
    auto near=[](float a,float b){return std::isfinite(a) && std::abs(a-b)<1e-5f;};
    for (int id=0;id<29;++id) {
        const auto it=std::find(kMotorIds.begin(),kMotorIds.end(),id);
        if (it==kMotorIds.end()) {
            require(modes[id]==0 && kp[id]==0 && kd[id]==0 && damping[id]==0,"absent motor enabled");
        } else {
            const auto i=it-kMotorIds.begin();
            require(modes[id]==1 && near(damping[id],3),"invalid passive damping");
            require(near(kp[id],kKp[i]) && near(kd[id],kKd[i]),"gain mismatch");
            for (int initial:stand_targets)
                require(near(poses[1][id],kStarts[initial][i]),"initial reference pose mismatch");
        }
    }
    for (const auto& [name,selected]:stages) {
        const auto directory=resolve(fsm[name]["policy_dir"].as<std::string>()).string();
        const int initial=selected<0?0:selected;
        Controller controller(directory,selected);
        Sensors sensors{}; sensors.quaternion[0]=1;
        for (size_t i=0;i<23;++i) sensors.position[kMotorIds[i]]=kStarts[initial][i];
        const auto result=controller.step(sensors);
        require(result.stage==initial && result.clock==0,"incorrect initial controller state");
        // Sequence construction loads all five; exercise inference in every actor too.
        if (selected==-1) for (int stage=1;stage<5;++stage) {
            Controller single(directory,stage);
            for (size_t i=0;i<23;++i) sensors.position[kMotorIds[i]]=kStarts[stage][i];
            single.step(sensors);
        }
        std::cout<<"Box preflight passed: "<<name<<" ("<<(selected<0?"sequence":kNames[selected])<<"); 75 inputs, 23 outputs.\n";
    }
    std::cout<<"Box preflight passed; no DDS connection or motor commands.\n";
}
} // namespace box_policy
