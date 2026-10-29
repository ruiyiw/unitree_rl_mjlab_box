#pragma once
#include <filesystem>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "BoxPolicy.h"

namespace box_policy {
// No DDS objects are constructed here. Validate the exact profile and run its
// selected actor(s), using the same binary and paths as the real launch.
inline void preflight(const YAML::Node& config, const std::filesystem::path& project) {
    const auto fsm=config["FSM"];
    auto require=[](bool ok,const char* message) {
        if (!ok) throw std::runtime_error(std::string("Box preflight: ")+message);
    };
    const auto states=fsm["_"];
    require(states.IsMap() && states.size()==3,"expected Passive, FixStand and BoxPolicy only");
    require(states["Passive"]["id"].as<int>()==1 && states["FixStand"]["id"].as<int>()==2 &&
            states["BoxPolicy"]["id"].as<int>()==7,"unexpected FSM IDs");
    require(states["BoxPolicy"]["type"].as<std::string>()=="BoxPolicy","incorrect FSM type");
    require(states.begin()->first.as<std::string>()=="Passive","first state must be Passive");
    const auto name=fsm["BoxPolicy"]["stage"].as<std::string>();
    int selected=-2;
    if (name=="sequence") selected=-1;
    for (int i=0;i<5;++i) if (name==kNames[i]) selected=i;
    require(selected>=-1,"unknown stage");
    const int initial=selected<0?0:selected;
    for (const auto& entry: std::array<std::pair<const char*,const char*>,3>{{
            {"Passive","FixStand"},{"FixStand","BoxPolicy"},{"BoxPolicy","Passive"}}}) {
        const std::string expected=std::string(entry.first)=="Passive"?"LT + up.on_pressed":
            std::string(entry.first)=="FixStand"?"RT + A.on_pressed":"LT + B.on_pressed";
        require(fsm[entry.first]["transitions"][entry.second].as<std::string>()==expected,"unexpected operator transition");
    }
    require(fsm["FixStand"]["transitions"]["Passive"].as<std::string>()=="LT + B.on_pressed","missing stand exit");
    auto modes=fsm["Passive"]["mode"].as<std::vector<int>>();
    auto damping=fsm["Passive"]["kd"].as<std::vector<float>>();
    auto kp=fsm["FixStand"]["kp"].as<std::vector<float>>();
    auto kd=fsm["FixStand"]["kd"].as<std::vector<float>>();
    auto times=fsm["FixStand"]["ts"].as<std::vector<float>>();
    auto poses=fsm["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
    require(modes.size()==29 && damping.size()==29 && kp.size()==29 && kd.size()==29,"expected 29 SDK slots");
    require(times==std::vector<float>{0,3} && poses.size()==2 && poses[0].empty() && poses[1].size()==29,"incorrect FixStand interpolation");
    auto near=[](float a,float b){return std::isfinite(a) && std::abs(a-b)<1e-5f;};
    for (int id=0;id<29;++id) {
        const auto it=std::find(kMotorIds.begin(),kMotorIds.end(),id);
        if (it==kMotorIds.end()) {
            require(modes[id]==0 && kp[id]==0 && kd[id]==0 && damping[id]==0,"absent motor enabled");
        } else {
            const auto i=it-kMotorIds.begin();
            require(modes[id]==1 && near(damping[id],3),"invalid passive damping");
            require(near(kp[id],kKp[i]) && near(kd[id],kKd[i]),"gain mismatch");
            require(near(poses[1][id],kStarts[initial][i]),"initial reference pose mismatch");
        }
    }
    std::filesystem::path directory=fsm["BoxPolicy"]["policy_dir"].as<std::string>();
    if (directory.is_relative()) directory=project/directory;
    Controller controller(directory.string(),selected);
    Sensors sensors{}; sensors.quaternion[0]=1;
    for (size_t i=0;i<23;++i) sensors.position[kMotorIds[i]]=kStarts[initial][i];
    const auto result=controller.step(sensors);
    require(result.stage==initial && result.clock==0,"incorrect initial controller state");
    // Sequence construction loads all five; exercise inference in every actor too.
    if (selected==-1) for (int stage=1;stage<5;++stage) {
        Controller single(directory.string(),stage);
        for (size_t i=0;i<23;++i) sensors.position[kMotorIds[i]]=kStarts[stage][i];
        single.step(sensors);
    }
    std::cout<<"Box preflight passed: "<<name<<"; 75 inputs, 23 outputs; no DDS connection or motor commands.\n";
}
} // namespace box_policy
