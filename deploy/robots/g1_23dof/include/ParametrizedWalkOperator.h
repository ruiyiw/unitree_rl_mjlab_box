#pragma once
// How the operator tells the robot how far to walk.  The walk states read the distance when
// a walk starts; it can be changed at any time before that, without rebuilding or relaunching:
//
//   --distance 3.5        at launch (else the profile's WalkDistance.default_m)
//   3.5 <Enter>           typed into the controller's terminal while it runs
//   gamepad               the profile's WalkDistance.buttons, each a step in metres
//
// A request is answered with the distance that will be walked: the library's nearest planned
// distance.  A request outside the library is refused and the previous distance kept.
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include "FSM/FSMState.h"
#include "ParametrizedWalkPreflight.h"

namespace parametrized_walk {
inline void report(float metres,const char* source) {
    spdlog::info("Walk distance: {:.1f} m ({}). It applies to the next walk.",metres,source);
}
// Call after the preflight and before the FSM is built.
inline void configure_distance(const YAML::Node& config,const std::filesystem::path& project,const float* launch) {
    std::filesystem::path directory=config["FSM"]["WalkGoal"]["policy_dir"].as<std::string>();
    if(directory.is_relative()) directory=project/directory;
    auto& distance=Distance::instance();
    distance.configure(Library::shared(directory.string()),config["WalkDistance"]["default_m"].as<float>());
    if(launch) distance.set(*launch);   // throws outside the library: a launch with a bad distance stops here
    report(distance.get(),launch?"--distance":"profile default");
}
// The gamepad steps, to be polled once per FSM tick (after the joystick is updated).
inline std::function<void()> distance_buttons(const YAML::Node& config) {
    std::vector<std::pair<std::function<bool()>,float>> buttons;
    if(config["WalkDistance"]["buttons"]) for(const auto& button:config["WalkDistance"]["buttons"]) {
        unitree::common::dsl::Parser parser(button.first.as<std::string>());
        auto ast=parser.Parse();
        auto pressed=unitree::common::dsl::Compile(*ast);
        buttons.emplace_back([pressed]{return pressed(FSMState::lowstate->joystick);},button.second.as<float>());
    }
    return [buttons] {
        for(const auto& button:buttons) if(button.first()) report(Distance::instance().step(button.second),"gamepad");
    };
}
// One distance per line from the terminal.  The thread only ever changes the selection.
inline void read_terminal() {
    std::cout<<"Type a distance in metres and press Enter to change how far the next walk goes ("
             <<Distance::instance().shortest()<<" to "<<Distance::instance().longest()<<" m).\n";
    std::thread([] {
        std::string line;
        while(std::getline(std::cin,line)) {
            std::istringstream text(line);
            float metres=0; std::string rest;
            if(!(text>>metres) || (text>>rest && rest!="m")) {
                if(line.find_first_not_of(" \t\r")!=std::string::npos)
                    spdlog::warn("Not a distance: '{}'. Type metres, for example 3.5",line);
                continue;
            }
            try { report(Distance::instance().set(metres),"terminal"); }
            catch(const std::exception&) {
                spdlog::warn("{} m is outside the planned range ({} to {} m); keeping {:.1f} m.",metres,
                             Distance::instance().shortest(),Distance::instance().longest(),Distance::instance().get());
            }
        }
    }).detach();
}
} // namespace parametrized_walk
