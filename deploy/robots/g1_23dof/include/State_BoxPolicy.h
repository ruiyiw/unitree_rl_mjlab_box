#pragma once
#include <chrono>
#include <mutex>
#include "FSM/FSMState.h"
#include "BoxPolicy.h"

class State_BoxPolicy : public FSMState {
public:
    State_BoxPolicy(int state,std::string name):FSMState(state,name),
        controller_(param::parser_policy_dir(param::config["FSM"][name]["policy_dir"].as<std::string>()).string(),
                    selection(param::config["FSM"][name]["stage"].as<std::string>())) {
        registered_checks.insert(registered_checks.begin(),{
            [this]{return fault_;},FSMStringMap.right.at("Passive")});
    }
    void enter() override {
        fault_=false; controller_.reset(); stage_=-1; done_=false; next_=Clock::now();
        if (lowstate->msg_.mode_machine()!=4) { fail("Requires 23-DoF G1 mode_machine=4"); return; }
        lowcmd->msg_.mode_pr()=0;
        run();
    }
    void run() override {
        if (fault_) return;
        if (lowstate->isTimeout()) { fail("Robot state timed out"); return; }
        const auto now=Clock::now();
        if (now<next_) return;
        // A reference clock cannot skip or catch up after a long stall safely.
        if (now-next_>std::chrono::milliseconds(20)) { fail("Missed box control deadline"); return; }
        next_+=std::chrono::milliseconds(20);
        try {
            box_policy::Sensors sensors{};
            {
                std::lock_guard<std::mutex> lock(lowstate->mutex_);
                for (size_t i=0;i<4;++i) sensors.quaternion[i]=lowstate->msg_.imu_state().quaternion()[i];
                for (size_t i=0;i<3;++i) sensors.gyro[i]=lowstate->msg_.imu_state().gyroscope()[i];
                for (int id:box_policy::kMotorIds) {
                    sensors.position[id]=lowstate->msg_.motor_state()[id].q();
                    sensors.velocity[id]=lowstate->msg_.motor_state()[id].dq();
                }
            }
            const auto check=box_policy::observation(sensors,{});
            if (-check[5]<std::cos(3.141592653589793/4)) { fail("Pelvis tilt exceeded 45 degrees"); return; }
            const auto result=controller_.step(sensors);
            if (Clock::now()>next_) { fail("Box inference exceeded 20 ms deadline"); return; }
            for (size_t i=0;i<23;++i) {
                auto& motor=lowcmd->msg_.motor_cmd()[box_policy::kMotorIds[i]];
                motor.mode()=1; motor.q()=result.target[i];
                motor.kp()=box_policy::kKp[i]; motor.kd()=box_policy::kKd[i];
                motor.dq()=0; motor.tau()=0;
            }
            if (result.stage!=stage_) {
                stage_=result.stage; spdlog::info("BoxPolicy: {}",box_policy::kNames[stage_]);
            }
            if (result.done && !done_) {
                done_=true;
                spdlog::info("BoxPolicy: sensor endpoint settled; holding final reference (physical success unmeasured)");
            }
        } catch (const std::exception& e) { fail(e.what()); }
    }
    void exit() override {controller_.reset();}
private:
    using Clock=std::chrono::steady_clock;
    box_policy::Controller controller_;
    Clock::time_point next_;
    bool fault_=false,done_=false;
    int stage_=-1;
    static int selection(const std::string& name) {
        if (name=="sequence") return -1;
        for (int i=0;i<5;++i) if (name==box_policy::kNames[i]) return i;
        throw std::runtime_error("Unknown box stage: "+name);
    }
    void fail(const std::string& reason) {
        fault_=true;
        for (int id:box_policy::kMotorIds) {
            auto& motor=lowcmd->msg_.motor_cmd()[id];
            motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
        }
        spdlog::error("BoxPolicy: {}",reason);
    }
};
REGISTER_FSM(State_BoxPolicy)
