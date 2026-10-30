#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include "FSM/FSMState.h"
#include "TableBoxPolicy.h"

// Same operator-gated entry/exit and cosine blending as velocity_integration.
// Each 376-input actor resets its four sensor frames on entry, as in training.
class State_TableBoxPolicy : public FSMState {
public:
    State_TableBoxPolicy(int state,std::string name):FSMState(state,name),
        selected_(selection(param::config["FSM"][name]["stage"].as<std::string>())),
        controller_(param::parser_policy_dir(param::config["FSM"][name]["policy_dir"].as<std::string>()).string(),selected_) {
        const auto cfg=param::config["FSM"][name];
        if (cfg["blend_s"]) blend_s_=cfg["blend_s"].as<float>();
        const int passive=FSMStringMap.right.at("Passive");
        for (auto& check:registered_checks) {
            if (check.second==passive) continue;
            const auto type=param::config["FSM"]["_"][FSMStringMap.left.at(check.second)]["type"];
            const bool box=type && type.as<std::string>()=="TableBoxPolicy";
            auto f=check.first;
            check.first=[this,f,box]{return (box?done_:ended_) && f();};
        }
        registered_checks.insert(registered_checks.begin(),{
            [this]{return fault_;},passive});
    }
    void enter() override {
        fault_=false; done_=false; ended_=false;
        const bool handoff=last_.done && Clock::now()-last_.time<std::chrono::milliseconds(50);
        last_.done=false;
        if (handoff) controller_.reset(last_.action); else controller_.reset();
        if (lowstate->msg_.mode_machine()!=4) { fail("Requires 23-DoF G1 mode_machine=4"); return; }
        lowcmd->msg_.mode_pr()=0;
        for (size_t i=0;i<23;++i) {               // the command this state blends in from
            const auto& motor=lowcmd->msg_.motor_cmd()[box_policy::kMotorIds[i]];
            q0_[i]=motor.q(); kp0_[i]=motor.kp(); kd0_[i]=motor.kd(); tau0_[i]=motor.tau();
        }
        blend_now_=handoff?0.f:blend_s_;
        spdlog::info("TableBoxPolicy: {} (376 inputs, arm gravity compensation)",table_box::kNames[selected_]);
        t_enter_=next_=Clock::now();
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
            const auto sensors=read_sensors();
            if (tilted(sensors)) { fail("Pelvis tilt exceeded 45 degrees"); return; }
            box_policy::Action command{};
            for(size_t i=0;i<23;++i) command[i]=lowcmd->msg_.motor_cmd()[box_policy::kMotorIds[i]].q();
            const auto result=controller_.step(sensors,command);
            if (Clock::now()>next_) { fail("Box inference exceeded 20 ms deadline"); return; }
            const double t=std::chrono::duration<double>(Clock::now()-t_enter_).count();
            const float s=blend_now_>0?.5f-.5f*std::cos(3.141592653589793*std::clamp(t/blend_now_,0.0,1.0)):1.f;
            for (size_t i=0;i<23;++i) {
                auto& motor=lowcmd->msg_.motor_cmd()[box_policy::kMotorIds[i]];
                motor.mode()=1; motor.q()=q0_[i]+(result.target[i]-q0_[i])*s;
                motor.kp()=kp0_[i]+(box_policy::kKp[i]-kp0_[i])*s;
                motor.kd()=kd0_[i]+(box_policy::kKd[i]-kd0_[i])*s;
                motor.dq()=0; motor.tau()=tau0_[i]+(result.torque[i]-tau0_[i])*s;
            }
            ended_=controller_.ended();
            if (result.done && !done_) {
                done_=true;
                spdlog::info("TableBoxPolicy: sensor endpoint settled; holding final reference (physical success unmeasured)");
            }
        } catch (const std::exception& e) { fail(e.what()); }
    }
    void exit() override {
        last_={Clock::now(),done_ && !fault_,controller_.previous()};
        controller_.reset();
    }
private:
    using Clock=std::chrono::steady_clock;
    struct Handoff { Clock::time_point time; bool done; box_policy::Action action; };
    inline static Handoff last_{};
    int selected_;
    table_box::Controller controller_;
    Clock::time_point next_,t_enter_;
    bool fault_=false,done_=false,ended_=false;
    float blend_s_=.5f,blend_now_=.5f;
    std::array<float,23> q0_{},kp0_{},kd0_{},tau0_{};
    static int selection(const std::string& name) {
        for (int i=0;i<3;++i) if (name==table_box::kNames[i]) return i;
        throw std::runtime_error("Unknown box stage: "+name);
    }
    box_policy::Sensors read_sensors() {
        box_policy::Sensors sensors{};
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (size_t i=0;i<4;++i) sensors.quaternion[i]=lowstate->msg_.imu_state().quaternion()[i];
        for (size_t i=0;i<3;++i) sensors.gyro[i]=lowstate->msg_.imu_state().gyroscope()[i];
        for (int id:box_policy::kMotorIds) {
            sensors.position[id]=lowstate->msg_.motor_state()[id].q();
            sensors.velocity[id]=lowstate->msg_.motor_state()[id].dq();
        }
        return sensors;
    }
    static bool tilted(const box_policy::Sensors& sensors) {
        const auto check=box_policy::observation(sensors,{});
        return -check[5]<std::cos(3.141592653589793/4);
    }
    void fail(const std::string& reason) {
        fault_=true;
        for (int id:box_policy::kMotorIds) {
            auto& motor=lowcmd->msg_.motor_cmd()[id];
            motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
        }
        spdlog::error("TableBoxPolicy: {}",reason);
    }
};
REGISTER_FSM(State_TableBoxPolicy)
