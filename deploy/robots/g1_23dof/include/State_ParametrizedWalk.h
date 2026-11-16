#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include "FSM/FSMState.h"
#include "ParametrizedWalkPolicy.h"

// walk(N) as an FSM state: one of the two policies walks the operator's distance and stops.
//
// Entered from Velocity with the box states' gating and 0.5 s cosine blend.  The distance is
// read once, on entry, from parametrized_walk::Distance.  The plan's first frame is held for
// hold_s while the blend settles; the walk's frame (origin, heading) is the robot's pose at
// entry.  Velocity is offered once the plan has played out; the other walk policy, and
// walking again on this one ("again"), once the robot has also stood still for a second.
class State_ParametrizedWalk : public FSMState {
public:
    State_ParametrizedWalk(int state,std::string name):FSMState(state,name),
        policy_(selection(param::config["FSM"][name]["policy"].as<std::string>())),
        controller_(param::parser_policy_dir(param::config["FSM"][name]["policy_dir"].as<std::string>()).string(),policy_) {
        const auto cfg=param::config["FSM"][name];
        if (cfg["blend_s"]) blend_s_=cfg["blend_s"].as<float>();
        if (cfg["hold_s"]) hold_s_=cfg["hold_s"].as<float>();
        if (cfg["shortest_m"]) shortest_=cfg["shortest_m"].as<float>();
        if (cfg["odometry_scale"]) scale_=cfg["odometry_scale"].as<float>();
        if (cfg["again"]) {
            unitree::common::dsl::Parser parser(cfg["again"].as<std::string>());
            auto ast=parser.Parse();
            auto pressed=unitree::common::dsl::Compile(*ast);
            again_=[pressed]{return pressed(FSMState::lowstate->joystick);};
        }
        const int passive=FSMStringMap.right.at("Passive");
        for (auto& check:registered_checks) {
            if (check.second==passive) continue;
            const auto type=param::config["FSM"]["_"][FSMStringMap.left.at(check.second)]["type"];
            const bool walk=type && type.as<std::string>()=="ParametrizedWalk";
            auto f=check.first;
            check.first=[this,f,walk]{return (walk?done_:ended_) && f();};
        }
        // A refused distance goes straight back to Velocity; a fault to Passive.
        if (FSMStringMap.right.count("Velocity"))
            registered_checks.insert(registered_checks.begin(),{[this]{return refused_;},FSMStringMap.right.at("Velocity")});
        registered_checks.insert(registered_checks.begin(),{[this]{return fault_;},passive});
    }
    void enter() override {
        fault_=false; refused_=false; done_=false; ended_=false; again_now_=false; reported_=false;
        const bool handoff=last_.done && Clock::now()-last_.time<std::chrono::milliseconds(50);
        last_.done=false;
        if (lowstate->msg_.mode_machine()!=4) { fail("Requires 23-DoF G1 mode_machine=4"); return; }
        try {
            const float want=parametrized_walk::Distance::instance().get();
            if (want<shortest_-1e-4f) {
                refused_=true;
                spdlog::error("ParametrizedWalk: {} m refused; the {} policy's shortest walk is {} m. Back to Velocity.",
                              want,parametrized_walk::kNames[policy_],shortest_);
                return;
            }
            controller_.estimator().scale=scale_;
            metres_=controller_.start(want,handoff?0.f:hold_s_,handoff?&last_.action:nullptr);
        } catch (const std::exception& e) { fail(e.what()); return; }
        lowcmd->msg_.mode_pr()=0;
        for (size_t i=0;i<23;++i) {               // the command this state blends in from
            const auto& motor=lowcmd->msg_.motor_cmd()[parametrized_walk::kMotorIds[i]];
            q0_[i]=motor.q(); kp0_[i]=motor.kp(); kd0_[i]=motor.kd(); tau0_[i]=motor.tau();
        }
        blend_now_=handoff?0.f:blend_s_;
        accel_sum_={0,0,0}; accel_count_=0;
        spdlog::info("ParametrizedWalk: {} policy, walking {:.1f} m ({:.1f} s plan)",
                     parametrized_walk::kNames[policy_],metres_,controller_.seconds());
        t_enter_=next_=Clock::now();
        run();
    }
    void run() override {
        if (fault_ || refused_) return;
        if (lowstate->isTimeout()) { fail("Robot state timed out"); return; }
        // A button press lasts one FSM tick: note it before waiting for the control period.
        if (done_ && again_ && again_()) again_now_=true;
        // Training's accelerometer is the mean specific force over a control period (the change
        // of velocity over 20 ms), so every state sample of the period is averaged, not the last.
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            const auto& a=lowstate->msg_.imu_state().accelerometer();
            for (size_t i=0;i<3;++i) accel_sum_[i]+=a[i];
            ++accel_count_;
        }
        const auto now=Clock::now();
        if (now<next_) return;
        // A reference clock cannot skip or catch up after a long stall safely.
        if (now-next_>std::chrono::milliseconds(20)) { fail("Missed walk control deadline"); return; }
        next_+=std::chrono::milliseconds(20);
        try {
            if (again_now_) walk_again();
            auto sensors=read_sensors();
            for (size_t i=0;i<3;++i) sensors.accelerometer[i]=float(accel_sum_[i]/accel_count_);
            accel_sum_={0,0,0}; accel_count_=0;
            const auto result=controller_.step(sensors);
            if (-result.observation[5]<std::cos(3.141592653589793/4)) { fail("Pelvis tilt exceeded 45 degrees"); return; }
            if (Clock::now()>next_) { fail("Walk inference exceeded 20 ms deadline"); return; }
            const double t=std::chrono::duration<double>(Clock::now()-t_enter_).count();
            const float s=blend_now_>0?.5f-.5f*std::cos(3.141592653589793*std::clamp(t/blend_now_,0.0,1.0)):1.f;
            for (size_t i=0;i<23;++i) {
                auto& motor=lowcmd->msg_.motor_cmd()[parametrized_walk::kMotorIds[i]];
                motor.mode()=1; motor.q()=q0_[i]+(result.target[i]-q0_[i])*s;
                motor.kp()=kp0_[i]+(parametrized_walk::kKp[i]-kp0_[i])*s;
                motor.kd()=kd0_[i]+(parametrized_walk::kKd[i]-kd0_[i])*s;
                motor.dq()=0; motor.tau()=tau0_[i]*(1.f-s);   // plain PD: no feedforward torque
            }
            ended_=result.ended;
            const auto walked=controller_.walked();
            if (ended_ && !reported_) {
                reported_=true;
                spdlog::info("ParametrizedWalk: plan ended; the robot's own estimate is {:.2f} m of {:.1f} m. [RB + X] returns to Velocity.",
                             walked[0],metres_);
            }
            if (result.done && !done_) {
                done_=true;
                spdlog::info("ParametrizedWalk: standing still; estimate {:.2f} m along, {:+.2f} m to the left (not a measurement). Holding.",
                             walked[0],walked[1]);
            }
        } catch (const std::exception& e) { fail(e.what()); }
    }
    void exit() override {
        last_={Clock::now(),done_ && !fault_ && !refused_,controller_.previous()};
    }
private:
    using Clock=std::chrono::steady_clock;
    struct Handoff { Clock::time_point time; bool done; parametrized_walk::Action action; };
    inline static Handoff last_{};
    int policy_;
    parametrized_walk::Controller controller_;
    std::function<bool()> again_;
    Clock::time_point next_,t_enter_;
    bool fault_=false,refused_=false,done_=false,ended_=false,again_now_=false,reported_=false;
    float blend_s_=.5f,blend_now_=.5f,hold_s_=1.f,shortest_=0.f,scale_=1.f,metres_=0.f;
    std::array<float,23> q0_{},kp0_{},kd0_{},tau0_{};
    std::array<double,3> accel_sum_{};
    int accel_count_=0;
    static int selection(const std::string& name) {
        for (size_t i=0;i<parametrized_walk::kNames.size();++i) if (name==parametrized_walk::kNames[i]) return int(i);
        throw std::runtime_error("Unknown walk policy: "+name);
    }
    // The settled robot walks the currently selected distance from where it stands, with no blend.
    void walk_again() {
        again_now_=false;
        const float want=parametrized_walk::Distance::instance().get();
        if (want<shortest_-1e-4f) {
            spdlog::error("ParametrizedWalk: {} m refused; the {} policy's shortest walk is {} m. Still holding.",
                          want,parametrized_walk::kNames[policy_],shortest_);
            return;
        }
        const auto previous=controller_.previous();
        metres_=controller_.start(want,0.f,&previous);
        done_=false; ended_=false; reported_=false; blend_now_=0;
        spdlog::info("ParametrizedWalk: {} policy again, walking {:.1f} m ({:.1f} s plan)",
                     parametrized_walk::kNames[policy_],metres_,controller_.seconds());
    }
    parametrized_walk::Sensors read_sensors() {
        parametrized_walk::Sensors sensors{};
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& imu=lowstate->msg_.imu_state();
        for (size_t i=0;i<4;++i) sensors.quaternion[i]=imu.quaternion()[i];
        for (size_t i=0;i<3;++i) sensors.gyro[i]=imu.gyroscope()[i];
        for (int id:parametrized_walk::kMotorIds) {
            sensors.position[id]=lowstate->msg_.motor_state()[id].q();
            sensors.velocity[id]=lowstate->msg_.motor_state()[id].dq();
        }
        return sensors;
    }
    void fail(const std::string& reason) {
        fault_=true;
        for (int id:parametrized_walk::kMotorIds) {
            auto& motor=lowcmd->msg_.motor_cmd()[id];
            motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
        }
        spdlog::error("ParametrizedWalk: {}",reason);
    }
};
REGISTER_FSM(State_ParametrizedWalk)
