#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include "FSM/FSMState.h"
#include "HumanWalkPolicy.h"

// The human walk as an FSM state: the robot walks the reference's 2.2 m and stops.
//
// Entered from Velocity with the box states' gating and 0.5 s cosine blend.  The reference's
// first frame is held for hold_s while the blend settles; the walk's frame (origin, heading)
// is the robot's pose at entry.  Velocity is offered once the reference has played out, and
// return_s after that the state hands back to Velocity by itself: this policy stands less
// well than it walks (HUMAN_WALK.md), and Velocity is the policy that stands on this robot.
class State_HumanWalk : public FSMState {
public:
    State_HumanWalk(int state,std::string name):FSMState(state,name),
        controller_(param::parser_policy_dir(param::config["FSM"][name]["policy_dir"].as<std::string>()).string()) {
        const auto cfg=param::config["FSM"][name];
        if (cfg["blend_s"]) blend_s_=cfg["blend_s"].as<float>();
        if (cfg["hold_s"]) hold_s_=cfg["hold_s"].as<float>();
        if (cfg["odometry_scale"]) controller_.scale=cfg["odometry_scale"].as<float>();
        if (cfg["return_s"]) return_s_=cfg["return_s"].as<float>();
        const int passive=FSMStringMap.right.at("Passive");
        for (auto& check:registered_checks) {
            if (check.second==passive) continue;
            auto f=check.first;
            check.first=[this,f]{return ended_ && f();};
        }
        if (return_s_>0 && FSMStringMap.right.count("Velocity"))
            registered_checks.insert(registered_checks.begin(),{[this]{return return_now_;},FSMStringMap.right.at("Velocity")});
        registered_checks.insert(registered_checks.begin(),{[this]{return fault_;},passive});
    }
    void enter() override {
        fault_=false; done_=false; ended_=false; reported_=false; return_now_=false; since_end_=0;
        if (lowstate->msg_.mode_machine()!=4) { fail("Requires 23-DoF G1 mode_machine=4"); return; }
        try { controller_.start(hold_s_); }
        catch (const std::exception& e) { fail(e.what()); return; }
        lowcmd->msg_.mode_pr()=0;
        for (size_t i=0;i<23;++i) {               // the command this state blends in from
            const auto& motor=lowcmd->msg_.motor_cmd()[human_walk::kMotorIds[i]];
            q0_[i]=motor.q(); kp0_[i]=motor.kp(); kd0_[i]=motor.kd(); tau0_[i]=motor.tau();
        }
        accel_sum_={0,0,0}; accel_count_=0;
        spdlog::info("HumanWalk: walking {:.2f} m ({:.1f} s reference, after a {:.1f} s hold); arm gravity compensation",
                     human_walk::kWalkMetres,controller_.seconds(),hold_s_);
        t_enter_=next_=Clock::now();
        run();
    }
    void run() override {
        if (fault_) return;
        if (lowstate->isTimeout()) { fail("Robot state timed out"); return; }
        // The accelerometer the position filter expects is the mean specific force over a control
        // period (the change of velocity over 20 ms), so every state sample of the period is averaged.
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
            auto sensors=read_sensors();
            for (size_t i=0;i<3;++i) sensors.accelerometer[i]=float(accel_sum_[i]/accel_count_);
            accel_sum_={0,0,0}; accel_count_=0;
            const auto result=controller_.step(sensors);
            if (-result.observation[human_walk::kGravity+2]<std::cos(3.141592653589793/4)) { fail("Pelvis tilt exceeded 45 degrees"); return; }
            if (Clock::now()>next_) { fail("Walk inference exceeded 20 ms deadline"); return; }
            const double t=std::chrono::duration<double>(Clock::now()-t_enter_).count();
            const float s=blend_s_>0?.5f-.5f*std::cos(3.141592653589793*std::clamp(t/blend_s_,0.0,1.0)):1.f;
            for (size_t i=0;i<23;++i) {
                auto& motor=lowcmd->msg_.motor_cmd()[human_walk::kMotorIds[i]];
                motor.mode()=1; motor.q()=q0_[i]+(result.target[i]-q0_[i])*s;
                motor.kp()=kp0_[i]+(human_walk::kKp[i]-kp0_[i])*s;
                motor.kd()=kd0_[i]+(human_walk::kKd[i]-kd0_[i])*s;
                motor.dq()=0; motor.tau()=tau0_[i]+(result.torque[i]-tau0_[i])*s;
            }
            if (ended_) ++since_end_;                 // control steps since the reference's last frame was first played
            ended_=result.ended;
            const auto walked=controller_.walked();
            if (ended_ && !reported_) {
                reported_=true;
                if (return_s_>0)
                    spdlog::info("HumanWalk: reference ended; the robot's own estimate is {:.2f} m of {:.2f} m. Velocity takes over in {:.1f} s: keep the sticks centred.",
                                 walked[0],human_walk::kWalkMetres,return_s_);
                else
                    spdlog::info("HumanWalk: reference ended; the robot's own estimate is {:.2f} m of {:.2f} m. [RB + X] returns to Velocity.",
                                 walked[0],human_walk::kWalkMetres);
            }
            if (return_s_>0 && ended_ && !return_now_ && since_end_*human_walk::kStepDt>=return_s_-1e-4f) {
                return_now_=true;
                spdlog::info("HumanWalk: handing back to Velocity; estimate {:.2f} m along, {:+.2f} m to the left (not a measurement).",
                             walked[0],walked[1]);
            }
            if (result.done && !done_) {
                done_=true;
                spdlog::info("HumanWalk: standing still; estimate {:.2f} m along, {:+.2f} m to the left (not a measurement). Holding.",
                             walked[0],walked[1]);
            }
        } catch (const std::exception& e) { fail(e.what()); }
    }
private:
    using Clock=std::chrono::steady_clock;
    human_walk::Controller controller_;
    Clock::time_point next_,t_enter_;
    bool fault_=false,done_=false,ended_=false,reported_=false,return_now_=false;
    float blend_s_=.5f,hold_s_=1.f,return_s_=0.f;
    int since_end_=0;
    std::array<float,23> q0_{},kp0_{},kd0_{},tau0_{};
    std::array<double,3> accel_sum_{};
    int accel_count_=0;
    human_walk::Sensors read_sensors() {
        human_walk::Sensors sensors{};
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& imu=lowstate->msg_.imu_state();
        for (size_t i=0;i<4;++i) sensors.quaternion[i]=imu.quaternion()[i];
        for (size_t i=0;i<3;++i) sensors.gyro[i]=imu.gyroscope()[i];
        for (int id:human_walk::kMotorIds) {
            sensors.position[id]=lowstate->msg_.motor_state()[id].q();
            sensors.velocity[id]=lowstate->msg_.motor_state()[id].dq();
        }
        return sensors;
    }
    void fail(const std::string& reason) {
        fault_=true;
        for (int id:human_walk::kMotorIds) {
            auto& motor=lowcmd->msg_.motor_cmd()[id];
            motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
        }
        spdlog::error("HumanWalk: {}",reason);
    }
};
REGISTER_FSM(State_HumanWalk)
