// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

// Changes from unitree_rl_mjlab (ported from the humanoid repo's g1_23dof deployment, for Velocity):
//  - the gains go to joint_ids_map's slots (deploy.yaml's stiffness/damping are in the policy's joint order;
//    upstream wrote them to slots 0..22, leaving the right arm's 23-26 at the previous state's gains);
//  - nothing is sent before the policy's first step, and its targets and gains are blended in over blend_s
//    (0.5 s) from the command the state was entered with;
//  - a non-finite target goes to Passive; exit() wakes the policy thread instead of waiting out its 20 ms.
class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    void enter()
    {
        const size_t slots = std::min<size_t>(q0_.size(), lowcmd->msg_.motor_cmd().size());
        for (size_t i = 0; i < slots; ++i)         // the command this state blends in from
        {
            const auto & m = lowcmd->msg_.motor_cmd()[i];
            q0_[i] = m.q(); kp0_[i] = m.kp(); kd0_[i] = m.kd(); tau0_[i] = m.tau();
        }
        // upstream's gains (slot i), kept for the robots whose run() writes only q; g1_23dof's run() blends the
        // gains into joint_ids_map's slots from the snapshot above, so these last only until the first step
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }
        for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[static_cast<int>(env->robot->data.joint_ids_map[i])].dq() = 0;
        }
        bad_ = false;
        started_ = false;
        t_enter_ = std::chrono::steady_clock::now();

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                started_ = true;

                // Sleep (exit() wakes it); after an overrun, drop the missed steps rather than bursting them
                {
                    std::unique_lock<std::mutex> lk(wake_mtx_);
                    wake_.wait_until(lk, sleepTill, [this] { return !policy_thread_running; });
                }
                sleepTill += dt;
                if (clock::now() > sleepTill) sleepTill = clock::now() + dt;
            }
        });
    }

    void run();

    void exit()
    {
        {
            std::lock_guard<std::mutex> lk(wake_mtx_);
            policy_thread_running = false;
        }
        wake_.notify_all();
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::mutex wake_mtx_;
    std::condition_variable wake_;

    std::atomic<bool> bad_{false}, started_{false};
    std::array<float, 29> q0_{}, kp0_{}, kd0_{}, tau0_{};
    std::chrono::steady_clock::time_point t_enter_;
    float blend_s_ = 0.5f;
};

REGISTER_FSM(State_RLBase)
