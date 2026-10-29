#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    env->alg->check(env->observation_manager->compute());   // the observation's size is the ONNX input's
    if (cfg["blend_s"]) blend_s_ = cfg["blend_s"].as<float>();

    const int passive = FSMStringMap.right.at("Passive");
    // another state's buttons wait for the sticks to be centred (not into a plan while still walking)
    for (auto& check : this->registered_checks) {
        if (check.second != passive) {
            auto f = check.first;
            check.first = [f]() -> bool {
                auto& j = FSMState::lowstate->joystick;
                return std::fabs(j.lx()) < 0.1f && std::fabs(j.ly()) < 0.1f && std::fabs(j.rx()) < 0.1f && f();
            };
        }
    }
    // Passive beyond limit_angle (rad) of tilt: isaaclab::mdp::bad_orientation is a stub (always false)
    float limit_angle = cfg["limit_angle"] ? cfg["limit_angle"].as<float>() : 1.0f;
    this->registered_checks.emplace_back(
        std::make_pair(
            [this, limit_angle]()->bool{
                const auto& g = env->robot->data.projected_gravity_b;
                return !g.allFinite() || std::acos(std::clamp(-g[2], -1.0f, 1.0f)) > limit_angle;
            },
            passive
        )
    );
    this->registered_checks.emplace_back([this]() -> bool { return bad_.load(); }, passive);
}

void State_RLBase::run()
{
    if (!started_) return;                          // no action yet: hold the command the state was entered with
    const auto action = env->action_manager->processed_actions();
    const auto& ids = env->robot->data.joint_ids_map;
    const auto& kp = env->robot->data.joint_stiffness;
    const auto& kd = env->robot->data.joint_damping;
    for (int i = 0; i < ids.size(); ++i) {
        if (!std::isfinite(action[i])) { bad_ = true; return; }   // nothing written; the check goes to Passive
    }
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_enter_).count();
    const float s = blend_s_ > 0 ? 0.5f - 0.5f * std::cos(M_PI * std::clamp(t / blend_s_, 0.0, 1.0)) : 1.0f;
    for (int i = 0; i < ids.size(); ++i) {
        const int slot = static_cast<int>(ids[i]);
        auto& m = lowcmd->msg_.motor_cmd()[slot];
        m.q() = q0_[slot] + (action[i] - q0_[slot]) * s;
        m.kp() = kp0_[slot] + (kp[i] - kp0_[slot]) * s;
        m.kd() = kd0_[slot] + (kd[i] - kd0_[slot]) * s;
        m.tau() = tau0_[slot] * (1.0f - s);
    }
}
