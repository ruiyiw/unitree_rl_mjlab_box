// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <mutex>
#include <stdexcept>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
        // optional (not in unitree_rl_mjlab): each step's change of each raw action held to rate_limit, the
        // policy's healthy rate with a margin (demos/action_rates.py); the limited action is the one observed
        if(cfg["rate_limit"] && !cfg["rate_limit"].IsNull()) {
            _rate_limit = cfg["rate_limit"].as<std::vector<float>>();
            if(_rate_limit.size() != _action_dim) throw std::runtime_error("JointAction: rate_limit has the wrong size");
        }
        if(!_clip.empty() && _clip.size() != _action_dim) throw std::runtime_error("JointAction: clip has the wrong size");
        reset();                                    // the first target is the default pose, never zeros
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        if(!_rate_limit.empty()) {
            for(int i(0); i<_action_dim; ++i) {
                actions[i] = _raw_actions[i] + std::clamp(actions[i] - _raw_actions[i], -_rate_limit[i], _rate_limit[i]);
            }
        }
        _raw_actions = actions;
        std::vector<float> out(_action_dim);
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                out[i] = _raw_actions[i] * _scale[i];
            } else {
                out[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                out[i] += _offset[i];
            }
        }
        if(!_clip.empty())                          // the joints' ranges (export.py): an OOD target goes no further
        {
            for(int i(0); i<_action_dim; ++i) {
                out[i] = std::clamp(out[i], _clip[i][0], _clip[i][1]);
            }
        }
        std::lock_guard<std::mutex> lock(_mtx);
        _processed_actions = out;
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        std::lock_guard<std::mutex> lock(_mtx);    // the FSM thread reads it while the policy thread writes it
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
        process_actions(std::vector<float>(_action_dim, 0.0f));   // the default pose (offset), clipped
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
    std::vector<float> _rate_limit;
    std::mutex _mtx;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};