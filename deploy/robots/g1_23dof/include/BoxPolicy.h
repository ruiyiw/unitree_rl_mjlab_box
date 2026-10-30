#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "../config/policy/box/v1/params/PolicyConfig.h"

namespace box_policy {
using Action = std::array<float,23>;
using Observation = std::array<float,75>;
struct Sensors {
    std::array<float,4> quaternion{}; // Pelvis IMU, wxyz.
    std::array<float,3> gyro{};
    std::array<float,29> position{}, velocity{}; // SDK motor IDs.
};

inline std::vector<char> verified_file(const std::string& path, uint64_t expected) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open box artifact: " + path);
    std::vector<char> bytes{std::istreambuf_iterator<char>(stream), {}};
    uint64_t hash=14695981039346656037ULL;
    for (unsigned char byte:bytes) { hash^=byte; hash*=1099511628211ULL; }
    // SHA256 provenance is also recorded in manifest.json. This compiled digest
    // prevents accidental actor/reference swaps, including shape-compatible ones.
    if (hash!=expected) throw std::runtime_error("Box artifact identity mismatch: " + path);
    return bytes;
}

inline Observation observation(const Sensors& sensors, const Action& previous) {
    Observation obs{};
    float norm=0;
    for (float v:sensors.quaternion) norm+=v*v;
    if (!std::isfinite(norm) || norm<1e-8f) throw std::runtime_error("Invalid pelvis IMU quaternion");
    const float inv=1/std::sqrt(norm);
    const float w=sensors.quaternion[0]*inv, x=sensors.quaternion[1]*inv;
    const float y=sensors.quaternion[2]*inv, z=sensors.quaternion[3]*inv;
    std::copy(sensors.gyro.begin(),sensors.gyro.end(),obs.begin());
    obs[3]=2*(w*y-x*z); obs[4]=-2*(y*z+w*x); obs[5]=-1+2*(x*x+y*y);
    for (size_t i=0;i<23;++i) {
        obs[6+i]=sensors.position[kMotorIds[i]]-kDefault[i];
        obs[29+i]=sensors.velocity[kMotorIds[i]];
        obs[52+i]=previous[i];
    }
    for (float v:obs) if (!std::isfinite(v)) throw std::runtime_error("Non-finite box observation");
    return obs;
}

inline bool settled(const Observation& obs, int stage) {
    if (std::sqrt(obs[0]*obs[0]+obs[1]*obs[1]+obs[2]*obs[2])>=.25f ||
        obs[5]>=-std::cos(15.0*3.141592653589793/180)) return false;
    for (size_t i=0;i<23;++i)
        if (std::abs(obs[6+i]-(kEnds[stage][i]-kDefault[i]))>=.3f || std::abs(obs[29+i])>=.5f) return false;
    return true;
}

template<size_t Width>
class ModelActor {
public:
    ModelActor(const std::string& path, uint64_t expected_hash):runtime_(ORT_LOGGING_LEVEL_WARNING,"g1_box") {
        const auto bytes=verified_file(path,expected_hash);
        options_.SetIntraOpNumThreads(1); options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_=std::make_unique<Ort::Session>(runtime_,bytes.data(),bytes.size(),options_);
        if (session_->GetInputCount()!=1 || session_->GetOutputCount()!=1)
            throw std::runtime_error("Expected one box actor input/output");
        const auto it=session_->GetInputTypeInfo(0), ot=session_->GetOutputTypeInfo(0);
        const auto in=it.GetTensorTypeAndShapeInfo(), out=ot.GetTensorTypeAndShapeInfo();
        if (in.GetShape()!=std::vector<int64_t>{1,Width} || out.GetShape()!=std::vector<int64_t>{1,23} ||
            in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unexpected box actor dimensions or tensor type");
        Ort::AllocatorWithDefaultOptions alloc;
        if (std::string(session_->GetInputNameAllocated(0,alloc).get())!="observations" ||
            std::string(session_->GetOutputNameAllocated(0,alloc).get())!="actions")
            throw std::runtime_error("Unexpected box tensor names");
    }
    Action infer(const std::array<float,Width>& obs) {
        for (float v:obs) if (!std::isfinite(v)) throw std::runtime_error("Non-finite box observation");
        auto info=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        const std::array<int64_t,2> shape{1,Width};
        auto input=Ort::Value::CreateTensor<float>(info,const_cast<float*>(obs.data()),Width,shape.data(),2);
        const char* ins[]{"observations"}; const char* outs[]{"actions"};
        auto result=session_->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
        const float* data=result[0].GetTensorData<float>();
        Action action{};
        for (size_t i=0;i<23;++i) {
            if (!std::isfinite(data[i])) throw std::runtime_error("Non-finite box actor output");
            action[i]=data[i]; // Residuals are intentionally NOT clipped.
        }
        return action;
    }
private:
    Ort::Env runtime_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};

class Actor : public ModelActor<75> {
public:
    Actor(const std::string& path,int stage):ModelActor(path,kActorHashes.at(stage)) {}
};

struct Step {
    Observation observation;
    Action action, target;
    int stage, clock;
    bool done; // Sensor handoff complete, not a physical box-success assertion.
};

class Controller {
public:
    // -1 selects the continuous sequence; 0..4 selects an isolated stage.
    Controller(const std::string& directory,int selected):selected_(selected) {
        if (selected < -1 || selected>4) throw std::runtime_error("Invalid box stage");
        for (int i=0;i<5;++i) {
            if (selected!=-1 && selected!=i) continue;
            const auto folder=directory+"/"+kNames[i];
            actors_[i]=std::make_unique<Actor>(folder+"/policy.onnx",i);
            const auto bytes=verified_file(folder+"/reference.bin",kReferenceHashes[i]);
            if (bytes.size()!=4+size_t(kLengths[i])*23*4) throw std::runtime_error("Invalid reference size");
            uint32_t length=0;
            std::memcpy(&length,bytes.data(),4);
            if (length!=uint32_t(kLengths[i])) throw std::runtime_error("Invalid reference length/endianness");
            references_[i].resize(length);
            std::memcpy(references_[i].data(),bytes.data()+4,bytes.size()-4);
        }
        reset();
    }
    void reset() {
        stage_=selected_==-1?0:selected_; clock_=0; elapsed_=0; stood_=0;
        fresh_=false; first_=true; done_=false; previous_.fill(0);
    }
    // A stage entered straight from the previous one's settled endpoint keeps
    // that actor's last residual, as the continuous sequence's handoff does.
    void reset(const Action& previous) { reset(); previous_=previous; }
    const Action& previous() const { return previous_; }
    Step step(const Sensors& sensors) {
        auto obs=observation(sensors,previous_);
        // Update the command using the NEW sensors after the previous action.
        // The source reset has already consumed its initial fresh flag. At a
        // handoff fresh stays true for one step, repeating qcmd[1] once.
        if (!first_) {
            if (!fresh_) { clock_=std::min(clock_+1,kLengths[stage_]-1); elapsed_+=kStepDt; }
            fresh_=false;
            stood_=(clock_==kLengths[stage_]-1 && settled(obs,stage_))?stood_+kStepDt:0.f;
            if (stood_>=1.f) {
                if (selected_==-1 && stage_<4) {
                    ++stage_; clock_=0; elapsed_=0; stood_=0; fresh_=true;
                } else done_=true;
            }
            if (!done_ && elapsed_>(kLengths[stage_]-1)*kStepDt+12.f)
                throw std::runtime_error("Box stage did not settle before timeout");
        }
        first_=false;
        const auto action=actors_[stage_]->infer(obs);
        Action target{};
        const auto& reference=references_[stage_][std::min(clock_+1,kLengths[stage_]-1)];
        for (size_t i=0;i<23;++i) {
            target[i]=reference[i]+kScale[i]*action[i];
            if (!std::isfinite(target[i])) throw std::runtime_error("Non-finite box target");
        }
        previous_=action; // Preserve unclipped residual across every handoff.
        return {obs,action,target,stage_,clock_,done_};
    }
private:
    int selected_,stage_=0,clock_=0;
    float elapsed_=0,stood_=0;
    bool fresh_=false,first_=true,done_=false;
    Action previous_{};
    std::array<std::unique_ptr<Actor>,5> actors_;
    std::array<std::vector<Action>,5> references_;
};
} // namespace box_policy
