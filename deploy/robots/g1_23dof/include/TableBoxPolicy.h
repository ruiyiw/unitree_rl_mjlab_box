#pragma once
#include "BoxPolicy.h"
#include "../config/policy/box_three/v1/params/PolicyConfig.h"

namespace table_box {
using box_policy::Action;
using box_policy::Sensors;
using Observation = std::array<float,376>;
using Vec = std::array<float,3>;
using Quat = std::array<float,4>;
inline Vec add(Vec a,const Vec& b) { for(int i=0;i<3;++i) a[i]+=b[i]; return a; }
inline Vec sub(Vec a,const Vec& b) { for(int i=0;i<3;++i) a[i]-=b[i]; return a; }
inline Vec mul(Vec a,float s) { for(float& v:a) v*=s; return a; }
inline float dot(const Vec& a,const Vec& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec cross(const Vec& a,const Vec& b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline Quat normalized(Quat q) {
    float n=0; for(float v:q) n+=v*v;
    if(!std::isfinite(n) || n<1e-8f) throw std::runtime_error("Invalid table-box IMU quaternion");
    for(float& v:q) v/=std::sqrt(n); return q;
}
inline Quat product(const Quat& a,const Quat& b) {
    Vec av{a[1],a[2],a[3]},bv{b[1],b[2],b[3]};
    const auto v=add(add(mul(bv,a[0]),mul(av,b[0])),cross(av,bv));
    return {a[0]*b[0]-dot(av,bv),v[0],v[1],v[2]};
}
inline Vec rotate(const Quat& q,const Vec& v) {
    const Vec u{q[1],q[2],q[3]}; const auto t=mul(cross(u,v),2);
    return add(v,add(mul(t,q[0]),cross(u,t)));
}
inline float yaw(const Quat& q) {return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));}
inline float wrap(float a) {return std::atan2(std::sin(a),std::cos(a));}

// Nominal actuator_only compensation from the training model, applied only to
// the ten arm motor joints. Evaluate from measured joints and pelvis IMU.
inline Action gravity_torque(const Sensors& sensors) {
    std::array<Vec,kBodies.size()> positions{},axes{},anchors{},coms{};
    std::array<Quat,kBodies.size()> rotations{};
    rotations[0]={1,0,0,0};
    for(size_t b=1;b<kBodies.size();++b) {
        const auto& body=kBodies[b];
        if(b==1) { rotations[b]=normalized(sensors.quaternion); continue; }
        const auto parent=rotations[body.parent];
        auto rot=product(parent,body.quaternion);
        auto pos=add(positions[body.parent],rotate(parent,body.position));
        if(body.motor>=0) {
            const float angle=sensors.position[box_policy::kMotorIds[body.motor]];
            if(!std::isfinite(angle)) throw std::runtime_error("Non-finite gravity-compensation joint");
            anchors[b]=add(pos,rotate(rot,body.anchor)); axes[b]=rotate(rot,body.axis);
            const float s=std::sin(angle/2);
            rot=product(rot,{std::cos(angle/2),body.axis[0]*s,body.axis[1]*s,body.axis[2]*s});
            pos=sub(anchors[b],rotate(rot,body.anchor));
        }
        positions[b]=pos; rotations[b]=rot; coms[b]=add(pos,rotate(rot,body.com));
    }
    Action result{};
    for(size_t b=1;b<kBodies.size();++b) if(kBodies[b].mass>0) {
        const Vec force{0,0,9.81f*kBodies[b].mass};
        for(int j=b;j>1;j=kBodies[j].parent) {
            const int motor=kBodies[j].motor;
            if(motor>=13) result[motor]+=dot(axes[j],cross(sub(coms[b],anchors[j]),force));
        }
    }
    return result;
}

struct Frame { std::array<float,30> qpos; std::array<float,29> qvel; Action qcmd; Vec grip; };
static_assert(sizeof(Frame)==85*sizeof(float));
struct Step { Observation observation; Action action,target,torque; int clock; bool done; };
class Controller {
public:
    Controller(const std::string& directory,int stage):stage_(stage),
        actor_(directory+"/"+std::string(kNames.at(stage))+"/policy.onnx",kActorHashes.at(stage)) {
        const auto bytes=box_policy::verified_file(directory+"/"+kNames[stage]+"/reference.bin",kReferenceHashes[stage]);
        uint32_t length=0,width=0;
        if(bytes.size()<8) throw std::runtime_error("Invalid table-box reference header");
        std::memcpy(&length,bytes.data(),4); std::memcpy(&width,bytes.data()+4,4);
        if(length!=uint32_t(kLengths[stage]) || width!=85 || bytes.size()!=8+length*sizeof(Frame))
            throw std::runtime_error("Invalid table-box reference dimensions");
        frames_.resize(length); std::memcpy(frames_.data(),bytes.data()+8,bytes.size()-8);
        reset();
    }
    void reset() {
        clock_=0; first_=true; stood_=0; elapsed_=0; done_=false; previous_.fill(0);
        // mjlab's explicit reset scopes command updates to env_ids, leaving
        // fresh=True. Restored take/put-down entries re-arm it once more in
        // HandoffCommand. Match those nominal per-stage entry timelines.
        startup_holds_=stage_==0?1:2;
    }
    void reset(const Action& previous) {reset(); previous_=previous;}
    const Action& previous() const {return previous_;}
    bool ended() const {return clock_==kLengths[stage_]-1;}
    // command is the actual preceding position target (after the FSM blend).
    Step step(const Sensors& sensors,const Action& command) {
        auto frame=box_policy::observation(sensors,previous_);
        const auto orientation=normalized(sensors.quaternion);
        if(first_) {
            heading_origin_=yaw(orientation)-reference_yaw(frames_[0]);
            std::fill(frame.begin()+52,frame.end(),0.f); // training resets sensor history, even at a handoff
            history_.fill(frame);
        } else {
            if(startup_holds_>0) --startup_holds_;
            else clock_=std::min(clock_+1,kLengths[stage_]-1);
            elapsed_+=box_policy::kStepDt;
            for(int i=0;i<3;++i) history_[i]=history_[i+1]; history_[3]=frame;
            stood_=(ended() && settled(frame))?stood_+box_policy::kStepDt:0.f;
            if(stood_>=1.f) done_=true; // Latch completion while holding, as the previous box controller does.
            if(!done_ && elapsed_>(kLengths[stage_]-1)*box_policy::kStepDt+12.f)
                throw std::runtime_error("Table-box stage did not settle before timeout");
        }
        first_=false;
        Observation obs{};
        const auto& ref=frames_[clock_];
        for(size_t i=0;i<23;++i) {obs[i]=ref.qpos[7+i]-box_policy::kDefault[i]; obs[23+i]=ref.qvel[6+i];}
        obs[46]=float(clock_)/(kLengths[stage_]-1); obs[47]=ended()?1.f:0.f;
        const float heading=wrap(yaw(orientation)-heading_origin_);
        int offset=48;
        for(int ahead:{0,10,25}) {
            const auto& future=frames_[std::min(clock_+ahead,kLengths[stage_]-1)];
            const float dx=future.qpos[0]-ref.qpos[0],dy=future.qpos[1]-ref.qpos[1];
            obs[offset++]=std::cos(heading)*dx+std::sin(heading)*dy;
            obs[offset++]=-std::sin(heading)*dx+std::cos(heading)*dy;
            const float error=wrap(reference_yaw(future)-heading);
            obs[offset++]=std::sin(error); obs[offset++]=std::cos(error);
        }
        const auto q=reference_quat(ref);
        const auto g=rotate({q[0],-q[1],-q[2],-q[3]},{0,0,-1});
        for(float v:g) obs[offset++]=v;
        for(float v:ref.grip) obs[offset++]=v;
        for(size_t i=13;i<23;++i) obs[offset++]=command[i]-sensors.position[box_policy::kMotorIds[i]];
        for(const auto& old:history_) for(float v:old) obs[offset++]=v;
        const auto action=actor_.infer(obs);
        Action target{};
        for(size_t i=0;i<23;++i) {
            target[i]=frames_[std::min(clock_+1,kLengths[stage_]-1)].qcmd[i]+box_policy::kScale[i]*action[i];
            if(!std::isfinite(target[i])) throw std::runtime_error("Non-finite table-box target");
        }
        previous_=action;
        return {obs,action,target,gravity_torque(sensors),clock_,done_};
    }
private:
    static Quat reference_quat(const Frame& f) {return {f.qpos[3],f.qpos[4],f.qpos[5],f.qpos[6]};}
    static float reference_yaw(const Frame& f) {return yaw(reference_quat(f));}
    bool settled(const box_policy::Observation& obs) const {
        if(std::sqrt(obs[0]*obs[0]+obs[1]*obs[1]+obs[2]*obs[2])>=.25f ||
           obs[5]>=-std::cos(15.f*3.141592653589793f/180)) return false;
        for(size_t i=0;i<23;++i)
            if(std::abs(obs[6+i]-(kEnds[stage_][i]-box_policy::kDefault[i]))>=.3f || std::abs(obs[29+i])>=.5f) return false;
        return true;
    }
    int stage_,clock_=0,startup_holds_=0;
    box_policy::ModelActor<376> actor_;
    std::vector<Frame> frames_;
    std::array<box_policy::Observation,4> history_{};
    Action previous_{};
    float heading_origin_=0,stood_=0,elapsed_=0;
    bool first_=true,done_=false;
};
} // namespace table_box
