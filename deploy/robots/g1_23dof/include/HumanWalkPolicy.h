#pragma once
// A person's walk: the robot tracks one reference motion (a motion-capture walk of 2.2 m,
// retargeted to the 23-DoF G1) and stops.
//
//   target = reference qcmd[min(clock + 1, last)] + scale * actor(observation)
//
// The actor was trained on the simulator's torso pose and IMU velocity.  The robot has no
// sensor for either, so both are estimated here, as unitree_human_walk/estimate.py does:
//   pelvis x, y and their rates   the walk controller's Kalman leg odometry
//   pelvis height and its rate    minus the height of the lowest sole point under the pelvis
//   orientation                   the IMU's, with its heading measured from the walk's start
//   torso                         the pelvis and the waist encoder
// Training compensated the arms' gravity; the arm motors do it here (the box-three torque).
//
// This header has no DDS, YAML or logging dependency, so the offline tests run the same code.
#include "ParametrizedWalkPolicy.h"   // Sensors, the position filter and the leg kinematics, small matrix helpers
#include "TableBoxPolicy.h"           // the arm motors' gravity-compensation torque
#include "../config/policy/human_walk/v1/params/PolicyConfig.h"
#include <limits>

namespace human_walk {
using parametrized_walk::Action;
using parametrized_walk::Mat3;
using parametrized_walk::Sensors;
using parametrized_walk::Vec2;
using parametrized_walk::Vec3;
using Observation = std::array<float,145>;
// ref_joints 46, ref_anchor 9, ref_preview 10, ref_clock 2, base_lin_vel 3, base_ang_vel 3,
// projected_gravity 3, joint_pos 23, joint_vel 23, actions 23
inline constexpr int kAnchor=46,kPreview=55,kClock=65,kLinVel=67,kGyro=70,kGravity=73,kJointPos=76,kJointVel=99,kPrevious=122;

// The robot is the one the walk and box controllers drive: the same motors, pose, scales and gains.
template<class T,size_t N> constexpr bool same(const std::array<T,N>& a,const std::array<T,N>& b) {
    for(size_t i=0;i<N;++i) if(a[i]!=b[i]) return false;
    return true;
}
static_assert(same(kMotorIds,parametrized_walk::kMotorIds) && same(kDefault,parametrized_walk::kDefault) &&
              same(kScale,parametrized_walk::kScale) && same(kKp,parametrized_walk::kKp) && same(kKd,parametrized_walk::kKd));
static_assert(same(kMotorIds,box_policy::kMotorIds) && kStepDt==parametrized_walk::kStepDt);

struct Frame { Action qcmd,q,dq; Vec3 torso; std::array<float,4> torso_q; float x,y; };
static_assert(sizeof(Frame)==78*sizeof(float));

inline Vec3 turn_back(const Mat3& a,const Vec3& v) { // transpose(a) * v
    return {a[0]*v[0]+a[3]*v[1]+a[6]*v[2],a[1]*v[0]+a[4]*v[1]+a[7]*v[2],a[2]*v[0]+a[5]*v[1]+a[8]*v[2]};
}
inline Mat3 transposed(const Mat3& a) { return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]}; }

// The lowest sole point under the pelvis, in the gravity-aligned frame: the leg chains of the
// walk controller's odometry, carried to the foot bodies.
inline float lowest_sole(const Sensors& sensors,const Mat3& attitude) {
    float low=std::numeric_limits<float>::infinity();
    for(int side=0;side<2;++side) {
        Vec3 p{}; Mat3 r{1,0,0,0,1,0,0,0,1};
        for(const auto& link:parametrized_walk::kLegs[side].links) {
            const float angle=sensors.position[kMotorIds[link.joint]];
            if(!std::isfinite(angle)) throw std::runtime_error("Non-finite leg joint position");
            const auto offset=parametrized_walk::turn_by(r,link.position);
            for(int i=0;i<3;++i) p[i]+=offset[i];
            r=parametrized_walk::mul(r,link.rotation);
            const auto turn=parametrized_walk::about_axis(link.axis,angle);
            const auto turned=parametrized_walk::turn_by(turn,link.anchor);
            const auto shift=parametrized_walk::turn_by(r,Vec3{link.anchor[0]-turned[0],link.anchor[1]-turned[1],link.anchor[2]-turned[2]});
            for(int i=0;i<3;++i) p[i]+=shift[i];
            r=parametrized_walk::mul(r,turn);
        }
        const auto& sole=side==0?kLeftSole:kRightSole;
        for(int k=0;k<kSolePoints;++k) {
            const auto point=parametrized_walk::turn_by(r,Vec3{sole[3*k],sole[3*k+1],sole[3*k+2]});
            const auto world=parametrized_walk::turn_by(attitude,Vec3{p[0]+point[0],p[1]+point[1],p[2]+point[2]});
            low=std::min(low,world[2]);
        }
    }
    return low;
}

struct Step {
    Observation observation;
    Action action,target,torque;   // torque: the arm motors' gravity compensation (zero elsewhere)
    int clock;
    bool ended;        // the reference has played to its last frame
    bool done;         // ... and the robot has stood settled for one second since (sensors only)
    Vec3 position;     // the pelvis estimate in the walk's frame: along the route, to its left, height (m)
    Vec3 velocity;     // its rate (m/s)
    float heading;     // rad, from the heading at the start
};

class Controller {
public:
    explicit Controller(const std::string& directory):actor_(directory+"/policy.onnx",kActorHash) {
        const auto bytes=box_policy::verified_file(directory+"/reference.bin",kReferenceHash);
        uint32_t length=0,width=0;
        if(bytes.size()<16 || std::memcmp(bytes.data(),"G1HWALK1",8)) throw std::runtime_error("Invalid human-walk reference header");
        std::memcpy(&length,bytes.data()+8,4); std::memcpy(&width,bytes.data()+12,4);
        if(length!=uint32_t(kFrames) || width!=78 || bytes.size()!=16+size_t(length)*sizeof(Frame))
            throw std::runtime_error("Invalid human-walk reference dimensions");
        frames_.resize(length); std::memcpy(frames_.data(),bytes.data()+16,bytes.size()-16);
        for(const auto& f:frames_) for(size_t i=0;i<78;++i)
            if(!std::isfinite(reinterpret_cast<const float*>(&f)[i])) throw std::runtime_error("Non-finite human-walk reference");
        start(0);
    }
    // Begin the walk from where the robot stands and the way it faces at the next step().
    // The reference's first frame is held for hold_s (while an entry blend settles), and for
    // one step more: training observes the first frame twice (mjlab does not advance the
    // clock on the step after a reset).
    void start(float hold_s) {
        if(!(hold_s>=0) || hold_s>10) throw std::runtime_error("Invalid human-walk hold time");
        clock_=0; holds_=int(std::lround(hold_s/kStepDt))+1; first_=true; stood_=0; done_=false; previous_.fill(0);
    }
    float seconds() const { return kPlanSeconds; }
    float scale=1;      // true distance per kinematic distance, from a calibration walk
    // The estimate measured from the walk's start: metres along the route, metres to its left.
    Vec2 walked() const { return {estimator_.position()[0]-frames_[0].x,estimator_.position()[1]-frames_[0].y}; }
    Step step(const Sensors& sensors) {
        const int last=kFrames-1;
        const auto q=parametrized_walk::normalized(sensors.quaternion);
        Mat3 attitude{};
        if(first_) {
            // The walk's frame: the origin under the pelvis now, x the way the robot faces now.
            heading_origin_=parametrized_walk::yaw(q);
            attitude=parametrized_walk::mul(parametrized_walk::about_z(-heading_origin_),parametrized_walk::rotation(q));
            estimator_.scale=scale;
            estimator_.reset(sensors,attitude,{frames_[0].x,frames_[0].y});
            height_=-lowest_sole(sensors,attitude); rise_=0;
        } else {
            if(holds_>0) --holds_; else clock_=std::min(clock_+1,last);
            attitude=parametrized_walk::mul(parametrized_walk::about_z(-heading_origin_),parametrized_walk::rotation(q));
            estimator_.update(sensors,attitude);
            const float height=-lowest_sole(sensors,attitude);
            rise_=(height-height_)/kStepDt; height_=height;
        }
        const Vec3 pelvis{estimator_.position()[0],estimator_.position()[1],height_};
        const Vec3 velocity{estimator_.velocity()[0],estimator_.velocity()[1],rise_};
        // the torso link: the pelvis and the waist joint
        const float waist=sensors.position[kMotorIds[kWaist]];
        const Mat3 base=parametrized_walk::mul(attitude,kTorsoRotation);
        const Mat3 turn=parametrized_walk::about_axis(kWaistAxis,waist);
        const auto turned=parametrized_walk::turn_by(turn,kWaistAnchor);
        const auto offset=parametrized_walk::turn_by(attitude,kTorsoPosition);
        const auto shift=parametrized_walk::turn_by(base,Vec3{kWaistAnchor[0]-turned[0],kWaistAnchor[1]-turned[1],kWaistAnchor[2]-turned[2]});
        const Vec3 torso{pelvis[0]+offset[0]+shift[0],pelvis[1]+offset[1]+shift[1],pelvis[2]+offset[2]+shift[2]};
        const Mat3 torso_r=parametrized_walk::mul(base,turn);

        Observation obs{};
        const auto& now=frames_[clock_];
        for(size_t i=0;i<23;++i) {
            obs[i]=now.q[i]-kDefault[i]; obs[23+i]=now.dq[i];
            obs[kJointPos+i]=sensors.position[kMotorIds[i]]-kDefault[i];
            obs[kJointVel+i]=sensors.velocity[kMotorIds[i]];
            obs[kPrevious+i]=previous_[i];
        }
        // the reference's torso in the robot's torso frame: position, and two columns of its orientation
        const auto apart=turn_back(torso_r,Vec3{now.torso[0]-torso[0],now.torso[1]-torso[1],now.torso[2]-torso[2]});
        const Mat3 relative=parametrized_walk::mul(transposed(torso_r),parametrized_walk::rotation(now.torso_q));
        std::copy(apart.begin(),apart.end(),obs.begin()+kAnchor);
        for(int row=0;row<3;++row) for(int col=0;col<2;++col) obs[kAnchor+3+2*row+col]=relative[3*row+col];
        // the reference's torso 0.2 s and 0.5 s ahead, in the robot's heading frame
        const float torso_yaw=std::atan2(torso_r[3],torso_r[0]),c=std::cos(torso_yaw),s=std::sin(torso_yaw);
        int at=kPreview;
        for(int ahead:{10,25}) {
            const auto& future=frames_[std::min(clock_+ahead,last)];
            const float dx=future.torso[0]-torso[0],dy=future.torso[1]-torso[1];
            const float error=parametrized_walk::wrap(parametrized_walk::yaw(future.torso_q)-torso_yaw);
            obs[at++]=c*dx+s*dy; obs[at++]=-s*dx+c*dy; obs[at++]=future.torso[2]-torso[2];
            obs[at++]=std::sin(error); obs[at++]=std::cos(error);
        }
        obs[kClock]=float(clock_)/float(kFrames); obs[kClock+1]=clock_==last?1.f:0.f;
        // what a velocimeter at the IMU would read
        const auto body=turn_back(attitude,velocity);
        const auto& w=sensors.gyro;
        obs[kLinVel]=body[0]+w[1]*kImu[2]-w[2]*kImu[1];
        obs[kLinVel+1]=body[1]+w[2]*kImu[0]-w[0]*kImu[2];
        obs[kLinVel+2]=body[2]+w[0]*kImu[1]-w[1]*kImu[0];
        std::copy(w.begin(),w.end(),obs.begin()+kGyro);
        obs[kGravity]=2*(q[0]*q[2]-q[1]*q[3]); obs[kGravity+1]=-2*(q[2]*q[3]+q[0]*q[1]); obs[kGravity+2]=-1+2*(q[1]*q[1]+q[2]*q[2]);
        if(!first_) {
            stood_=(clock_==last && settled(obs))?stood_+kStepDt:0.f;
            if(stood_>=1.f) done_=true; // latched, as the box controllers do
        }
        first_=false;
        const auto action=actor_.infer(obs);
        const auto& next=frames_[std::min(clock_+1,last)];
        Action target{};
        for(size_t i=0;i<23;++i) {
            target[i]=next.qcmd[i]+kScale[i]*action[i];
            if(!std::isfinite(target[i])) throw std::runtime_error("Non-finite human-walk target");
        }
        previous_=action; // not clipped, as in training
        box_policy::Sensors arm{};
        arm.quaternion=sensors.quaternion; arm.gyro=sensors.gyro; arm.position=sensors.position; arm.velocity=sensors.velocity;
        return {obs,action,target,table_box::gravity_torque(arm),clock_,clock_==last,done_,pelvis,velocity,
                parametrized_walk::wrap(parametrized_walk::yaw(q)-heading_origin_)};
    }
    const Frame& frame(int i) const { return frames_.at(i); }
    // The correction of the last step, which the actor reads back.  The offline comparison with
    // training sets it to training's, so that each step is checked on identical inputs.
    const Action& previous() const { return previous_; }
    void previous(const Action& action) { previous_=action; }
private:
    // Standing still: the pelvis barely turning and within 15 degrees of upright, no joint
    // moving.  No pose test: the policy stands at the end in its own posture.
    static bool settled(const Observation& obs) {
        if(std::sqrt(obs[kGyro]*obs[kGyro]+obs[kGyro+1]*obs[kGyro+1]+obs[kGyro+2]*obs[kGyro+2])>=.25f ||
           obs[kGravity+2]>=-std::cos(15.f*3.141592653589793f/180)) return false;
        for(size_t i=0;i<23;++i) if(std::abs(obs[kJointVel+i])>=.5f) return false;
        return true;
    }
    box_policy::ModelActor<145> actor_;
    std::vector<Frame> frames_;
    parametrized_walk::Estimator estimator_;
    Action previous_{};
    int clock_=0,holds_=1;
    float heading_origin_=0,height_=0,rise_=0,stood_=0;
    bool first_=true,done_=false;
};
} // namespace human_walk
