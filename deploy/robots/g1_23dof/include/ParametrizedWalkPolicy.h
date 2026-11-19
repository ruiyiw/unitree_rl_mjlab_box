#pragma once
// walk(N): walk N metres straight ahead and stop.
//
// A planned walk of N metres (from the library, plans.bin) gives the joint targets; the actor
// adds a correction to them from the robot's own sensors and from where the robot believes it
// is along the plan.  That belief is leg odometry fused with the IMU's accelerometer (the
// training code's KalmanOdometry): no position sensor is read.
//
//   target = plan qcmd[min(clock + 1, last)] + scale * actor(observation)
//
// The actor reads the last kHistory readings of the five sensor terms (each term's readings
// together, oldest first, as the training env stacks them) and the plan and goal terms once.
//
// This header has no DDS, YAML or logging dependency, so the offline tests run the same code.
#include "BoxPolicy.h"
#include "../config/policy/parametrized_walk/v3/params/PolicyConfig.h"
#include <map>
#include <mutex>

namespace parametrized_walk {
using Action = std::array<float,23>;
using Observation = std::array<float,109>;          // one frame: 75 sensor numbers, then 34 plan and goal numbers
using SensorFrame = std::array<float,75>;
using ActorInput = std::array<float,kActorInputs>; // kHistory sensor frames by term, then the 34
// first index and width of gyro, gravity, joint positions, joint velocities, previous actions
inline constexpr std::array<std::array<int,2>,5> kSensorTerms{{{0,3},{3,3},{6,23},{29,23},{52,23}}};
using Vec2 = std::array<float,2>;
using Vec3 = std::array<float,3>;
using Mat3 = std::array<float,9>; // row-major

struct Sensors {
    std::array<float,4> quaternion{};  // pelvis IMU, wxyz
    Vec3 gyro{};                       // rad/s, pelvis frame
    Vec3 accelerometer{};              // m/s^2 specific force, pelvis frame (+9.81 on z at rest)
    std::array<float,29> position{}, velocity{}; // SDK motor IDs
};

inline Mat3 mul(const Mat3& a,const Mat3& b) {
    Mat3 c{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) c[3*i+j]+=a[3*i+k]*b[3*k+j];
    return c;
}
inline Vec3 turn_by(const Mat3& a,const Vec3& v) {
    return {a[0]*v[0]+a[1]*v[1]+a[2]*v[2],a[3]*v[0]+a[4]*v[1]+a[5]*v[2],a[6]*v[0]+a[7]*v[1]+a[8]*v[2]};
}
inline Mat3 rotation(const std::array<float,4>& q) {
    const float w=q[0],x=q[1],y=q[2],z=q[3];
    return {1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
            2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
}
inline Mat3 about_z(float angle) {
    const float c=std::cos(angle),s=std::sin(angle);
    return {c,-s,0,s,c,0,0,0,1};
}
inline Mat3 about_axis(const Vec3& u,float angle) { // Rodrigues, unit axis
    const Mat3 k{0,-u[2],u[1],u[2],0,-u[0],-u[1],u[0],0};
    const Mat3 kk=mul(k,k);
    const float s=std::sin(angle),c=std::cos(angle);
    Mat3 r{1,0,0,0,1,0,0,0,1};
    for(int i=0;i<9;++i) r[i]+=s*k[i]+(1-c)*kk[i];
    return r;
}
inline std::array<float,4> normalized(std::array<float,4> q) {
    float n=0; for(float v:q) n+=v*v;
    if(!std::isfinite(n) || n<1e-8f) throw std::runtime_error("Invalid pelvis IMU quaternion");
    for(float& v:q) v/=std::sqrt(n);
    return q;
}
inline float yaw(const std::array<float,4>& q) {
    return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));
}
inline float wrap(float a) { return std::atan2(std::sin(a),std::cos(a)); }

// ------------------------------------------------------------------ the planned walks
struct Frame { Action qcmd; float x,y,sin_phase,cos_phase,walking; };
static_assert(sizeof(Frame)==28*sizeof(float));
struct Plan { uint32_t millimetres,length,first; Action end; }; // end: the plan's last joint pose (for the record)
static_assert(sizeof(Plan)==12+23*sizeof(float));

class Library {
public:
    explicit Library(const std::string& directory) {
        const auto bytes=box_policy::verified_file(directory+"/plans.bin",kLibraryHash);
        uint32_t count=0,width=0;
        if(bytes.size()<16 || std::memcmp(bytes.data(),"G1WALKN1",8)) throw std::runtime_error("Invalid walk library header");
        std::memcpy(&count,bytes.data()+8,4); std::memcpy(&width,bytes.data()+12,4);
        const size_t table=size_t(count)*sizeof(Plan),body=size_t(kLibraryFrames)*sizeof(Frame);
        if(count!=uint32_t(kPlans) || width!=28 || bytes.size()!=16+table+body)
            throw std::runtime_error("Invalid walk library dimensions");
        plans_.resize(count); std::memcpy(plans_.data(),bytes.data()+16,table);
        frames_.resize(kLibraryFrames); std::memcpy(frames_.data(),bytes.data()+16+table,body);
        for(size_t i=0;i<plans_.size();++i) {
            const auto& p=plans_[i];
            if(p.length<2 || size_t(p.first)+p.length>frames_.size() || (i && p.millimetres<=plans_[i-1].millimetres))
                throw std::runtime_error("Invalid walk library table");
        }
        for(const auto& f:frames_) for(size_t i=0;i<28;++i)
            if(!std::isfinite(reinterpret_cast<const float*>(&f)[i])) throw std::runtime_error("Non-finite walk plan");
    }
    // Both policies read the same plans: one copy per process.
    static std::shared_ptr<const Library> shared(const std::string& directory) {
        static std::mutex mutex; static std::map<std::string,std::shared_ptr<const Library>> loaded;
        std::lock_guard<std::mutex> lock(mutex);
        auto& entry=loaded[directory];
        if(!entry) entry=std::make_shared<const Library>(directory);
        return entry;
    }
    size_t size() const { return plans_.size(); }
    const Plan& plan(size_t i) const { return plans_.at(i); }
    const Frame* frames(const Plan& p) const { return frames_.data()+p.first; }
    static float metres(const Plan& p) { return p.millimetres*.001f; }
    float shortest() const { return metres(plans_.front()); }
    float longest() const { return metres(plans_.back()); }
    // The planned distance closest to a request (the request must be finite).
    size_t nearest(float request) const {
        if(!std::isfinite(request)) throw std::runtime_error("Walk distance is not a number");
        size_t best=0;
        for(size_t i=1;i<plans_.size();++i)
            if(std::abs(metres(plans_[i])-request)<std::abs(metres(plans_[best])-request)) best=i;
        return best;
    }
private:
    std::vector<Plan> plans_;
    std::vector<Frame> frames_;
};

// ------------------------------------------------------------------ where the pelvis is
// The training code's KalmanOdometry, one robot: per horizontal axis the state is position,
// velocity and accelerometer bias.  The feet's kinematics give the pelvis velocity (a planted
// foot does not move), each foot weighted toward the lower one; a foot whose velocity
// disagrees with the prediction is trusted less.
class Estimator {
public:
    float scale=1;          // true distance per kinematic distance, from a calibration walk
    void reset(const Sensors& sensors,const Mat3& attitude,const Vec2& start) {
        position_=start; velocity_={0,0}; bias_={0,0};
        p_={kP0Position,0,0, 0,kP0Velocity,0, 0,0,kP0Bias};
        feet_=feet(sensors,attitude);
    }
    // attitude: the IMU's orientation with its heading measured from the walk's start.
    void update(const Sensors& sensors,const Mat3& attitude) {
        const float dt=kStepDt;
        const auto now=feet(sensors,attitude);
        std::array<float,2> weight{}; std::array<Vec2,2> measured{};
        const float z0=.5f*(now[0][2]+feet_[0][2]),z1=.5f*(now[1][2]+feet_[1][2]);
        const float low=std::min(z0,z1);
        weight={std::exp(-(z0-low)/kSoft),std::exp(-(z1-low)/kSoft)};
        const float sum=weight[0]+weight[1];
        for(int f=0;f<2;++f) {
            weight[f]/=sum;
            for(int a=0;a<2;++a) measured[f][a]=-scale*(now[f][a]-feet_[f][a])/dt;
        }
        feet_=now;
        const Vec3 specific=turn_by(attitude,sensors.accelerometer); // horizontal part: gravity drops out
        Vec2 pos{},vel{};
        for(int a=0;a<2;++a) {
            const float acc=specific[a]-bias_[a];
            pos[a]=position_[a]+velocity_[a]*dt+.5f*acc*dt*dt;
            vel[a]=velocity_[a]+acc*dt;
        }
        const Mat3 f{1,dt,-.5f*dt*dt, 0,1,-dt, 0,0,1};
        const Mat3 ft{f[0],f[3],f[6],f[1],f[4],f[7],f[2],f[5],f[8]};
        Mat3 p=mul(mul(f,p_),ft);
        p[4]+=(kSigA*dt)*(kSigA*dt); p[8]+=kSigB*kSigB*dt;
        float information=0; Vec2 mean{};
        for(int foot=0;foot<2;++foot) {
            float r=kSigV*kSigV/std::max(weight[foot],1e-6f);
            const float e=std::hypot(measured[foot][0]-vel[0],measured[foot][1]-vel[1]);
            const float gate=std::max(e/(kGate*std::sqrt(p[4]+r)),1.f);
            r*=gate*gate;
            information+=1/r;
            for(int a=0;a<2;++a) mean[a]+=measured[foot][a]/r;
        }
        const float r=1/information;
        const Vec3 gain{p[1]/(p[4]+r),p[4]/(p[4]+r),p[7]/(p[4]+r)};
        for(int a=0;a<2;++a) {
            const float innovation=mean[a]*r-vel[a];
            position_[a]=pos[a]+gain[0]*innovation;
            velocity_[a]=vel[a]+gain[1]*innovation;
            bias_[a]+=gain[2]*innovation;
        }
        Mat3 next{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) next[3*i+j]=p[3*i+j]-gain[i]*p[3+j];
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) p_[3*i+j]=.5f*(next[3*i+j]+next[3*j+i]);
        for(float v:position_) if(!std::isfinite(v)) throw std::runtime_error("Non-finite position estimate");
    }
    const Vec2& position() const { return position_; }
    const Vec2& velocity() const { return velocity_; }
    // Both foot sites from the leg encoders, in the start-heading, gravity-aligned frame.
    static std::array<Vec3,2> feet(const Sensors& sensors,const Mat3& attitude) {
        std::array<Vec3,2> out{};
        for(int side=0;side<2;++side) {
            Vec3 p{}; Mat3 r{1,0,0,0,1,0,0,0,1};
            for(const auto& link:kLegs[side].links) {
                const float angle=sensors.position[kMotorIds[link.joint]];
                if(!std::isfinite(angle)) throw std::runtime_error("Non-finite leg joint position");
                const auto offset=turn_by(r,link.position);
                for(int i=0;i<3;++i) p[i]+=offset[i];
                r=mul(r,link.rotation);
                const auto turn=about_axis(link.axis,angle);
                const auto turned=turn_by(turn,link.anchor);
                const auto shift=turn_by(r,Vec3{link.anchor[0]-turned[0],link.anchor[1]-turned[1],link.anchor[2]-turned[2]});
                for(int i=0;i<3;++i) p[i]+=shift[i];
                r=mul(r,turn);
            }
            const auto site=turn_by(r,kLegs[side].site);
            out[side]=turn_by(attitude,Vec3{p[0]+site[0],p[1]+site[1],p[2]+site[2]});
        }
        return out;
    }
private:
    Vec2 position_{},velocity_{},bias_{};
    Mat3 p_{};
    std::array<Vec3,2> feet_{};
};

// ------------------------------------------------------------------ the controller
struct Step {
    Observation observation;
    Action action,target;
    int clock;
    bool ended;        // the plan has played to its last frame
    bool done;         // ... and the robot has stood settled for one second since (sensors only)
    Vec2 position;     // the estimate, in the walk's frame: x along the route from the start (m)
    float heading;     // rad, from the heading at the start
};

class Controller {
public:
    Controller(const std::string& directory,int policy):
        actor_(directory+"/"+std::string(kNames.at(policy))+"/policy.onnx",kActorHashes.at(policy)),
        library_(Library::shared(directory)) {}
    // Begin a walk of the planned distance nearest to `metres`, from where the robot stands
    // and the way it faces at the next step().  The plan's first frame is held for hold_s
    // (while an entry blend settles).  previous: the correction the joints carry now, when
    // the robot comes straight from another walk.  Returns the distance that will be walked.
    float start(float metres,float hold_s=0,const Action* previous=nullptr) {
        if(!(hold_s>=0) || hold_s>10) throw std::runtime_error("Invalid walk hold time");
        plan_=&library_->plan(library_->nearest(metres));
        frames_=library_->frames(*plan_);
        clock_=0; holds_=int(std::lround(hold_s/kStepDt)); first_=true; stood_=0; done_=false;
        if(previous) previous_=*previous; else previous_.fill(0);
        return Library::metres(*plan_);
    }
    bool started() const { return plan_!=nullptr; }
    float metres() const { return plan_?Library::metres(*plan_):0.f; }
    float seconds() const { return plan_?(plan_->length-1)*kStepDt:0.f; }
    const Action& previous() const { return previous_; }
    // Deployment only (training has none): each step's change of each raw action is held to `limit` (raw action
    // units per 20 ms step, per joint).  A policy that leaves its training distribution, in a fall for example,
    // cannot command a larger change than its healthy self did.  The limited action is the one applied, and the
    // one the actor reads as its previous action.
    void rate_limit(const Action& limit) {
        for(float v:limit) if(!std::isfinite(v) || v<=0) throw std::runtime_error("Invalid action rate limit");
        rate_=limit; limited_=true;
    }
    void no_rate_limit() { limited_=false; }
    bool rate_limited() const { return limited_; }
    // The estimate measured from the walk's start: metres along the route, metres to its left.
    Vec2 walked() const {
        if(!plan_) return {0,0};
        return {estimator_.position()[0]-frames_[0].x,estimator_.position()[1]-frames_[0].y};
    }
    const Library& library() const { return *library_; }
    Estimator& estimator() { return estimator_; }
    Step step(const Sensors& sensors) {
        if(!plan_) throw std::runtime_error("No walk distance selected");
        const int last=int(plan_->length)-1;
        const auto q=normalized(sensors.quaternion);
        Observation obs{};
        std::copy(sensors.gyro.begin(),sensors.gyro.end(),obs.begin());
        obs[3]=2*(q[0]*q[2]-q[1]*q[3]); obs[4]=-2*(q[2]*q[3]+q[0]*q[1]); obs[5]=-1+2*(q[1]*q[1]+q[2]*q[2]);
        for(size_t i=0;i<23;++i) {
            obs[6+i]=sensors.position[kMotorIds[i]]-kDefault[i];
            obs[29+i]=sensors.velocity[kMotorIds[i]];
            obs[52+i]=previous_[i];
        }
        if(first_) {
            // The walk's frame: the origin under the pelvis now, x the way the robot faces now.
            heading_origin_=yaw(q);
            estimator_.reset(sensors,mul(about_z(-heading_origin_),rotation(q)),{frames_[0].x,frames_[0].y});
        } else {
            if(holds_>0) --holds_; else clock_=std::min(clock_+1,last);
            estimator_.update(sensors,mul(about_z(-heading_origin_),rotation(q)));
            stood_=(clock_==last && settled(obs))?stood_+kStepDt:0.f;
            if(stood_>=1.f) done_=true; // latched, as the box controllers do
        }
        SensorFrame frame{};
        std::copy(obs.begin(),obs.begin()+75,frame.begin());
        if(first_) history_.fill(frame); // as the training env: the first reading fills the history
        else { for(int h=0;h+1<kHistory;++h) history_[h]=history_[h+1]; history_[kHistory-1]=frame; }
        first_=false;
        const auto& now=frames_[clock_]; const auto& next=frames_[std::min(clock_+1,last)]; const auto& goal=frames_[last];
        const auto& position=estimator_.position();
        const float heading=wrap(yaw(q)-heading_origin_);
        for(size_t i=0;i<23;++i) obs[75+i]=next.qcmd[i]-kDefault[i];
        obs[98]=now.sin_phase; obs[99]=now.cos_phase; obs[100]=now.walking;
        obs[101]=std::clamp(goal.x-now.x,0.f,1.f);
        obs[102]=clock_==last?1.f:0.f;
        // Along and across the route (not the robot's heading frame): the plan from the
        // estimate, the goal from the estimate (a far goal looks like 1 m), the heading.
        obs[103]=std::clamp(now.x-position[0],-.5f,.5f); obs[104]=std::clamp(now.y-position[1],-.5f,.5f);
        obs[105]=std::clamp(goal.x-position[0],-.5f,1.f); obs[106]=std::clamp(goal.y-position[1],-.5f,.5f);
        obs[107]=std::sin(heading); obs[108]=std::cos(heading);
        ActorInput input{};
        size_t at=0;
        for(const auto& term:kSensorTerms) for(int h=0;h<kHistory;++h) for(int i=0;i<term[1];++i) input[at++]=history_[h][term[0]+i];
        std::copy(obs.begin()+75,obs.end(),input.begin()+at);
        auto action=actor_.infer(input);
        if(limited_) for(size_t i=0;i<23;++i) {   // an action inside the limit is left bit for bit as the actor gave it
            const float change=action[i]-previous_[i];
            if(change>rate_[i]) action[i]=previous_[i]+rate_[i];
            else if(change<-rate_[i]) action[i]=previous_[i]-rate_[i];
        }
        Action target{};
        for(size_t i=0;i<23;++i) {
            target[i]=next.qcmd[i]+kScale[i]*action[i];
            if(!std::isfinite(target[i])) throw std::runtime_error("Non-finite walk target");
        }
        previous_=action; // not clipped, as in training; rate-limited if a limit is set
        return {obs,action,target,clock_,clock_==last,done_,position,heading};
    }
private:
    // Standing still: the pelvis barely turning and within 15 degrees of upright, no joint
    // moving.  Unlike the box stages there is no pose test: these policies stand at the goal
    // in their own posture, up to 0.5 rad from the plan's last frame in single joints.
    static bool settled(const Observation& obs) {
        if(std::sqrt(obs[0]*obs[0]+obs[1]*obs[1]+obs[2]*obs[2])>=.25f ||
           obs[5]>=-std::cos(15.f*3.141592653589793f/180)) return false;
        for(size_t i=0;i<23;++i) if(std::abs(obs[29+i])>=.5f) return false;
        return true;
    }
    box_policy::ModelActor<kActorInputs> actor_;
    std::array<SensorFrame,kHistory> history_{};
    std::shared_ptr<const Library> library_;
    const Plan* plan_=nullptr;
    const Frame* frames_=nullptr;
    Estimator estimator_;
    Action previous_{},rate_{};
    bool limited_=false;
    int clock_=0,holds_=0;
    float heading_origin_=0,stood_=0;
    bool first_=true,done_=false;
};

// ------------------------------------------------------------------ the operator's distance
// The distance the next walk will use.  It is set at launch (--distance), typed into the
// controller's terminal, or stepped with the gamepad, and read when a walk starts.
class Distance {
public:
    static Distance& instance() { static Distance d; return d; }
    void configure(std::shared_ptr<const Library> library,float metres) {
        std::lock_guard<std::mutex> lock(mutex_);
        library_=std::move(library); index_=checked(metres);
    }
    bool configured() const { std::lock_guard<std::mutex> lock(mutex_); return bool(library_); }
    float get() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!library_) throw std::runtime_error("Walk distance is not configured");
        return Library::metres(library_->plan(index_));
    }
    // Returns the planned distance now selected; throws std::out_of_range outside the library.
    float set(float metres) {
        std::lock_guard<std::mutex> lock(mutex_);
        index_=checked(metres);
        return Library::metres(library_->plan(index_));
    }
    // Step by delta metres, stopping at the library's ends.
    float step(float delta) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!library_) throw std::runtime_error("Walk distance is not configured");
        const float want=Library::metres(library_->plan(index_))+delta;
        index_=library_->nearest(std::clamp(want,library_->shortest(),library_->longest()));
        return Library::metres(library_->plan(index_));
    }
    float shortest() const { std::lock_guard<std::mutex> lock(mutex_); return library_->shortest(); }
    float longest() const { std::lock_guard<std::mutex> lock(mutex_); return library_->longest(); }
private:
    size_t checked(float metres) const {
        if(!library_) throw std::runtime_error("Walk distance is not configured");
        // half a library spacing of slack at the ends; anything further is refused, not clamped
        if(!std::isfinite(metres) || metres<library_->shortest()-.05f || metres>library_->longest()+.05f)
            throw std::out_of_range("Walk distance outside the planned range");
        return library_->nearest(metres);
    }
    mutable std::mutex mutex_;
    std::shared_ptr<const Library> library_;
    size_t index_=0;
};
} // namespace parametrized_walk
