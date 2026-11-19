#include "ParametrizedWalkPolicy.h"
// C interface to the walk controller g1_ctrl runs, for the offline tests (no DDS).
// telemetry: IMU quaternion 4, gyro 3, accelerometer 3, 29 motor positions, 29 motor speeds.
static thread_local std::string error;
using parametrized_walk::Controller;
extern "C" {
const char* walk_error() {return error.c_str();}
void* walk_create(const char* directory,int policy) {
    try {return new Controller(directory,policy);}
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void walk_destroy(void* p) {delete static_cast<Controller*>(p);}
int walk_plans(void* p) {return int(static_cast<Controller*>(p)->library().size());}
float walk_plan_metres(void* p,int i) {
    const auto& lib=static_cast<Controller*>(p)->library();
    return parametrized_walk::Library::metres(lib.plan(i));
}
// Returns the planned distance, or a negative number on error.
float walk_start(void* p,float metres,float hold_s,const float* previous,float scale) {
    try {
        auto* c=static_cast<Controller*>(p);
        parametrized_walk::Action action{};
        if(previous) std::copy(previous,previous+23,action.begin());
        c->estimator().scale=scale;
        return c->start(metres,hold_s,previous?&action:nullptr);
    } catch(const std::exception& e) {error=e.what();return -1;}
}
// The action-rate limit (23 values, raw action units per step); a null pointer removes it.
int walk_rate_limit(void* p,const float* limit) {
    try {
        auto* c=static_cast<Controller*>(p);
        if(!limit) {c->no_rate_limit();return 0;}
        parametrized_walk::Action action{};
        std::copy(limit,limit+23,action.begin());
        c->rate_limit(action);
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
int walk_step(void* p,const float* telemetry,float* obs,float* action,float* target,int* state,float* estimate) {
    try {
        parametrized_walk::Sensors sensors{};
        std::copy(telemetry,telemetry+4,sensors.quaternion.begin());
        std::copy(telemetry+4,telemetry+7,sensors.gyro.begin());
        std::copy(telemetry+7,telemetry+10,sensors.accelerometer.begin());
        std::copy(telemetry+10,telemetry+39,sensors.position.begin());
        std::copy(telemetry+39,telemetry+68,sensors.velocity.begin());
        const auto result=static_cast<Controller*>(p)->step(sensors);
        std::copy(result.observation.begin(),result.observation.end(),obs);
        std::copy(result.action.begin(),result.action.end(),action);
        std::copy(result.target.begin(),result.target.end(),target);
        state[0]=result.clock; state[1]=result.ended; state[2]=result.done;
        estimate[0]=result.position[0]; estimate[1]=result.position[1]; estimate[2]=result.heading;
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
// The distance the operator's selector would use for a request (negative: refused).
float walk_select(const char* directory,float metres) {
    try {
        auto& d=parametrized_walk::Distance::instance();
        if(!d.configured()) d.configure(parametrized_walk::Library::shared(directory),2.f);
        return d.set(metres);
    } catch(const std::exception& e) {error=e.what();return -1;}
}
float walk_select_step(float delta) {
    try {return parametrized_walk::Distance::instance().step(delta);}
    catch(const std::exception& e) {error=e.what();return -1;}
}
}

// Any single-input, single-output float actor (the simulation runs the Velocity policy with it).
struct Onnx {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"walk_test"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string in,out;
};
extern "C" {
void* onnx_open(const char* path) {
    try {
        auto* m=new Onnx;
        m->options.SetIntraOpNumThreads(1); m->options.SetInterOpNumThreads(1);
        m->session=std::make_unique<Ort::Session>(m->env,path,m->options);
        Ort::AllocatorWithDefaultOptions alloc;
        m->in=m->session->GetInputNameAllocated(0,alloc).get();
        m->out=m->session->GetOutputNameAllocated(0,alloc).get();
        return m;
    } catch(const std::exception& e) {error=e.what();return nullptr;}
}
void onnx_close(void* p) {delete static_cast<Onnx*>(p);}
int onnx_run(void* p,const float* input,int n,float* output,int m) {
    try {
        auto* model=static_cast<Onnx*>(p);
        auto info=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        const std::array<int64_t,2> shape{1,n};
        auto value=Ort::Value::CreateTensor<float>(info,const_cast<float*>(input),n,shape.data(),2);
        const char* ins[]{model->in.c_str()}; const char* outs[]{model->out.c_str()};
        auto result=model->session->Run(Ort::RunOptions{nullptr},ins,&value,1,outs,1);
        if(result[0].GetTensorTypeAndShapeInfo().GetElementCount()!=size_t(m)) throw std::runtime_error("Unexpected actor output size");
        std::copy(result[0].GetTensorData<float>(),result[0].GetTensorData<float>()+m,output);
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
}
