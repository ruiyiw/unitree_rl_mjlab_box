#include "HumanWalkPolicy.h"
// C interface to the human-walk controller g1_ctrl runs, for the offline tests (no DDS).
// telemetry: IMU quaternion 4, gyro 3, accelerometer 3, 29 motor positions, 29 motor speeds.
static thread_local std::string error;
using human_walk::Controller;
extern "C" {
const char* hw_error() {return error.c_str();}
void* hw_create(const char* directory) {
    try {return new Controller(directory);}
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void hw_destroy(void* p) {delete static_cast<Controller*>(p);}
int hw_start(void* p,float hold_s,float scale) {
    try {
        auto* c=static_cast<Controller*>(p);
        c->scale=scale; c->start(hold_s);
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
void hw_previous(void* p,const float* action) {
    human_walk::Action a{}; std::copy(action,action+23,a.begin());
    static_cast<Controller*>(p)->previous(a);
}
// state: clock, ended, done.  estimate: pelvis x y z, its rates, heading.
int hw_step(void* p,const float* telemetry,float* obs,float* action,float* target,float* torque,int* state,float* estimate) {
    try {
        human_walk::Sensors sensors{};
        std::copy(telemetry,telemetry+4,sensors.quaternion.begin());
        std::copy(telemetry+4,telemetry+7,sensors.gyro.begin());
        std::copy(telemetry+7,telemetry+10,sensors.accelerometer.begin());
        std::copy(telemetry+10,telemetry+39,sensors.position.begin());
        std::copy(telemetry+39,telemetry+68,sensors.velocity.begin());
        const auto result=static_cast<Controller*>(p)->step(sensors);
        std::copy(result.observation.begin(),result.observation.end(),obs);
        std::copy(result.action.begin(),result.action.end(),action);
        std::copy(result.target.begin(),result.target.end(),target);
        std::copy(result.torque.begin(),result.torque.end(),torque);
        state[0]=result.clock; state[1]=result.ended; state[2]=result.done;
        std::copy(result.position.begin(),result.position.end(),estimate);
        std::copy(result.velocity.begin(),result.velocity.end(),estimate+3);
        estimate[6]=result.heading;
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
}

// Any single-input, single-output float actor (the simulation runs the Velocity policy with it).
struct Onnx {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"human_walk_test"};
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
