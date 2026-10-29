// Offline bridge to the same C++ actor/controller used by the DDS FSM.
#include "BoxPolicy.h"
static thread_local std::string error;
extern "C" {
const char* box_error() {return error.c_str();}
void* box_create(const char* directory,int stage) {
    try {return new box_policy::Controller(directory,stage);}
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void box_destroy(void* p) {delete static_cast<box_policy::Controller*>(p);}
void box_reset(void* p) {static_cast<box_policy::Controller*>(p)->reset();}
int box_step(void* p,const float* telemetry,float* obs,float* action,float* target,int* state) {
    try {
        box_policy::Sensors sensors{};
        std::copy(telemetry,telemetry+4,sensors.quaternion.begin());
        std::copy(telemetry+4,telemetry+7,sensors.gyro.begin());
        std::copy(telemetry+7,telemetry+36,sensors.position.begin());
        std::copy(telemetry+36,telemetry+65,sensors.velocity.begin());
        const auto result=static_cast<box_policy::Controller*>(p)->step(sensors);
        std::copy(result.observation.begin(),result.observation.end(),obs);
        std::copy(result.action.begin(),result.action.end(),action);
        std::copy(result.target.begin(),result.target.end(),target);
        state[0]=result.stage;state[1]=result.clock;state[2]=result.done;
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
}
