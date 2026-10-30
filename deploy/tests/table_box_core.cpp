#include "TableBoxPolicy.h"
static thread_local std::string error;
extern "C" {
const char* table_error() {return error.c_str();}
void* table_create(const char* directory,int stage) {
    try {return new table_box::Controller(directory,stage);}
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void table_destroy(void* p) {delete static_cast<table_box::Controller*>(p);}
void table_reset(void* p,const float* previous) {
    box_policy::Action action{}; std::copy(previous,previous+23,action.begin());
    static_cast<table_box::Controller*>(p)->reset(action);
}
int table_step(void* p,const float* telemetry,const float* command,float* obs,float* action,float* target,float* torque,int* state) {
    try {
        box_policy::Sensors sensors{};
        std::copy(telemetry,telemetry+4,sensors.quaternion.begin());
        std::copy(telemetry+4,telemetry+7,sensors.gyro.begin());
        std::copy(telemetry+7,telemetry+36,sensors.position.begin());
        std::copy(telemetry+36,telemetry+65,sensors.velocity.begin());
        box_policy::Action cmd{}; std::copy(command,command+23,cmd.begin());
        const auto result=static_cast<table_box::Controller*>(p)->step(sensors,cmd);
        std::copy(result.observation.begin(),result.observation.end(),obs);
        std::copy(result.action.begin(),result.action.end(),action);
        std::copy(result.target.begin(),result.target.end(),target);
        std::copy(result.torque.begin(),result.torque.end(),torque);
        state[0]=result.clock; state[1]=result.done;
        return 0;
    } catch(const std::exception& e) {error=e.what();return 1;}
}
}
