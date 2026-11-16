// A stand-in for the robot's DDS side, for the loopback test (unitree_parametrized_skills/loopback.py).
// stdin : one robot state per frame: float32[68] (IMU quaternion 4, gyro 3, accelerometer 3,
//         29 motor positions, 29 motor speeds) then 40 bytes of wireless remote.
// stdout: for each frame, the latest rt/lowcmd: uint32 commands received so far,
//         float32[29*5] (q, dq, kp, kd, tau per motor), uint8[29] motor modes.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>

int main(int argc,char** argv) {
    using State=unitree_hg::msg::dds_::LowState_;
    using Command=unitree_hg::msg::dds_::LowCmd_;
    unitree::robot::ChannelFactory::Instance()->Init(0,argc>1?argv[1]:"lo");
    unitree::robot::ChannelPublisher<State> publisher("rt/lowstate");
    publisher.InitChannel();
    std::mutex mutex; uint32_t count=0;
    std::array<float,29*5> latest{}; std::array<uint8_t,29> modes{};
    unitree::robot::ChannelSubscriber<Command> subscriber("rt/lowcmd");
    subscriber.InitChannel([&](const void* message) {
        const auto& cmd=*static_cast<const Command*>(message);
        std::lock_guard<std::mutex> lock(mutex);
        for(int i=0;i<29;++i) {
            const auto& m=cmd.motor_cmd()[i];
            latest[5*i]=m.q(); latest[5*i+1]=m.dq(); latest[5*i+2]=m.kp(); latest[5*i+3]=m.kd(); latest[5*i+4]=m.tau();
            modes[i]=m.mode();
        }
        ++count;
    });
    std::array<float,68> in{}; std::array<uint8_t,40> remote{};
    while(std::fread(in.data(),sizeof(float),in.size(),stdin)==in.size() && std::fread(remote.data(),1,40,stdin)==40) {
        State state{};
        state.mode_machine()=4;
        for(int i=0;i<4;++i) state.imu_state().quaternion()[i]=in[i];
        for(int i=0;i<3;++i) { state.imu_state().gyroscope()[i]=in[4+i]; state.imu_state().accelerometer()[i]=in[7+i]; }
        for(int i=0;i<29;++i) { state.motor_state()[i].q()=in[10+i]; state.motor_state()[i].dq()=in[39+i]; }
        state.wireless_remote()=remote;
        publisher.Write(state);
        std::lock_guard<std::mutex> lock(mutex);
        std::fwrite(&count,4,1,stdout); std::fwrite(latest.data(),sizeof(float),latest.size(),stdout);
        std::fwrite(modes.data(),1,modes.size(),stdout); std::fflush(stdout);
    }
    return 0;
}
