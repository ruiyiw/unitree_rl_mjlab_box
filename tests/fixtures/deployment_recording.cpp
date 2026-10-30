// Offline fixture: constructs DDS value types, never a DDS participant/publisher.
#include "DeploymentRecordMessages.h"
#include "FSM/CtrlFSM.h"
#include <cassert>
#include <iostream>
#include <csignal>
#include <sys/resource.h>

using namespace deployment_recording;
struct CountingState : BaseState {
    CountingState(int id):BaseState(id,"Test"+std::to_string(id)) {}
    std::atomic<int> runs{0},exits{0};
    void run() override { ++runs; }
    void exit() override { ++exits; }
};

int main(int argc,char**argv) {
    try {
        std::string mode=argc>2?argv[2]:"normal";
        Recorder recorder(argv[1],mode=="overflow"?1:4096);
        if (mode=="overflow") {
            recorder.transition(-1,1); recorder.transition(1,2); recorder.transition(2,-1);
            recorder.start(); recorder.finish();
            assert(recorder.dropped()==2); return 0;
        }
        recorder.start();
        if (mode=="write_failure") {
            // Force ENOSPC-like stream failure without filling the host disk.
            std::signal(SIGXFSZ,SIG_IGN);
            const rlimit limit{16,16};
            assert(setrlimit(RLIMIT_FSIZE,&limit)==0);
            recorder.transition(-1,1);
            assert(!recorder.finish() && recorder.failed());
            return 0;
        }
        if (mode=="fsm") {
            auto first=std::make_shared<CountingState>(101);
            auto second=std::make_shared<CountingState>(102);
            first->registered_checks.emplace_back([first]{return first->runs.load()>=3;},102);
            CtrlFSM fsm(first); fsm.add(second);
            fsm.on_transition=[&](int from,int to){recorder.transition(from,to);};
            fsm.start();
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while (second->runs<3 && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            assert(second->runs>=3);
            fsm.stop();
            auto count=second->runs.load();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            assert(second->runs==count && first->exits==1 && second->exits==1);
            fsm.stop(); assert(second->exits==1);
            assert(recorder.finish()); return 0;
        }
        recorder.transition(-1,3);
        unitree_hg::msg::dds_::LowState_ state;
        state.version()={7,9}; state.mode_pr()=1; state.mode_machine()=4; state.tick()=123456;
        state.imu_state().quaternion()={.5f,-.5f,.5f,-.5f};
        state.imu_state().gyroscope()={1,2,3}; state.imu_state().accelerometer()={4,5,6};
        state.imu_state().rpy()={7,8,9}; state.imu_state().temperature()=-30;
        state.reserve()={10,11,12,13}; state.crc()=0xfeedabcd;
        for (size_t i=0;i<40;++i) state.wireless_remote()[i]=i;
        unitree_hg::msg::dds_::LowCmd_ cmd;
        cmd.mode_pr()=1; cmd.mode_machine()=4; cmd.reserve()={20,21,22,23}; cmd.crc()=0xabcdef12;
        for (int i=0;i<35;++i) {
            auto& m=state.motor_state()[i];
            m.mode()=i; m.q()=i+.25f; m.dq()=-i-.5f; m.ddq()=i+.75f; m.tau_est()=-i-.25f;
            m.temperature()={int16_t(i),int16_t(-i)}; m.vol()=48.f+i;
            m.sensor()={uint32_t(i),uint32_t(i+1)}; m.motorstate()=100+i;
            m.reserve()={uint32_t(i+2),uint32_t(i+3),uint32_t(i+4),uint32_t(i+5)};
            auto& c=cmd.motor_cmd()[i];
            c.mode()=i; c.q()=i+.125f; c.dq()=-i-.125f; c.tau()=i+.5f; c.kp()=40+i; c.kd()=3+i; c.reserve()=200+i;
        }
        // Two producers exercise concurrent sensor/command callbacks.
        std::thread a([&]{for(int i=0;i<200;++i) { state.tick()=123456+i; recorder.submit(pack(state)); }});
        std::thread b([&]{for(int i=0;i<200;++i) recorder.submit(pack(cmd)); });
        a.join(); b.join();
        recorder.transition(3,5); recorder.transition(5,-1);
        assert(recorder.finish() && recorder.dropped()==0);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
