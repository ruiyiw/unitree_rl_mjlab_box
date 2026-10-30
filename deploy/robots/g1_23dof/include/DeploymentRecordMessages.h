#pragma once
#include "DeploymentRecorder.h"
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>

namespace deployment_recording {
inline Packet pack(const unitree_hg::msg::dds_::LowState_& msg) {
    Packet p; p.type=1; p.time_ns=monotonic_ns();
    p.put(msg.version()); p.put(msg.mode_pr()); p.put(msg.mode_machine()); p.put(msg.tick());
    const auto& imu=msg.imu_state();
    p.put(imu.quaternion()); p.put(imu.gyroscope()); p.put(imu.accelerometer());
    p.put(imu.rpy()); p.put(imu.temperature());
    for (const auto& m:msg.motor_state()) {
        p.put(m.mode()); p.put(m.q()); p.put(m.dq()); p.put(m.ddq()); p.put(m.tau_est());
        p.put(m.temperature()); p.put(m.vol()); p.put(m.sensor()); p.put(m.motorstate()); p.put(m.reserve());
    }
    p.put(msg.wireless_remote()); p.put(msg.reserve()); p.put(msg.crc());
    return p;
}
inline Packet pack(const unitree_hg::msg::dds_::LowCmd_& msg) {
    Packet p; p.type=2; p.time_ns=monotonic_ns();
    p.put(msg.mode_pr()); p.put(msg.mode_machine());
    for (const auto& m:msg.motor_cmd()) {
        p.put(m.mode()); p.put(m.q()); p.put(m.dq()); p.put(m.tau()); p.put(m.kp()); p.put(m.kd()); p.put(m.reserve());
    }
    p.put(msg.reserve()); p.put(msg.crc());
    return p;
}
} // namespace deployment_recording
