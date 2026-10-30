#pragma once

// Portable recording format. No DDS, policy or filesystem work is done by the
// controller callback beyond packing a fixed-size record and enqueueing it.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace deployment_recording {
inline uint64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline uint64_t hash_bytes(const char* data, size_t size, uint64_t hash=14695981039346656037ULL) {
    for (size_t i=0; i<size; ++i) { hash ^= static_cast<unsigned char>(data[i]); hash *= 1099511628211ULL; }
    return hash;
}

struct Packet {
    uint32_t type=0, size=0;
    uint64_t sequence=0, time_ns=0;
    std::array<char,4096> data{};
    template<class T> void put(T value) {
        static_assert(std::is_arithmetic_v<T> && sizeof(T)<=8);
        if (size+sizeof(T)>data.size()) throw std::length_error("Recording packet too large");
        // Preserve float bits; never serialize compiler padding or host byte order.
        using Bits = std::conditional_t<sizeof(T)==8,uint64_t,
                     std::conditional_t<sizeof(T)==4,uint32_t,
                     std::conditional_t<sizeof(T)==2,uint16_t,uint8_t>>>;
        Bits bits=0; std::memcpy(&bits,&value,sizeof(T));
        for (size_t i=0;i<sizeof(T);++i) data[size++]=static_cast<char>(bits>>(8*i));
    }
    template<class T,size_t N> void put(const std::array<T,N>& values) { for (T v:values) put(v); }
};

class Recorder {
public:
    explicit Recorder(const std::filesystem::path& directory, size_t capacity=4096)
        : queue_(capacity), start_ns_(monotonic_ns()) {
        if (!capacity) throw std::invalid_argument("Recording queue cannot be empty");
        // Atomic creation refuses existing sessions, including empty directories.
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Recording directory already exists: "+directory.string());
        output_.exceptions(std::ios::badbit | std::ios::failbit);
        output_.open(directory / "records.bin",std::ios::binary);
        output_.write("G1REC001",8);
        Packet header; header.put(uint32_t(1)); header.put(uint32_t(0));
        output_.write(header.data.data(),header.size);
        output_.flush();
        // Start explicitly after session metadata is saved and before DDS init.
    }
    ~Recorder() { finish(); }
    uint64_t start_ns() const { return start_ns_; }
    uint64_t dropped() const { return dropped_.load(); }
    bool failed() const { return failed_.load(); }
    void start() { worker_=std::thread([this]{write_loop();}); }

    void submit(Packet packet) {
        // This lock only protects bounded memory copies; the writer releases it
        // before any filesystem operation. Queue saturation never waits for disk.
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return;
        packet.sequence=attempted_++;
        if (count_==queue_.size() || failed_) { ++dropped_; return; }
        queue_[(head_+count_)%queue_.size()]=packet;
        ++count_;
    }
    void transition(int32_t from,int32_t to) {
        Packet p; p.type=3; p.time_ns=monotonic_ns(); p.put(from); p.put(to); submit(p);
    }
    bool finish() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
        if (worker_.joinable()) worker_.join();
        return !failed_;
    }
private:
    void write_packet(const Packet& p) {
        Packet header; header.put(p.size); header.put(p.type); header.put(p.sequence);
        header.put(p.time_ns-start_ns_);
        // Protect record header as well as payload.
        auto checksum=hash_bytes(header.data.data(),header.size);
        header.put(hash_bytes(p.data.data(),p.size,checksum));
        output_.write(header.data.data(),header.size);
        output_.write(p.data.data(),p.size);
    }
    void write_loop() noexcept {
        try {
            auto last_flush=monotonic_ns();
            for (;;) {
                Packet p; bool have=false,done=false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (count_) { p=queue_[head_]; head_=(head_+1)%queue_.size(); --count_; have=true; }
                    done=stopping_ && count_==0;
                }
                if (have) { write_packet(p); ++written_; }
                if (done) break;
                if (monotonic_ns()-last_flush>250000000) { output_.flush(); last_flush=monotonic_ns(); }
                if (!have) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            Packet end; end.type=4; end.time_ns=monotonic_ns(); end.sequence=attempted_;
            end.put(attempted_); end.put(written_); end.put(dropped_.load());
            write_packet(end); output_.flush(); output_.close();
        } catch (...) { failed_=true; }
    }
    std::vector<Packet> queue_;
    size_t head_=0,count_=0;
    uint64_t start_ns_,attempted_=0,written_=0;
    std::atomic<uint64_t> dropped_{0};
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
    bool stopping_=false;
    std::ofstream output_;
    std::thread worker_;
};
} // namespace deployment_recording
