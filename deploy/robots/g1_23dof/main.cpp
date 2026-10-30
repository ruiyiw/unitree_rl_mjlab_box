#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_HumanPushRecovery.h"
#include "State_BoxPolicy.h"
#include "BoxPreflight.h"
#include "State_TableBoxPolicy.h"
#include "TableBoxPreflight.h"
#include "DeploymentRecordMessages.h"
#include <csignal>
#include <set>
#include <sstream>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace {
volatile std::sig_atomic_t stop_requested=0;
void request_stop(int) { stop_requested=1; }

void save_recording_metadata(const std::filesystem::path& dir,
                             const deployment_recording::Recorder& recorder,
                             int argc,char** argv) {
    YAML::Node metadata;
    metadata["format"]="g1-deployment-recording-v1";
    metadata["monotonic_start_ns"]=recorder.start_ns();
    metadata["unix_time_ns"]=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    metadata["unix_time_monotonic_ns"]=deployment_recording::monotonic_ns();
    metadata["project_dir"]=param::proj_dir.string();
    for (int i=0;i<argc;++i) metadata["argv"].push_back(argv[i]);
    metadata["config"]=YAML::Clone(param::config);
    for (int id : {0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26})
        metadata["motor_ids_23dof"].push_back(id);
    // Fingerprint deployed assets before DDS startup, without duplicating weights
    // in each session. Preserve the matching repository for policy re-execution.
    std::set<std::filesystem::path> files{param::bin_path};
    for (const auto& state:param::config["FSM"]["_"]) {
        const auto name=state.first.as<std::string>();
        const auto node=param::config["FSM"][name]["policy_dir"];
        if (!node) continue;
        const auto path=param::parser_policy_dir(node.as<std::string>());
        metadata["resolved_policy_dirs"][name]=path.string();
        for (const auto& entry:std::filesystem::recursive_directory_iterator(path)) {
            if (!entry.is_regular_file()) continue;
            const auto ext=entry.path().extension().string();
            if (ext==".onnx" || ext==".bin" || ext==".yaml" || ext==".json" || ext==".npz") files.insert(entry.path());
        }
    }
    for (const auto& file:files) {
        std::ifstream input(file,std::ios::binary);
        if (!input) throw std::runtime_error("Cannot fingerprint "+file.string());
        uint64_t hash=14695981039346656037ULL; std::array<char,65536> buffer;
        while (input.read(buffer.data(),buffer.size()) || input.gcount())
            hash=deployment_recording::hash_bytes(buffer.data(),input.gcount(),hash);
        if (!input.eof()) throw std::runtime_error("Cannot read "+file.string());
        std::ostringstream hex; hex << std::hex << std::setw(16) << std::setfill('0') << hash;
        metadata["files"][file.string()]["fnv1a64"]=hex.str();
        metadata["files"][file.string()]["bytes"]=std::filesystem::file_size(file);
    }
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(dir/"session.yaml"); out << metadata << '\n'; out.close();
}
}

bool init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    while (FSMState::lowstate->isTimeout()) {
        if (stop_requested) return false;
        usleep(10000);
    }
    spdlog::info("Connected to robot.");
    return !stop_requested;
}

int main(int argc, char** argv)
{
    // Offline deployment check: return before constructing any DDS objects.
    if (argc==3 && std::string(argv[1])=="--check-box-config") {
        try {
            param::bin_path=param::get_bin_path();
            param::load_config_file();
            const auto config=YAML::LoadFile(argv[2]);
            if (table_box::is_profile(config)) table_box::preflight(config,param::proj_dir);
            else box_policy::preflight(config,param::proj_dir);
            return 0;
        } catch (const std::exception& e) {
            std::cerr<<e.what()<<"\n";
            return 1;
        }
    }
    // Load parameters
    param::po::options_description recording_options("Deployment recording");
    recording_options.add_options()
        ("record",param::po::value<std::string>(),"record measured states, commands and FSM transitions to a NEW directory");
    auto vm = param::helper(argc, argv,&recording_options);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Fail a malformed box profile before waiting for or commanding a robot.
    bool box_profile=false;
    for (const auto& state:param::config["FSM"]["_"])
        if (state.second["type"] && state.second["type"].as<std::string>()=="BoxPolicy") box_profile=true;
    if (table_box::is_profile(param::config)) {
        try { table_box::preflight(param::config,param::proj_dir); }
        catch (const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
    } else if (box_profile) {
        try { box_policy::preflight(param::config,param::proj_dir); }
        catch (const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
    }

    std::shared_ptr<deployment_recording::Recorder> recorder;
    if (vm.count("record")) {
        try {
            const auto directory=std::filesystem::absolute(vm["record"].as<std::string>());
            recorder=std::make_shared<deployment_recording::Recorder>(directory);
            save_recording_metadata(directory,*recorder,argc,argv);
            recorder->start();
            spdlog::info("Recording deployment to {}",directory.string());
        } catch (const std::exception& e) {
            std::cerr << "Cannot start recording: " << e.what() << '\n'; return 1;
        }
    }

    std::signal(SIGINT,request_stop);
    std::signal(SIGTERM,request_stop);
    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    // Independent observers copy the delivered DDS samples, never the policy's
    // mutable command buffers. They do not publish anything.
    using StateSubscriber=unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>;
    using CommandSubscriber=unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>;
    std::unique_ptr<StateSubscriber> state_recording;
    std::unique_ptr<CommandSubscriber> command_recording;
    if (recorder) {
        state_recording=std::make_unique<StateSubscriber>("rt/lowstate");
        command_recording=std::make_unique<CommandSubscriber>("rt/lowcmd");
        state_recording->InitChannel([recorder](const void* msg) {
            recorder->submit(deployment_recording::pack(*static_cast<const unitree_hg::msg::dds_::LowState_*>(msg)));
        });
        command_recording->InitChannel([recorder](const void* msg) {
            recorder->submit(deployment_recording::pack(*static_cast<const unitree_hg::msg::dds_::LowCmd_*>(msg)));
        });
    }

    if (!init_fsm_state()) {
        FSMState::lowcmd.reset();
        return 0;
    }

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        FSMState::lowcmd.reset();
        return 1;
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    if (recorder) fsm->on_transition=[recorder](int from,int to) { recorder->transition(from,to); };
    fsm->start();

    // The profile's buttons (LT, RT, LB, RB are the remote's L2, R2, L1, R1).
    for (const auto& state:param::config["FSM"]["_"]) {
        const auto name=state.first.as<std::string>();
        for (const auto& t:param::config["FSM"][name]["transitions"])
            std::cout << "  " << name << " -> " << t.first.as<std::string>() << ": [" << t.second.as<std::string>() << "]\n";
    }

    uint64_t reported_drops=0;
    bool reported_failure=false;
    while (!stop_requested)
    {
        sleep(1);
        if (recorder && recorder->dropped()!=reported_drops) {
            reported_drops=recorder->dropped();
            spdlog::error("Recording incomplete: {} samples dropped",reported_drops);
        }
        if (recorder && recorder->failed() && !reported_failure) {
            reported_failure=true;
            spdlog::error("Recording write failed; controller remains active. Recording is incomplete.");
        }
    }
    // Stop control workers before flushing the recording. Ctrl+C does not make
    // an unsupported robot safe: use the normal Passive/suspended stop procedure.
    fsm->stop();
    FSMState::lowcmd.reset();
    if (state_recording) state_recording->CloseChannel();
    if (command_recording) command_recording->CloseChannel();
    if (recorder) {
        if (!recorder->finish() || recorder->dropped()) {
            spdlog::error("Recording incomplete: {} dropped samples, write failure={}",recorder->dropped(),recorder->failed());
            return 1;
        }
        spdlog::info("Recording saved.");
    }
    return 0;
}

