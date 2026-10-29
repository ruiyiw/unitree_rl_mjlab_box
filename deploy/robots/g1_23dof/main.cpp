#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_HumanPushRecovery.h"
#include "State_BoxPolicy.h"
#include "BoxPreflight.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
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
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Offline deployment check: return before constructing any DDS objects.
    if (argc==3 && std::string(argv[1])=="--check-box-config") {
        try {
            param::bin_path=param::get_bin_path();
            param::load_config_file();
            box_policy::preflight(YAML::LoadFile(argv[2]),param::proj_dir);
            return 0;
        } catch (const std::exception& e) {
            std::cerr<<e.what()<<"\n";
            return 1;
        }
    }
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Fail a malformed box profile before waiting for or commanding a robot.
    bool box_profile=false;
    for (const auto& state:param::config["FSM"]["_"])
        if (state.second["type"] && state.second["type"].as<std::string>()=="BoxPolicy") box_profile=true;
    if (box_profile) {
        try { box_policy::preflight(param::config,param::proj_dir); }
        catch (const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
    }

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // The profile's buttons (LT, RT, LB, RB are the remote's L2, R2, L1, R1).
    for (const auto& state:param::config["FSM"]["_"]) {
        const auto name=state.first.as<std::string>();
        for (const auto& t:param::config["FSM"][name]["transitions"])
            std::cout << "  " << name << " -> " << t.first.as<std::string>() << ": [" << t.second.as<std::string>() << "]\n";
    }

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

