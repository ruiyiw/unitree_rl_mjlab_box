"""Native MuJoCo smoke/playback using the real C++ deployment controller.

This CPU runner is a transport-free integration smoke test, not the archived
batched mjlab evaluation or its physical-success criteria.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
from . import MOTOR_IDS, ORDER, POLICY_DIR
from .runtime import Core, build_core, reference_path, verify


def play(stage,output,seconds=125.,viewer=False):
    import mujoco
    import torch
    reference_path();manifest=verify()
    from five_plans import configure_dimensions
    from geometry import geometry_cfg,initial_data
    from mjlab.scene import Scene
    from five_task import env_cfg
    configure_dimensions()
    scene=Scene(geometry_cfg('five-stage'),device='cpu')
    model=scene.compile()
    env_cfg('put-down',1,play=True).sim.mujoco.apply(model)
    data=initial_data(model)
    assert model.nu==23 and not model.body_gravcomp.any()
    bundle=torch.load(POLICY_DIR/'policy.pt',weights_only=False,map_location='cpu')
    initial=0 if stage=='sequence' else ORDER.index(stage)
    plan=bundle['references'][ORDER[initial]]['plan']
    names=manifest['interface']['joint_order'];ids=np.asarray(MOTOR_IDS)
    joint_ids=[model.joint('robot/'+n).id for n in names]
    qadr=model.jnt_qposadr[joint_ids];vadr=model.jnt_dofadr[joint_ids]
    actuator_joints=model.actuator_trnid[:,0].tolist()
    actuators=np.array([actuator_joints.index(j) for j in joint_ids])
    np.testing.assert_allclose(model.actuator_gainprm[actuators,0],manifest['interface']['kp'])
    np.testing.assert_allclose(-model.actuator_biasprm[actuators,2],manifest['interface']['kd'])
    root=model.joint('robot/floating_base_joint');qa=int(root.qposadr[0]);va=int(root.dofadr[0])
    data.qpos[qa:qa+7]=plan['qpos'][0,0,:7].numpy()
    data.qvel[va:va+6]=plan['qvel'][0,0,:6].numpy()
    data.qpos[qadr]=plan['qpos'][0,0,7:].numpy();data.qvel[vadr]=plan['qvel'][0,0,6:].numpy()
    box=model.joint('box/box_joint');bqa=int(box.qposadr[0]);bva=int(box.dofadr[0])
    xyz_yaw=plan['box'][0,0].numpy()
    data.qpos[bqa:bqa+7]=[*xyz_yaw[:3],math.cos(xyz_yaw[3]/2),0,0,math.sin(xyz_yaw[3]/2)]
    data.qvel[bva:bva+3]=plan['box_lin'][0,0].numpy();data.qvel[bva+5]=plan['box_wz'][0,0].item()
    mujoco.mj_forward(model,data)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=build_core(output)
    display=None
    if viewer:
        import mujoco.viewer
        display=mujoco.viewer.launch_passive(model,data)
    frames=[];transitions=[];last=-1;reason='duration reached';done=False
    try:
        with Core(library,-1 if stage=='sequence' else initial) as core:
            for tick in range(round(seconds/.02)):
                start=time.monotonic()
                telemetry=np.zeros(65,np.float32)
                telemetry[:4]=data.qpos[qa+3:qa+7]
                telemetry[4:7]=data.qvel[va+3:va+6] # MuJoCo free-joint angular velocity is local.
                telemetry[7+ids]=data.qpos[qadr];telemetry[36+ids]=data.qvel[vadr]
                try:obs,action,target,state=core.step(telemetry)
                except RuntimeError as error:reason=str(error);break
                if -obs[5]<math.cos(math.pi/4):reason='pelvis tilt exceeded 45 degrees';break
                if data.qpos[qa+2]<.45:reason='root height below .45 m';break
                if int(state[0])!=last:
                    last=int(state[0]);transitions.append({'time_s':tick*.02,'stage':ORDER[last]})
                done=bool(state[2])
                frames.append(data.qpos.copy());data.ctrl[actuators]=target
                for _ in range(4):mujoco.mj_step(model,data)
                if display:
                    if not display.is_running():reason='viewer closed';break
                    display.sync();time.sleep(max(0,.02-(time.monotonic()-start)))
    finally:
        if display:display.close()
    report={'stage':stage,'elapsed_s':float(data.time),'stop_reason':reason,
        'sensor_endpoint_done':done,'transitions':transitions,'plain_pd':True,
        'physics':'native CPU MuJoCo; not the batched GPU evaluation',
        'physical_task_success':'not scored','hardware_tested':False}
    np.savez_compressed(output/'trajectory.npz',qpos=np.asarray(frames),dt=.02)
    (output/'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report
