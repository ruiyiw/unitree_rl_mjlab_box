"""Plain-PD default with explicit modes for reproducing historical research.

Body gravcomp generates generalized anti-gravity forces on the floating base as
well as arm joints. Only the scalar motor-joint terms can represent onboard
gravity-compensation torques. The actuator_only mode removes all passive
gravcomp while retaining the existing motor-joint compensation.
"""
import warp as wp
import mujoco_warp._src.passive as passive

_native_passive=passive.passive
_mode='none'


@wp.kernel
def _remove_passive_gravcomp(jnt_actgravcomp:wp.array(dtype=wp.int32),
                            dof_jntid:wp.array(dtype=wp.int32),
                            gravity:wp.array2d(dtype=wp.float32),
                            force:wp.array2d(dtype=wp.float32)):
    world,dof=wp.tid()
    if jnt_actgravcomp[dof_jntid[dof]]==0:
        force[world,dof]=force[world,dof]-gravity[world,dof]


def _motor_only_passive(model,data):
    _native_passive(model,data)
    wp.launch(_remove_passive_gravcomp,dim=(data.nworld,model.nv),
        inputs=[model.jnt_actgravcomp,model.dof_jntid,data.qfrc_gravcomp],
        outputs=[data.qfrc_passive])


def make_robot_spec(mode='none',base_spec_fn=None,gravity_scale=1.):
    if mode not in ('legacy','none','actuator_only'):raise ValueError(mode)
    if base_spec_fn is None:
        import scene
        base_spec_fn=scene.robot_spec
    spec=base_spec_fn()
    for body in spec.bodies:
        body.gravcomp=(gravity_scale if mode!='none' and
            any(k in body.name for k in ('shoulder','elbow','wrist')) else 0.)
    for joint in spec.joints:
        if any(k in joint.name for k in ('shoulder','elbow','wrist')):
            joint.actgravcomp=mode!='none'
    return spec


def configure_actuation(cfg,mode='none',gravity_scale=1.):
    """Call before constructing an environment and its captured CUDA graphs."""
    global _mode
    if mode not in ('legacy','none','actuator_only'):raise ValueError(mode)
    passive.passive=_motor_only_passive if mode=='actuator_only' else _native_passive
    _mode=mode
    native=cfg.scene.entities['robot'].spec_fn
    def robot_spec():return make_robot_spec(mode,native,gravity_scale)
    cfg.scene.entities['robot'].spec_fn=robot_spec
    return cfg


def audit_actuation(env,mode=None):
    """Record actual compiled-model and GPU forces after reset, not just flags."""
    mode=_mode if mode is None else mode
    model=env.sim.mj_model;data=env.sim.data
    joint=model.joint('robot/floating_base_joint');start=int(joint.dofadr[0])
    root_passive=data.qfrc_passive[0,start:start+6]
    root_actuator=data.qfrc_actuator[0,start:start+6]
    count=int((model.body_gravcomp!=0).sum())
    if mode=='none' and count:raise AssertionError('Plain-PD model still has body gravity compensation')
    if mode in ('none','actuator_only'):
        if root_passive.abs().max()>1e-5 or root_actuator.abs().max()>1e-5:
            raise AssertionError('Non-actuated floating-base assistance remains')
    return dict(mode=mode,compensated_body_count=count,
        floating_base_passive_force=root_passive.tolist(),floating_base_actuator_force=root_actuator.tolist(),
        no_floating_base_gravity_assistance=bool(root_passive.abs().max()<1e-5 and root_actuator.abs().max()<1e-5))
