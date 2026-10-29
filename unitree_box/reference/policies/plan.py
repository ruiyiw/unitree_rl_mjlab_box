#!/usr/bin/env python3
"""plan.py — whole-body reference motions for the three skills, from the MPC.

Runs in the `mjlab` env.  A plan is one skill's motion, 50 frames a second:
the robot's joint positions and base pose, the joint TARGETS (they differ from
the positions only while the hands squeeze the box), the tracked bodies'
poses and velocities, and the box's pose.  It is built in four layers:

  1. a path of pelvis poses and the footsteps along it          (gait.py)
  2. the centre of mass and the final footsteps: the walking MPC,
     a QP every 0.1 s, solved by Rekin                          (mpc.py)
  3. a schedule for the hands, the box and the pelvis height
     (reach, close, lift, carry, lower, set down, let go)
  4. whole-body inverse kinematics every frame: feet on their
     footsteps (and swing arcs), the CoM of robot + box on the
     MPC's, the palms on their targets, the torso upright

Every plan is in the frame of the box it handles, x pointing out of the
prop's front toward the robot, origin on the floor under the box's centre:

  fetch   from a start pose 0.5–1.0 m in front of the box: walk up, hands
          forward, reach around the box, close, lift, pull it to the carry
          pose (box frame on the cart)
  carry   from the fetch's end: step back from the cart, turn, walk to the
          stance in front of the shelf (the same frame)
  place   from the carry's end: lower the box onto the board, let go, arms
          back down (box frame on the shelf)

  python plan.py --bank            # banks/plans.pt (training) and banks/plans_eval.pt
  python plan.py --show carry      # media/plan_carry.gif: one plan, played kinematically
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import gait as gt  # noqa: E402
import mpc  # noqa: E402
import scene as sc  # noqa: E402
import g1_23dof as g1  # noqa: E402

FPS = 50
DT = 1.0 / FPS
SWING_H = 0.07                     # m, swing-foot clearance
FOOT_Z = 0.0015                    # m, foot site height standing
BOX_Z0 = sc.CART_TOP + sc.BOX_HALF[2]          # box centre on the cart deck
BOX_REST_Z = sc.SHELF_TOP + sc.BOX_HALF[2]     # box centre resting on the shelf
STANCE = (sc.HOLD_X, 0.0, math.pi)             # pelvis at the box, facing it (box frame)
CARRY_BOX = (0.27, 0.0, 0.10)      # box centre in the pelvis yaw frame while carried
READY = (0.22, sc.BOX_HALF[1] + sc.PALM_R + 0.07, 0.03)   # palms, pelvis frame, before reaching
OPEN = 0.04                        # m, palms outside the box's faces before closing
LIFT = 0.05                        # m
PLACE_DIP = 0.08                   # m, pelvis lowered while setting the box down
BACK_UP = 0.30                     # m, carry steps back from the cart before turning
BOX_MASS = sc.BOX_MASS

TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
           "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
           "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_rubber_hand")
HAND = {s: f"{s}_wrist_roll_rubber_hand" for s in ("left", "right")}     # the palm's body
ANCHOR = TRACKED.index("torso_link")

# layouts (box frame of the cart): the same ranges as the task's
START_DIST = (0.5, 1.0)            # m, fetch start in front of the stance
START_LAT, START_YAW = 0.3, 0.5    # per metre of start distance
SHELF_R = (2.0, 3.2)               # m, shelf place point from the fetch stance
SHELF_ANGLE = (60.0, 120.0)        # deg, either side of the stance heading
SHELF_YAW_NOISE = 0.25             # rad


def smooth(s):
    """Min-jerk 0 → 1."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def pivot(xy, yaw, pitch, edge):
    """The foot site of a sole turned by `pitch` (deg, toe down positive) about its
    edge (gait.HEEL_X or TOE_X along the foot, on the floor), where the flat sole's
    site would be at xy: (site xy, site z)."""
    th = math.radians(pitch)
    d = edge * (1.0 - math.cos(th))
    return np.asarray(xy) + d * np.array([math.cos(yaw), math.sin(yaw)]), FOOT_Z + edge * math.sin(th)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rz(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ry(th):
    """About y: positive turns +x toward −z (a sole's toe down)."""
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def quat_z(th):
    return np.array([math.cos(th / 2), 0, 0, math.sin(th / 2)])


# ------------------------------------------------------------------ the robot model

class Robot:
    """Unitree's 23-DoF G1 (the same kinematics mjlab simulates) and an IK solver."""

    def __init__(self):
        self.m = g1.get_spec().compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.joints = [m.joint(i).name for i in range(1, m.njnt)]
        self.site = {n: m.site(n).id for n in ("left_foot", "right_foot", "left_palm", "right_palm")}
        self.body = {n: m.body(n).id for n in TRACKED}
        self.mass = float(m.body_subtreemass[1])
        lim = m.jnt_range[1:]
        mid, half = lim.mean(1), (lim[:, 1] - lim[:, 0]) / 2
        self.lo, self.hi = mid - 0.95 * half, mid + 0.95 * half
        self.q_nom = np.zeros(m.nq)
        self.q_nom[2], self.q_nom[3] = sc.PELVIS_Z, 1.0
        for i, n in enumerate(self.joints):
            for pat, v in g1.KNEES_BENT_KEYFRAME.joint_pos.items():
                if re.fullmatch(pat, n):
                    self.q_nom[7 + i] = v
        self.arm = np.array([any(a in n for a in sc.ARM_JOINTS) for n in self.joints])
        self.leg = np.array([("hip" in n or "knee" in n or "ankle" in n) for n in self.joints])
        self.waist = np.array(["waist" in n for n in self.joints])
        self.arm_side = {s: np.array([n.startswith(s) and a for n, a in zip(self.joints, self.arm)])
                         for s in ("left", "right")}

    def fk(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d

    def rest_palms(self):
        """Palm positions in the pelvis frame, arms at rest (the keyframe)."""
        d = self.fk(self.q_nom)
        p = d.xpos[self.body["pelvis"]]
        return {s: d.site_xpos[self.site[f"{s}_palm"]] - p for s in ("left", "right")}

    def solve(self, q, tk, iters=8, arms_only=False):
        """Gauss–Newton with damping on the weighted task errors; tk is a dict of
        targets (see Planner.targets).  Returns the new q and the max errors."""
        m, d = self.m, self.d
        nv = m.nv
        q = q.copy()
        for _ in range(iters):
            self.fk(q)
            Js, es = [], []

            def add(J, e, w):
                Js.append(math.sqrt(w) * J)
                es.append(math.sqrt(w) * e)

            def site_rows(name, pos, R=None, wp=1.0, wr=1.0, body=None):
                jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
                sid = self.site[name]
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                add(jp, pos - d.site_xpos[sid], wp)
                if R is not None:
                    Rc = d.xmat[body].reshape(3, 3) if body is not None else d.site_xmat[sid].reshape(3, 3)
                    if body is not None:
                        mujoco.mj_jacBody(m, d, jp, jr, body)
                    er = 0.5 * sum(np.cross(Rc[:, k], R[:, k]) for k in range(3))
                    add(jr, er, wr)

            def body_rot(bid, R, w, z=None, wz=0.0, pitch=1.0):
                """Orientation to R; the pitch about R's y axis weighs `pitch` times less."""
                jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jp, jr, bid)
                Rc = d.xmat[bid].reshape(3, 3)
                e = 0.5 * sum(np.cross(Rc[:, k], R[:, k]) for k in range(3))
                S = np.diag([1.0, math.sqrt(pitch), 1.0])
                add(S @ R.T @ jr, S @ R.T @ e, w)
                if z is not None:
                    add(jp[2:3], np.array([z - d.xpos[bid][2]]), wz)

            if not arms_only:
                for s in ("left", "right"):
                    R = rz(tk[f"{s}_foot_yaw"]) @ ry(math.radians(tk.get(f"{s}_foot_pitch", 0.0)))
                    site_rows(f"{s}_foot", tk[f"{s}_foot"], R, 100.0, 30.0)
                jc = np.zeros((3, nv))
                mujoco.mj_jacSubtreeCom(m, d, jc, 1)
                add(jc[:2], tk["com"] - d.subtree_com[1][:2], 100.0)
                # Opt-in upright manipulation: do not trade torso posture or
                # standing height for an awkward palm target. Defaults preserve
                # the existing loaded and locomotion references.
                upright = getattr(self, "upright_manipulation", False)
                body_rot(self.body["pelvis"], rz(tk["yaw"]), 100.0 if upright else 5.0,
                         tk["pelvis_z"], 500.0 if upright else 20.0, pitch=1.0 if upright else 0.2)
                body_rot(self.body["torso_link"], rz(tk["yaw"]), 200.0 if upright else 2.0,
                         pitch=1.0 if upright else 0.2)
            wp = tk.get("palm_w", 1.0)          # palm tasks fade in/out against the rest posture
            for s in ("left", "right"):
                if tk.get(f"{s}_palm") is not None and wp > 0:
                    site_rows(f"{s}_palm", tk[f"{s}_palm"], rz(tk["yaw"]), 30.0 * wp, 1.0 * wp,
                              body=self.body[HAND[s]])
            # posture: legs and waist to the keyframe, arms to theirs (strong when untasked)
            nj = nv - 6
            P = np.zeros((nj, nv))
            P[np.arange(nj), 6 + np.arange(nj)] = 1.0
            qa = q[7:]
            w = np.where(self.leg, 0.02, 0.0) + np.where(self.waist, 2.0, 0.0)
            for s in ("left", "right"):
                ws = 3.0 if tk.get(f"{s}_palm") is None else 0.02 + 3.0 * (1.0 - wp)
                w = w + np.where(self.arm_side[s], ws, 0.0)
            tgt = tk.get("arm_rest", self.q_nom[7:])
            add(P * np.sqrt(w)[:, None], np.sqrt(w) * (tgt - qa), 1.0)
            J, e = np.vstack(Js), np.concatenate(es)
            if arms_only:
                cols = 6 + np.nonzero(self.arm)[0]
                Ja = J[:, cols]
                dqa = np.linalg.solve(Ja.T @ Ja + 1e-4 * np.eye(len(cols)), Ja.T @ e)
                dq = np.zeros(nv)
                dq[cols] = dqa
            else:
                dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(nv), J.T @ e)
            # Optional trust region for difficult high-reach references. Legacy
            # planners retain their original step size unless explicitly enabled.
            max_step = getattr(self, "max_ik_step", None)
            if max_step is not None:
                dq *= min(1.0, max_step / max(float(np.abs(dq).max()), 1e-12))
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[7:] = np.clip(q[7:], self.lo, self.hi)
        d = self.fk(q)
        err = {}
        for s in ("left", "right"):
            if not arms_only:
                err[f"{s}_foot"] = float(np.linalg.norm(tk[f"{s}_foot"] - d.site_xpos[self.site[f"{s}_foot"]]))
            if tk.get(f"{s}_palm") is not None and tk.get("palm_w", 1.0) >= 1.0:
                err[f"{s}_palm"] = float(np.linalg.norm(tk[f"{s}_palm"] - d.site_xpos[self.site[f"{s}_palm"]]))
        if not arms_only:
            err["com"] = float(np.linalg.norm(tk["com"] - d.subtree_com[1][:2]))
        return q, err


# ------------------------------------------------------------------ schedules

@dataclass
class Key:
    """One manipulation keyframe: time (s), the box pose (world, or None:
    unchanged), the palms' mode and the pelvis height."""
    t: float
    box: tuple | None          # (x, y, z, yaw) or None
    palms: str                 # "rest" | "ready" | "open" | "contact" | "away"
    pelvis_z: float = sc.PELVIS_Z
    grip: bool = False         # the hands squeeze the box from here to the next key


def _palm_offsets(mode):
    """Palm offsets from the box centre along the robot's left axis, or None."""
    w = sc.BOX_HALF[1] + sc.PALM_R
    return {"open": w + OPEN, "contact": w, "away": w + 0.06}.get(mode)


class Planner:
    def __init__(self, check_every=0):
        self.r = Robot()
        self.rest = self.r.rest_palms()
        self.check_every = check_every
        self.stats = mpc.QPStats()

    # -------------------------------------------------- feet over time
    def feet(self, gait, t):
        """(left xy, yaw, z, pitch), (right ...) at time t from the gait's footsteps.
        The human walk's forward steps leave the floor flat, swing with its sole's
        pitch and height (human_gait.json), strike with the heel and roll flat on
        it over the double support; every other step swings flat on a min-jerk arc."""
        st = getattr(gait, "style", None)
        k = t / mpc.TICK
        out = {}
        for side in (1, -1):
            idx = [i for i, f in enumerate(gait.steps) if f.side == side]
            cur = idx[0]
            pos = None
            for a, b in zip(idx[:-1], idx[1:]):
                lift, land = gait.lift[b], gait.steps[b].land
                if k >= land:
                    cur = b
                elif k > lift:
                    s = (k - lift) / (land - lift)
                    fa, fb = gait.steps[a], gait.steps[b]
                    e = smooth(s)
                    yaw = fa.yaw + wrap(fb.yaw - fa.yaw) * e
                    if st is not None and gait.fwd.get(b):
                        td = st.td_pitch
                        p = np.interp(s, st.sw_s, st.sw_pitch)
                        p += (1 - s) * (0.0 - st.sw_pitch[0]) + s * (td - st.sw_pitch[-1])
                        xy1, z1 = pivot(fb.xy, fb.yaw, td, gt.HEEL_X)
                        z = FOOT_Z + np.interp(s, st.sw_s, st.sw_z)
                        z += (1 - s) * (0.0 - st.sw_z[0]) + s * (z1 - FOOT_Z - st.sw_z[-1])
                        pos = (fa.xy + (xy1 - fa.xy) * e, yaw, z, float(p))
                    else:
                        xy = fa.xy + (fb.xy - fa.xy) * e
                        z = FOOT_Z + SWING_H * 64 * s ** 3 * (1 - s) ** 3
                        pos = (xy, yaw, z, 0.0)
                    break
                else:
                    break
            if pos is None:
                f = gait.steps[cur]
                pos = (f.xy, f.yaw, FOOT_Z, 0.0)
                if st is not None and gait.fwd.get(cur) and k < f.land + gait.k_ds:
                    p = st.td_pitch * (1.0 - smooth((k - f.land) / gait.k_ds))
                    xy, z = pivot(f.xy, f.yaw, p, gt.HEEL_X)
                    pos = (xy, f.yaw, z, float(p))
            out["left" if side > 0 else "right"] = pos
        return out

    # -------------------------------------------------- one plan
    def plan(self, start_pose, vias, keys, q0=None, box0=None, t_pre=0.4, t_post=0.6, seconds=None, stand=False,
             carry=False):
        """Plan a motion: stand t_pre s, walk from start_pose through vias (or stand), with
        the manipulation keys; carry=True holds the box at CARRY_BOX throughout."""
        if stand:
            gait, k_walk, (rp, rv) = gt.stand_gait(start_pose, seconds)
        else:
            gait, k_walk, (rp, rv) = gt.make_gait(start_pose, vias, t_pre=t_pre, t_post=t_post)
        T_end = (len(gait.support) - 1) * mpc.TICK
        # the reference CoM is the robot's alone; with the box in hand, the MPC
        # plans the CoM of both, which sits BOX_MASS/(M+m) of the way to the box
        c0 = 0.5 * (gait.steps[0].xy + gait.steps[1].xy)
        nominal = np.array([[f.xy[0], f.xy[1], f.yaw, f.side] for f in gait.steps], np.float32)
        X, _ = mpc.run(gait, rp, rv, c0, self.stats, self.check_every)
        r = self.r
        nf = int(round(T_end * FPS)) + 1
        q = (q0 if q0 is not None else self._initial_q(start_pose)).copy()
        out = {k: [] for k in ("qpos", "qcmd", "box", "grip", "err")}
        box = np.array(box0 if box0 is not None else (0.0, 0.0, BOX_Z0, 0.0), float)
        keys = sorted(keys, key=lambda k: k.t)
        for f in range(nf):
            t = f * DT
            ft = self.feet(gait, t)
            yaw = math.atan2(math.sin(ft["left"][1]) + math.sin(ft["right"][1]),
                             math.cos(ft["left"][1]) + math.cos(ft["right"][1]))
            com, _ = mpc.com_at(X, t)
            tk = {"yaw": yaw}
            for s in ("left", "right"):
                xy, fy, z, fp = ft[s]
                tk[f"{s}_foot"] = np.array([xy[0], xy[1], z])
                tk[f"{s}_foot_yaw"] = fy
                tk[f"{s}_foot_pitch"] = fp
            # manipulation: interpolate the keys around t
            k_prev = max([k for k in keys if k.t <= t], key=lambda k: k.t, default=keys[0])
            k_next = min([k for k in keys if k.t > t], key=lambda k: k.t, default=None)
            s = 0.0 if k_next is None else smooth((t - k_prev.t) / (k_next.t - k_prev.t))
            pz = k_prev.pelvis_z + (0.0 if k_next is None else (k_next.pelvis_z - k_prev.pelvis_z) * s)
            pz += gt.pelvis_dz(gait, t)          # the human walk's rise and bob (0 otherwise)
            tk["pelvis_z"] = pz
            pel = r.fk(q).xpos[r.body["pelvis"]].copy()
            pel_xy = pel[:2]
            if carry:
                bx = np.array([pel_xy[0], pel_xy[1], 0.0]) + rz(yaw) @ np.array(CARRY_BOX)
                bx[2] = pz + CARRY_BOX[2]
                box = np.array([bx[0], bx[1], bx[2], yaw + math.pi])
            else:
                b0 = self._box_key(k_prev, pel_xy, yaw, pz, box)
                b1 = self._box_key(k_next, pel_xy, yaw, pz, box) if k_next is not None else b0
                box = b0 + (b1 - b0) * s
                box[3] = b0[3] + wrap(b1[3] - b0[3]) * s
            palms = self._palms(k_prev, k_next, s, box, pel_xy, yaw, pz, carry)
            tk.update(palms)
            held = carry or k_prev.grip
            box_xy = box[:2]
            tk["com"] = com if not held else ((r.mass + BOX_MASS) * com - BOX_MASS * box_xy) / r.mass
            q, err = r.solve(q, tk, iters=30 if f == 0 else int(os.environ.get("IKIT", 10)))
            qc = q.copy()
            if held:   # the joint targets: palms SQUEEZE inside the faces
                tc = dict(tk)
                for side, sg in (("left", 1.0), ("right", -1.0)):
                    tc[f"{side}_palm"] = tk[f"{side}_palm"] - sg * sc.SQUEEZE * rz(yaw)[:, 1]
                qc, _ = r.solve(q, tc, iters=6, arms_only=True)
            out["qpos"].append(q.copy())
            out["qcmd"].append(qc[7:].copy())
            out["box"].append(box.copy())
            out["grip"].append(held)
            out["err"].append(max(err.values()))
        packed = self._pack(out, gait, X)
        packed["nominal"] = nominal          # the footsteps before the MPC moved them
        return packed

    def _initial_q(self, pose):
        q = self.r.q_nom.copy()
        q[0], q[1] = pose[0], pose[1]
        q[3:7] = quat_z(pose[2])
        return q

    def _box_key(self, k, pel_xy, yaw, pz, box):
        """The box pose a key asks for: given (world), or the current one."""
        if k.box is None:
            return box.copy()
        if k.box == "carry":
            bx = np.array([pel_xy[0], pel_xy[1], 0.0]) + rz(yaw) @ np.array(CARRY_BOX)
            return np.array([bx[0], bx[1], pz + CARRY_BOX[2], yaw + math.pi])
        return np.array(k.box, float)

    def _palms(self, k0, k1, s, box, pel_xy, yaw, pz, carry):
        """Palm targets (world) between two keys."""
        def at(k):
            mode = "contact" if carry else k.palms
            if mode == "rest":
                return None
            left = rz(yaw)[:, 1]
            if mode == "ready" or isinstance(mode, tuple):
                base = np.array([pel_xy[0], pel_xy[1], pz])
                target = READY if mode == "ready" else mode
                return {sd: base + rz(yaw) @ np.array([target[0], sg * target[1], target[2]])
                        for sd, sg in (("left", 1), ("right", -1))}
            off = _palm_offsets(mode)
            return {sd: box[:3] + sg * off * left for sd, sg in (("left", 1), ("right", -1))}

        def rest_world():
            base = np.array([pel_xy[0], pel_xy[1], pz])
            return {sd: base + rz(yaw) @ self.rest[sd] for sd in ("left", "right")}

        a = at(k0)
        b = at(k1) if k1 is not None else a
        if a is None and b is None:
            return {"left_palm": None, "right_palm": None}
        # to or from rest: the palm tasks fade against the arms' rest posture
        w = 1.0 if (a is not None and b is not None) else (s if a is None else 1.0 - s)
        a = a if a is not None else rest_world()
        b = b if b is not None else rest_world()
        out = {f"{sd}_palm": a[sd] + (b[sd] - a[sd]) * s for sd in ("left", "right")}
        out["palm_w"] = float(w)
        return out

    # -------------------------------------------------- output
    def _pack(self, out, gait, X):
        r = self.r
        Q = np.array(out["qpos"])
        nf = len(Q)
        bodies = np.zeros((nf, len(TRACKED), 7))
        for f in range(nf):
            d = r.fk(Q[f])
            for j, n in enumerate(TRACKED):
                bid = r.body[n]
                bodies[f, j, :3] = d.xpos[bid]
                bodies[f, j, 3:] = d.xquat[bid]
        qvel = np.zeros((nf, r.m.nv))
        for f in range(nf):
            a, b = max(f - 1, 0), min(f + 1, nf - 1)
            mujoco.mj_differentiatePos(r.m, qvel[f], (b - a) * DT, Q[a], Q[b])
        # free joint: MuJoCo's angular velocity is in the body frame; store it in the world
        for f in range(nf):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, Q[f, 3:7])
            qvel[f, 3:6] = R.reshape(3, 3) @ qvel[f, 3:6]
        blin = np.gradient(bodies[:, :, :3], DT, axis=0)
        bang = np.zeros((nf, len(TRACKED), 3))
        for j in range(len(TRACKED)):
            qs = bodies[:, j, 3:]
            for f in range(nf):
                a, b = max(f - 1, 0), min(f + 1, nf - 1)
                v = np.zeros(3)
                mujoco.mju_subQuat(v, qs[b], qs[a])          # body-frame rotation vector
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, qs[f])
                bang[f, j] = R.reshape(3, 3) @ v / max((b - a) * DT, DT)
        box = np.array(out["box"])
        box_lin = np.gradient(box[:, :3], DT, axis=0)
        box_wz = np.gradient(np.unwrap(box[:, 3]), DT)
        feet_z = np.array([[r.fk(Q[f]).site_xpos[r.site[s]][2] for s in ("left_foot", "right_foot")]
                           for f in range(nf)])
        return {"qpos": Q.astype(np.float32), "qcmd": np.array(out["qcmd"], np.float32),
                "qvel": qvel.astype(np.float32), "body": bodies.astype(np.float32),
                "body_lin": blin.astype(np.float32), "body_ang": bang.astype(np.float32),
                "box": box.astype(np.float32), "box_lin": box_lin.astype(np.float32),
                "box_wz": box_wz.astype(np.float32), "grip": np.array(out["grip"]),
                "contact": (feet_z < FOOT_Z + 0.005), "ik_err": np.array(out["err"], np.float32),
                "com": X[:, :, 0].astype(np.float32),
                "zmp": (X[:, :, 0] - (mpc.H / mpc.G if gait.eta is None else gait.eta[:, None]) * X[:, :, 2]
                        ).astype(np.float32),
                "steps": np.array([[f.xy[0], f.xy[1], f.yaw, f.side] for f in gait.steps], np.float32)}


# ------------------------------------------------------------------ the three skills

def fetch_plan(P: Planner, start):
    """Walk from start (box frame) to the stance, then reach, close, lift, carry pose."""
    manip = [("open", 0.7), ("contact", 0.4), ("lift", 0.6), ("carry", 0.6), ("hold", 0.4)]
    t_manip = sum(d for _, d in manip)
    probe, _, _ = gt.make_gait(start, [STANCE], t_post=0.6)
    t_walk = (len(probe.support) - 1) * mpc.TICK
    t = t_walk - 0.4                       # the reach starts as the robot settles
    keys = [Key(0.0, None, "rest"), Key(max(0.3, t_walk - 1.6), None, "rest"), Key(t, None, "ready")]
    box0 = (0.0, 0.0, BOX_Z0, 0.0)
    for name, dur in manip:
        t += dur
        if name == "open":
            keys.append(Key(t, box0, "open"))
        elif name == "contact":
            keys.append(Key(t, box0, "contact", grip=True))
        elif name == "lift":
            keys.append(Key(t, (0.0, 0.0, BOX_Z0 + LIFT, 0.0), "contact", grip=True))
        elif name == "carry":
            keys.append(Key(t, "carry", "contact", grip=True))
        else:
            keys.append(Key(t, "carry", "contact", grip=True))
    # the key BEFORE closing has no grip; closing ends gripped
    plan = P.plan(start, [STANCE], keys, t_post=0.6 + t_manip - 0.4 + 0.2)
    plan["events"] = {"walk_end": t_walk}
    return plan


def carry_plan(P: Planner, q0, shelf_stance):
    """From the fetch's end: back up from the cart, turn to the shelf, walk to its stance."""
    x0, y0 = q0[0], q0[1]
    yaw0 = STANCE[2]
    back = (x0 + BACK_UP * math.cos(yaw0 + math.pi), y0 + BACK_UP * math.sin(yaw0 + math.pi), yaw0)
    gx, gy, gyaw = shelf_stance
    travel = math.atan2(gy - back[1], gx - back[0])
    d = math.hypot(gx - back[0], gy - back[1])
    turned = (back[0], back[1], travel)
    pre = (gx - 0.5 * math.cos(travel), gy - 0.5 * math.sin(travel), travel) if d > 0.8 else turned
    vias = [back, turned, pre, (gx, gy, gyaw)]
    vias[0] = (back[0], back[1], yaw0 + 0.5 * wrap(travel - yaw0))   # people turn toward where they go while backing away
    start = (x0, y0, yaw0)
    keys = [Key(0.0, "carry", "contact", grip=True)]
    return P.plan(start, vias, keys, q0=q0, carry=True)


def place_plan(P: Planner, q0):
    """At the shelf stance (box frame on the shelf): lower, set down, let go, arms down."""
    t = 0.0
    keys = [Key(t, "carry", "contact", grip=True)]
    t += 1.1
    keys.append(Key(t, (0.0, 0.0, BOX_REST_Z + 0.02, 0.0), "contact", sc.PELVIS_Z - PLACE_DIP, grip=True))
    t += 0.5
    keys.append(Key(t, (0.0, 0.0, BOX_REST_Z, 0.0), "contact", sc.PELVIS_Z - PLACE_DIP))
    t += 0.5
    keys.append(Key(t, None, "away", sc.PELVIS_Z - PLACE_DIP))
    t += 1.0
    keys.append(Key(t, None, "rest"))
    t += 0.8
    keys.append(Key(t, None, "rest"))
    return P.plan(STANCE, [], keys, q0=q0, box0=(0.0, 0.0, 0.0, 0.0), stand=True, seconds=t)


# ------------------------------------------------------------------ layouts

def sample_fetch_start(rng):
    d = rng.uniform(*START_DIST)
    lat = d * rng.uniform(-START_LAT, START_LAT)
    yaw = math.pi + d * rng.uniform(-START_YAW, START_YAW)
    return (STANCE[0] + d, lat, yaw)


def sample_shelf(rng):
    """The shelf's box frame in the cart's box frame: place point xy and its yaw
    (x out of the shelf), and the stance in front of it (cart's box frame)."""
    side = rng.choice((-1.0, 1.0))
    ang = STANCE[2] + side * math.radians(rng.uniform(*SHELF_ANGLE))
    r = rng.uniform(*SHELF_R)
    px, py = STANCE[0] + r * math.cos(ang), STANCE[1] + r * math.sin(ang)
    syaw = ang + math.pi + rng.uniform(-SHELF_YAW_NOISE, SHELF_YAW_NOISE)
    stance = (px + sc.HOLD_X * math.cos(syaw), py + sc.HOLD_X * math.sin(syaw), syaw + math.pi)
    return (px, py, syaw), stance


# ------------------------------------------------------------------ kinematic playback

def show_model(cart, shelf):
    """The G1, a floor, the box (free) and the props at (x, y, yaw) in one MuJoCo model."""
    spec = g1.get_spec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(8, 8, 0.1), rgba=(0.55, 0.55, 0.52, 1))
    spec.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8))
    for name, fn, pose in (("cart", sc.cart_spec, cart), ("shelf", sc.shelf_spec, shelf), ("box", sc.box_spec, None)):
        if name != "box" and pose is None:
            continue
        x, y, yaw = pose if pose is not None else (0.0, 0.0, 0.0)
        fr = spec.worldbody.add_frame(pos=(x, y, 0.0), quat=quat_z(yaw))
        fr.attach_body(fn().body(name), name + "/", "")
    return spec.compile()


def show(frames, out, cart=None, shelf=None, every=2, size=(360, 640), distance=3.2, azimuth=150.0):
    """A GIF of a plan (qpos and box per frame), played kinematically."""
    from PIL import Image
    m = show_model(cart, shelf)
    d = mujoco.MjData(m)
    nq = len(frames["qpos"][0])                     # the robot's; the box's free joint follows
    rnd = mujoco.Renderer(m, *size)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, -20.0
    imgs = []
    for f in range(0, len(frames["qpos"]), every):
        d.qpos[:nq] = frames["qpos"][f]
        b = frames["box"][f]
        d.qpos[nq:nq + 3] = b[:3]
        d.qpos[nq + 3:nq + 7] = quat_z(float(b[3]))
        mujoco.mj_forward(m, d)
        cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.6)
        rnd.update_scene(d, cam)
        imgs.append(Image.fromarray(rnd.render()))
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 * every / FPS), loop=0)
    return out


def transform_q(q, frm, to):
    """A qpos in frame `frm` (x, y, yaw in world) re-expressed in frame `to`."""
    q = q.copy()
    c = np.array(frm[:2]) + rz(frm[2])[:2, :2] @ q[:2]
    q[:2] = rz(-to[2])[:2, :2] @ (c - np.array(to[:2]))
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, quat_z(frm[2] - to[2]), q[3:7].copy())
    q[3:7] = dq
    return q


def chain(P: Planner, rng):
    """One layout's three plans: fetch, carry (cart's box frame) and place (shelf's)."""
    start = sample_fetch_start(rng)
    (px, py, syaw), stance = sample_shelf(rng)
    f = fetch_plan(P, start)
    c = carry_plan(P, f["qpos"][-1].astype(np.float64), stance)
    shelf_frame = (px, py, syaw)
    q0 = transform_q(c["qpos"][-1].astype(np.float64), (0.0, 0.0, 0.0), shelf_frame)
    p = place_plan(P, q0)
    return {"start": start, "shelf": shelf_frame, "fetch": f, "carry": c, "place": p}


def _bank_worker(args):
    seed, n = args
    P = Planner(check_every=10)
    rng = np.random.default_rng(seed)
    chains = [chain(P, rng) for _ in range(n)]
    st = P.stats
    return chains, (st.n, st.secs, dict(st.statuses), st.check_n, st.check_max_rel, st.check_secs)


SKILL_KEYS = ("qpos", "qcmd", "qvel", "body", "body_lin", "body_ang", "box", "box_lin", "box_wz", "grip", "contact")


def make_bank(n, seed, out, workers=8):
    """n layouts (chains of fetch, carry, place plans), planned in parallel."""
    import multiprocessing as mp
    import torch
    per = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_bank_worker, [(seed * 1000 + i, k) for i, k in enumerate(per) if k > 0])
    wall = time.time() - t0
    chains = [c for cs, _ in res for c in cs]
    qp = {"n": sum(r[1][0] for r in res), "rekin_s": sum(r[1][1] for r in res),
          "statuses": {}, "checked": sum(r[1][3] for r in res),
          "max_rel_cost_vs_qpoases": max(r[1][4] for r in res), "qpoases_s": sum(r[1][5] for r in res)}
    for r in res:
        for k, v in r[1][2].items():
            qp["statuses"][k] = qp["statuses"].get(k, 0) + v
    bank = {"joint_names": Robot().joints, "body_names": list(TRACKED), "fps": FPS,
            "start": torch.tensor([c["start"] for c in chains], dtype=torch.float32),
            "shelf": torch.tensor([c["shelf"] for c in chains], dtype=torch.float32),
            "stats": {"layouts": len(chains), "wall_s": wall, "workers": workers, "qp": qp}}
    for k in ("fetch", "carry", "place"):
        lens = [len(c[k]["qpos"]) for c in chains]
        T = max(lens)
        sk = {"len": torch.tensor(lens)}
        for key in SKILL_KEYS:
            arrs = []
            for c in chains:
                a = c[k][key]
                pad = np.repeat(a[-1:], T - len(a), axis=0)
                arrs.append(np.concatenate([a, pad], 0))
            sk[key] = torch.from_numpy(np.stack(arrs))
        sk["ik_err"] = torch.tensor([float(c[k]["ik_err"].max()) for c in chains])
        bank[k] = sk
        bank["stats"][k] = {"frames_mean": float(np.mean(lens)), "frames_max": T,
                            "ik_err_max": float(sk["ik_err"].max()), "ik_err_median": float(sk["ik_err"].median())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(bank, out)
    return bank["stats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None, help="fetch | carry | place | all: a GIF of one random layout's plans")
    ap.add_argument("--bank", action="store_true", help="plan the training and evaluation layouts")
    ap.add_argument("--train", type=int, default=512)
    ap.add_argument("--eval", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.bank:
        import json
        for name, n, seed in (("plans", a.train, 1), ("plans_eval", a.eval, 2)):
            st = make_bank(n, seed, os.path.join(sc.out_dir("banks"), f"{name}.pt"), a.workers)
            print(name, json.dumps(st, indent=1), flush=True)
        return
    if a.show:
        rng = np.random.default_rng(a.seed)
        P = Planner(check_every=4)
        t0 = time.time()
        ch = chain(P, rng)
        print(f"planned in {time.time() - t0:.1f} s; {P.stats.n} QPs, Rekin {P.stats.secs:.2f} s, "
              f"{P.stats.statuses}; vs qpOASES on {P.stats.check_n}: max rel cost diff {P.stats.check_max_rel:.1e}")
        for k in ("fetch", "carry", "place"):
            e = ch[k]["ik_err"]
            print(k, "frames", len(e), "IK err max %.4f p99 %.4f" % (e.max(), np.percentile(e, 99)))
        cart = (-sc.BOX_ON_PROP_X, 0.0, 0.0)
        px, py, syaw = ch["shelf"]
        shelf = (px - sc.BOX_ON_PROP_X * math.cos(syaw), py - sc.BOX_ON_PROP_X * math.sin(syaw), syaw)
        os.makedirs(sc.out_dir("media"), exist_ok=True)
        ks = ("fetch", "carry", "place") if a.show == "all" else (a.show,)
        for k in ks:
            fr = ch[k]
            if k == "place":   # back into the cart's frame for the picture
                fr = dict(fr)
                fr["qpos"] = np.array([transform_q(q.astype(np.float64), ch["shelf"], (0, 0, 0)) for q in fr["qpos"]])
                bx = fr["box"].astype(np.float64).copy()
                bx[:, :2] = np.array(ch["shelf"][:2]) + bx[:, :2] @ rz(ch["shelf"][2])[:2, :2].T
                bx[:, 3] += ch["shelf"][2]
                fr["box"] = bx
            out = show(fr, os.path.join(sc.out_dir("media"), f"plan_{k}.gif"), cart=cart, shelf=shelf)
            print("wrote", out)


if __name__ == "__main__":
    main()
