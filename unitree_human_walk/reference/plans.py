#!/usr/bin/env python3
"""plans.py — the motion: a person's walk, from standing to standing.  No MPC.

Runs in the `mjlab` env.  The reference is one segment of a human motion-capture clip,
Unitree's retargeting of LAFAN1 to the 29-DoF G1 (30 fps): the person stands, walks about
2.2 m straight ahead and stops.  It becomes a plan for the 23-DoF G1 in the format of
the MPC's plans (../policy-go-to/plans.py), so ../motion.py trains on it unchanged:

  1. the clip's frames START..END, the stillest frames either side of the walk;
  2. the 23-DoF pose per frame: the root and the 23 shared joints copied (the two models'
     legs are the same, so the feet land where the clip's do), then inverse kinematics
     for what the 6 missing joints did (waist roll and pitch, wrist pitch and yaw): the
     feet stay, and the pelvis leans to bring the torso and the palms near the clip's;
  3. the plan's frame: the origin on the floor under the pelvis at START, x along its
     heading; the height shifted so the lower sole's lowest edge is on the floor (the
     median over the clip's frames);
  4. 50 Hz by cubic splines, with the clip's clock eased in and out over EASE s, so the
     plan starts and ends at rest; it stands T_PRE s before and T_POST s after.

The clips are not in the repository (LAFAN1's licence is CC BY-NC-ND 4.0):

  python plans.py --download   # data/lafan1/g1/<clip>.csv, from Hugging Face
  python plans.py              # banks/plan.pt, for ../motion.py
  python plans.py --show       # media/plan.gif: the plan, played kinematically
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

import gait as gt  # noqa: E402
import plan as pl  # noqa: E402

CLIP = "walk3_subject1"
START, END = 6679, 6821            # frames of the clip: standing, the walk, standing
CLIP_FPS = 30
URL = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{}.csv"
T_PRE, T_POST, EASE = 0.4, 1.0, 0.3     # s
TITLE = "Unitree G1 (23 DoF) · a person's walk (LAFAN1), stand to stand"
SOURCE = "a person's walk (LAFAN1), retargeted"
KEYS = ("qpos", "qcmd", "qvel", "body", "body_lin", "body_ang", "contact")
# inverse kinematics: weights on the squared errors (m, rad)
W_FOOT, W_FOOT_ROT, W_PELVIS, W_PELVIS_ROT, W_TORSO_ROT, W_PALM = 100.0, 30.0, 20.0, 2.0, 5.0, 5.0
W_POSTURE = {"leg": 0.05, "waist": 0.5, "arm": 0.1}      # towards the clip's own joint angles
SOLE = (gt.HEEL_X, gt.TOE_X)       # m, the sole's back and front edges from the foot site
ON_FLOOR = (0.01, 0.015)           # m: a sole's lowest edge and its foot site this low: on the floor


def clip_path(name):
    return os.path.join(HERE, "data", "lafan1", "g1", name + ".csv")


def download(name):
    os.makedirs(os.path.dirname(clip_path(name)), exist_ok=True)
    urllib.request.urlretrieve(URL.format(name), clip_path(name))
    return clip_path(name)


def g1_29dof():
    """mjlab's 29-DoF G1, the model the clips were retargeted to."""
    import mjlab
    xml = os.path.join(os.path.dirname(mjlab.__file__), "asset_zoo", "robots", "unitree_g1", "xmls", "g1.xml")
    return mujoco.MjModel.from_xml_path(xml)


def load_clip(name, start, end):
    """The clip's frames start..end as the 29-DoF model's qpos: the CSV's root quaternion
    is (x, y, z, w), its joints are in the model's order."""
    a = np.loadtxt(clip_path(name), delimiter=",")[start:end + 1]
    return np.concatenate([a[:, :3], a[:, 6:7], a[:, 3:6], a[:, 7:]], 1)


def rot_err(Rc, R):
    """The rotation vector (world) from Rc to R, to first order."""
    return 0.5 * sum(np.cross(Rc[:, k], R[:, k]) for k in range(3))


class Retarget:
    """A 29-DoF pose on the 23-DoF G1."""

    def __init__(self):
        self.r = r = pl.Robot()
        self.m29 = m = g1_29dof()
        self.d29 = mujoco.MjData(m)
        j29 = [m.joint(i).name for i in range(1, m.njnt)]
        self.shared = np.array([j29.index(n) for n in r.joints])
        self.sites = ("left_foot", "right_foot", "left_palm", "right_palm")
        w = np.where(r.leg, W_POSTURE["leg"], 0.0) + np.where(r.waist, W_POSTURE["waist"], 0.0)
        self.w_post = w + np.where(r.arm, W_POSTURE["arm"], 0.0)

    def targets(self, q29):
        m, d = self.m29, self.d29
        d.qpos[:] = q29
        mujoco.mj_kinematics(m, d)
        t = {s: (d.site_xpos[m.site(s).id].copy(), d.site_xmat[m.site(s).id].reshape(3, 3).copy())
             for s in self.sites}
        for b in ("pelvis", "torso_link"):
            t[b] = (d.xpos[m.body(b).id].copy(), d.xmat[m.body(b).id].reshape(3, 3).copy())
        return t

    def solve(self, q29, iters=12):
        """The 23-DoF qpos for a 29-DoF one, and the errors left: each foot and palm (m),
        the torso's orientation (rad), the pelvis (m, rad)."""
        r = self.r
        m, d, nv = r.m, r.d, r.m.nv
        t = self.targets(q29)
        q0 = np.concatenate([q29[:7], q29[7:][self.shared]])
        q0[7:] = np.clip(q0[7:], r.lo, r.hi)
        q = q0.copy()
        P = np.zeros((nv - 6, nv))
        P[np.arange(nv - 6), 6 + np.arange(nv - 6)] = 1.0
        jp, jr = np.zeros((3, nv)), np.zeros((3, nv))

        def errors():
            r.fk(q)
            e = {s: float(np.linalg.norm(t[s][0] - d.site_xpos[r.site[s]])) for s in self.sites}
            for b in ("pelvis", "torso_link"):
                e[b + "_rot"] = float(np.linalg.norm(rot_err(d.xmat[r.body[b]].reshape(3, 3), t[b][1])))
            e["pelvis"] = float(np.linalg.norm(t["pelvis"][0] - d.xpos[r.body["pelvis"]]))
            return e

        for _ in range(iters):
            r.fk(q)
            Js, es = [], []

            def add(J, e, w):
                Js.append(np.sqrt(w) * J)
                es.append(np.sqrt(w) * e)

            for s in self.sites:
                sid = r.site[s]
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                add(jp.copy(), t[s][0] - d.site_xpos[sid], W_FOOT if "foot" in s else W_PALM)
                if "foot" in s:
                    add(jr.copy(), rot_err(d.site_xmat[sid].reshape(3, 3), t[s][1]), W_FOOT_ROT)
            for b, wp, wr in (("pelvis", W_PELVIS, W_PELVIS_ROT), ("torso_link", 0.0, W_TORSO_ROT)):
                bid = r.body[b]
                mujoco.mj_jacBody(m, d, jp, jr, bid)
                if wp:
                    add(jp.copy(), t[b][0] - d.xpos[bid], wp)
                add(jr.copy(), rot_err(d.xmat[bid].reshape(3, 3), t[b][1]), wr)
            add(P * np.sqrt(self.w_post)[:, None], np.sqrt(self.w_post) * (q0[7:] - q[7:]), 1.0)
            J, e = np.vstack(Js), np.concatenate(es)
            dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(nv), J.T @ e)
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[7:] = np.clip(q[7:], r.lo, r.hi)
        return q, errors()


def soles(r, Q):
    """Per frame and foot: the foot site (nf, 2, 3) and the height of the sole's lowest
    edge above the floor (nf, 2), heel or toe.  The site is FOOT_Z above the sole."""
    site = np.zeros((len(Q), 2, 3))
    low = np.zeros((len(Q), 2))
    for f, q in enumerate(Q):
        d = r.fk(q)
        for i, s in enumerate(("left_foot", "right_foot")):
            sid = r.site[s]
            site[f, i] = d.site_xpos[sid]
            ax = d.site_xmat[sid].reshape(3, 3)[2, 0]         # the sole's x axis, its rise
            low[f, i] = site[f, i, 2] - pl.FOOT_Z + min(SOLE[0] * ax, SOLE[1] * ax)
    return site, low


def clock(n_clip):
    """The clip's time (s) at each of the plan's frames: stands T_PRE s, plays the clip
    with its rate eased 0 -> 1 -> 0 over EASE s at each end, stands T_POST s."""
    D = (n_clip - 1) / CLIP_FPS
    S = lambda x: np.clip(x, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(x, 0.0, 1.0))  # noqa: E731
    total = T_PRE + D + EASE + T_POST
    t = np.arange(0.0, total + 1e-9, 1e-3)
    rate = S((t - T_PRE) / EASE) * S((T_PRE + D + EASE - t) / EASE)
    u = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * 1e-3)])
    tf = np.arange(int(round(total * pl.FPS)) + 1) / pl.FPS
    return np.clip(np.interp(tf, t, u), 0.0, D), T_PRE + D + EASE


def motion(name=CLIP, start=START, end=END):
    """The plan (the planner's keys), its phases and its numbers."""
    rt = Retarget()
    r = rt.r
    Q29 = load_clip(name, start, end)
    # 2. the 23-DoF pose per frame
    Q, errs = [], []
    for q29 in Q29:
        q, e = rt.solve(q29)
        Q.append(q)
        errs.append(e)
    Q = np.array(Q)
    copied = np.concatenate([Q29[:, :7], Q29[:, 7:][:, rt.shared]], 1)
    # 3. the plan's frame, and the soles on the floor
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, Q[0, 3:7])
    yaw0 = float(np.arctan2(R0[3], R0[0]))
    Q = np.array([pl.transform_q(q, (0.0, 0.0, 0.0), (Q[0, 0], Q[0, 1], yaw0)) for q in Q])
    site, low = soles(r, Q)
    dz = -float(np.median(low.min(1)))
    Q[:, 2] += dz
    # 4. 50 Hz, at rest at both ends
    for f in range(1, len(Q)):                     # one sign of the quaternion throughout
        if Q[f, 3:7] @ Q[f - 1, 3:7] < 0:
            Q[f, 3:7] *= -1
    u, t_stop = clock(len(Q))
    Q = CubicSpline(np.arange(len(Q)) / CLIP_FPS, Q, axis=0)(u)
    Q[:, 3:7] /= np.linalg.norm(Q[:, 3:7], axis=1, keepdims=True)
    qvel, bodies, blin, bang = pl.kinematics(r, Q)
    site, low = soles(r, Q)
    on = (low < ON_FLOOR[0]) & (site[:, :, 2] < ON_FLOOR[1])
    slide = np.linalg.norm(np.gradient(site[:, :, :2], pl.DT, axis=0), axis=-1)
    span = (r.hi - r.lo) / 0.95
    margin = np.minimum(Q[:, 7:] - (r.lo - 0.025 * span), (r.hi + 0.025 * span) - Q[:, 7:]).min(0)
    xy = Q[:, :2]
    total = (len(Q) - 1) / pl.FPS
    phases = [("stand", 0.0, "stand"), ("walk", T_PRE, "walk as the person in the clip walks, stop"),
              ("rest", t_stop, "stand"), ("end", total, None)]
    p = {"qpos": Q.astype(np.float32), "qcmd": Q[:, 7:].astype(np.float32), "qvel": qvel.astype(np.float32),
         "body": bodies.astype(np.float32), "body_lin": blin.astype(np.float32),
         "body_ang": bang.astype(np.float32), "contact": on}
    mx = lambda k: round(max(e[k] for e in errs), 4)  # noqa: E731
    stats = {
        "clip": name, "frames": [start, end], "clip_fps": CLIP_FPS, "seconds": round(total, 2),
        "phases_s": {n: round(t, 2) for n, t, _ in phases},
        "walk_m": round(float(np.linalg.norm(xy[-1] - xy[0])), 4),
        "walk_to": [round(float(v), 3) for v in xy[-1]],
        "pelvis_speed_max_mps": round(float(np.linalg.norm(qvel[:, :2], axis=1).max()), 3),
        "pelvis_z_m": [round(float(Q[:, 2].min()), 3), round(float(Q[:, 2].max()), 3)],
        "floor_shift_m": round(dz, 4),
        "retarget": {
            "foot_err_m_max": max(mx("left_foot"), mx("right_foot")),
            "palm_err_m_max": max(mx("left_palm"), mx("right_palm")),
            "torso_rot_err_rad_max": mx("torso_link_rot"), "pelvis_rot_err_rad_max": mx("pelvis_rot"),
            "pelvis_err_m_max": mx("pelvis"),
            "torso_rot_err_rad_max_if_only_copied": round(float(max(
                np.linalg.norm(rot_err(r.fk(c).xmat[r.body["torso_link"]].reshape(3, 3), rt.targets(q29)["torso_link"][1]))
                for c, q29 in zip(copied, Q29))), 4)},
        "sole_lowest_m": round(float(low.min()), 4),
        "sole_below_floor_share": round(float((low < -0.002).any(1).mean()), 4),
        "sole_on_floor_slide_mps": {"median": round(float(np.median(slide[on])), 3),
                                    "p95": round(float(np.percentile(slide[on], 95)), 3),
                                    "max": round(float(slide[on].max()), 3)},
        "joint_limit_margin_rad_min": round(float(margin.min()), 3),
        "joint_limit_margin_worst": r.joints[int(margin.argmin())],
        "joint_speed_max_radps": round(float(np.abs(qvel[:, 6:]).max()), 2),
        "joint_speed_max_joint": r.joints[int(np.abs(qvel[:, 6:]).max(0).argmax())],
    }
    walk = {"from": (0.0, 0.0), "to": (float(xy[-1, 0]), float(xy[-1, 1]))}
    return p, phases, walk, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch the clip from Hugging Face")
    ap.add_argument("--show", action="store_true", help="a GIF of the plan, played kinematically")
    a = ap.parse_args()
    if a.download:
        print("wrote", download(CLIP))
        return
    if not os.path.exists(clip_path(CLIP)):
        sys.exit(f"{clip_path(CLIP)} is missing: python plans.py --download")
    import torch
    t0 = time.time()
    p, phases, walk, stats = motion()
    stats["planned_in_s"] = round(time.time() - t0, 1)
    q = p["qpos"]
    bank = {"joint_names": pl.Robot().joints, "body_names": list(pl.TRACKED), "fps": pl.FPS, "title": TITLE,
            "source": SOURCE, "phases": phases, "hold": None, "walk": walk, "stats": stats,
            "plan": {"len": torch.tensor([len(q)]), **{k: torch.from_numpy(np.asarray(p[k]))[None] for k in KEYS}}}
    os.makedirs(os.path.join(HERE, "banks"), exist_ok=True)
    torch.save(bank, os.path.join(HERE, "banks", "plan.pt"))
    print(json.dumps(stats, indent=1))
    if a.show:
        os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
        frames = dict(p, box=np.tile([0.0, 0.0, -5.0, 0.0], (len(q), 1)))      # the planner's box: out of sight
        print("wrote", pl.show(frames, os.path.join(HERE, "media", "plan.gif"), distance=4.2, azimuth=100.0))


if __name__ == "__main__":
    main()
