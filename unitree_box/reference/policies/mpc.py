"""mpc.py — the walking MPC, solved by Rekin.

Runs in the `mjlab` env, with `rekin` installed (pip install rekin).

The model is Kajita's linear inverted pendulum: the centre of mass (robot and
box together) at a constant height H, its horizontal motion driven by the
zero-moment point (ZMP) z = c − (H/g) c̈.  The MPC's state per axis is
(c, ċ, c̈); its input is the jerk, constant over a TICK.  Over a horizon of
N ticks it chooses the jerks AND the positions of the footsteps that land
inside the horizon (Herdt et al. 2010, "Online walking motion generation
with automatic foot step placement"), minimising

    Σ_k  W_POS |c_k − c_ref,k|² + W_VEL |ċ_k − ċ_ref,k|² + W_ZMP |z_k − m_k|²
         + W_JERK |j_k|²  +  Σ_i W_STEP |f_i − f_nom,i|²

subject to, at every tick, the ZMP inside the support (the stance foot's
sole, or in double support a box around the two feet) and every new footstep
inside the reach of the foot before it.  The foot orientations come from the
path (fixed), so the problem is a convex QP, about 50 variables and 80
inequality rows.  It is re-solved every tick from the predicted state
(receding horizon); a footstep is frozen once it has landed.

The human walk (gait.py) gives the centre of mass a height
profile h(t) fixed in advance (the pelvis's rise and bob from human_gait.json).
The ZMP is then z = c − h/(g + ḧ)·c̈ with that factor known per tick, so the
problem stays the same convex QP.  Its ZMP targets also roll from heel to toe:
along the stance sole in single support, from the trailing toe to the leading
heel in double support.

Each QP goes to Rekin (`rekin.Problem`, its certified convex QP path); the
planner re-solves every 10th with CasADi's qpOASES and compares.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

G = 9.81
H = 0.67               # m, centre-of-mass height (the 23-DoF G1 standing knees bent: 0.669)
TICK = 0.1             # s
N = 16                 # ticks in the horizon (1.6 s)
W_POS, W_VEL, W_ZMP, W_JERK, W_STEP = 10.0, 1.0, 10.0, 1e-4, 1000.0

# the sole, around the foot site (x forward, y left), shrunk for a margin
SOLE_X, SOLE_Y = (-0.04, 0.05), (-0.02, 0.02)
# a new foot relative to the previous one, in that foot's frame, for a LEFT
# foot after a right one (a right foot after a left: y mirrored)
REACH_X, REACH_Y = (-0.18, 0.28), (0.16, 0.32)


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


@dataclass
class Footstep:
    side: int                    # +1 left, −1 right
    xy: np.ndarray               # nominal position (foot site)
    yaw: float
    land: int                    # tick at which it starts bearing weight
    fixed: bool = False


@dataclass
class Gait:
    """Footsteps in landing order and, per tick, the support: (i,) the stance
    foot i, or (i, j, s) double support from foot i to foot j, s ∈ [0, 1].
    eta: per tick, the ZMP's factor h/(g + ḧ) (None: H/G throughout); zmp_off:
    per tick, the ZMP target's offsets along the two support feet's x axes
    (from, to; None: the soles' centres); reach_y: a new foot's sideways reach."""
    steps: list[Footstep]
    support: list[tuple]
    lift: dict = field(default_factory=dict)   # step index -> tick its swing starts
    eta: np.ndarray | None = None
    zmp_off: np.ndarray | None = None
    reach_y: tuple = None
    fwd: dict = field(default_factory=dict)    # step index -> the swing foot walks forward


@dataclass
class QPStats:
    n: int = 0
    secs: float = 0.0
    worst_status: str = ""
    check_n: int = 0
    check_max_rel: float = 0.0
    check_secs: float = 0.0
    statuses: dict = field(default_factory=dict)


def lipm_matrices(n):
    """c, ċ, c̈ at ticks 1..n as  S s0 + U j  (per axis)."""
    A = np.array([[1, TICK, TICK ** 2 / 2], [0, 1, TICK], [0, 0, 1]])
    B = np.array([TICK ** 3 / 6, TICK ** 2 / 2, TICK])
    S = np.zeros((n, 3, 3))
    U = np.zeros((n, 3, n))
    Ak = np.eye(3)
    for k in range(n):
        Ak = A @ Ak
        S[k] = Ak
        for i in range(k + 1):
            U[k, :, i] = np.linalg.matrix_power(A, k - i) @ B
    return S, U


_MATS = {}


def _mats(n):
    if n not in _MATS:
        _MATS[n] = lipm_matrices(n)
    return _MATS[n]


@dataclass
class QP:
    """min Σ w (a·u − b)²  s.t.  lo ≤ C u ≤ hi,  lb ≤ u ≤ ub."""
    nu: int
    ls: list            # (w, a (nu,), b)
    C: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    def cost(self, u):
        return float(sum(w * (a @ u - b) ** 2 for w, a, b in self.ls))

    def viol(self, u):
        r = self.C @ u
        return float(max(0.0, (self.lo - r).max(initial=0), (r - self.hi).max(initial=0),
                         (self.lb - u).max(initial=0), (u - self.ub).max(initial=0)))


def build_qp(gait: Gait, k0: int, s0x, s0y, ref_p, ref_v, n: int) -> tuple[QP, list[int]]:
    """The QP at tick k0 over ticks k0+1 .. k0+n.  ref_p, ref_v: (T, 2) per tick.
    Returns the QP and the indices of the free footsteps (variables after the jerks)."""
    S, U = _mats(n)
    free = [i for i, f in enumerate(gait.steps) if not f.fixed and f.land <= k0 + n]
    nj = 2 * n
    nu = nj + 2 * len(free)
    col = {i: nj + 2 * q for q, i in enumerate(free)}

    def foot(i):
        """(a_x, a_y, b): the foot's x and y as linear forms in u, plus constants."""
        ax, ay = np.zeros(nu), np.zeros(nu)
        f = gait.steps[i]
        if i in col:
            ax[col[i]], ay[col[i] + 1] = 1.0, 1.0
            return ax, ay, np.zeros(2)
        return ax, ay, f.xy.copy()

    ls, rows, lo, hi = [], [], [], []
    T = len(gait.support)
    for k in range(n):
        t = min(k0 + 1 + k, T - 1)
        # state at tick k0+1+k, per axis, as linear forms in the jerks
        forms = {}
        for ax_i, s0 in ((0, s0x), (1, s0y)):
            for d in range(3):
                a = np.zeros(nu)
                a[ax_i * n:(ax_i + 1) * n] = U[k, d]
                forms[(ax_i, d)] = (a, float(S[k, d] @ s0))
        # position and velocity tracking
        for ax_i in (0, 1):
            a, c = forms[(ax_i, 0)]
            ls.append((W_POS, a, ref_p[t, ax_i] - c))
            a, c = forms[(ax_i, 1)]
            ls.append((W_VEL, a, ref_v[t, ax_i] - c))
        # ZMP forms
        eta = H / G if gait.eta is None else float(gait.eta[t])
        z = []
        for ax_i in (0, 1):
            ap, cp = forms[(ax_i, 0)]
            aa, ca = forms[(ax_i, 2)]
            z.append((ap - eta * aa, cp - eta * ca))
        sup = gait.support[t]
        if len(sup) == 1:
            fx, fy, fb = foot(sup[0])
            yaw = gait.steps[sup[0]].yaw
            mx, my, mb = fx, fy, fb
            half = (SOLE_X, SOLE_Y)
        else:
            i, j, s = sup
            fxi, fyi, fbi = foot(i)
            fxj, fyj, fbj = foot(j)
            mx, my, mb = (1 - s) * fxi + s * fxj, (1 - s) * fyi + s * fyj, (1 - s) * fbi + s * fbj
            yi, yj = gait.steps[i].yaw, gait.steps[j].yaw
            yaw = math.atan2(math.sin(yi) + math.sin(yj), math.cos(yi) + math.cos(yj))
            # a box around the two soles' centres, from their nominal offset
            off = rot(-yaw) @ (gait.steps[j].xy - gait.steps[i].xy)
            ex, ey = abs(off[0]) / 2, abs(off[1]) / 2
            half = ((SOLE_X[0] - ex, SOLE_X[1] + ex), (SOLE_Y[0] - ey, SOLE_Y[1] + ey))
            # the box is centred between the feet; s only moves the ZMP target
            mx, my, mb = 0.5 * (fxi + fxj), 0.5 * (fyi + fyj), 0.5 * (fbi + fbj)
            sx, sy, sb = (1 - s) * fxi + s * fxj, (1 - s) * fyi + s * fyj, (1 - s) * fbi + s * fbj
        # ZMP centring target: the stance sole, or the blend in double support
        tx, ty, tb = (mx, my, mb) if len(sup) == 1 else (sx, sy, sb)
        if gait.zmp_off is not None:       # heel to toe: offsets along the feet's x axes
            o0, o1 = gait.zmp_off[t]
            if len(sup) == 1:
                yv = gait.steps[sup[0]].yaw
                tb = tb + o0 * np.array([math.cos(yv), math.sin(yv)])
            else:
                yi, yj = gait.steps[sup[0]].yaw, gait.steps[sup[1]].yaw
                tb = tb + ((1 - sup[2]) * o0 * np.array([math.cos(yi), math.sin(yi)])
                           + sup[2] * o1 * np.array([math.cos(yj), math.sin(yj)]))
        ls.append((W_ZMP, z[0][0] - tx, tb[0] - z[0][1]))
        ls.append((W_ZMP, z[1][0] - ty, tb[1] - z[1][1]))
        # support constraint: R(yaw)ᵀ (z − m) ∈ half
        c, s_ = math.cos(yaw), math.sin(yaw)
        for (rx, ry), (l, h) in (((c, s_), half[0]), ((-s_, c), half[1])):
            a = rx * (z[0][0] - mx) + ry * (z[1][0] - my)
            b = rx * (z[0][1] - mb[0]) + ry * (z[1][1] - mb[1])
            rows.append(a)
            lo.append(l - b)
            hi.append(h - b)
    # jerk regularisation
    for q in range(nj):
        a = np.zeros(nu)
        a[q] = 1.0
        ls.append((W_JERK, a, 0.0))
    # footsteps: near their nominal spots, within reach of the foot before
    lb, ub = np.full(nu, -200.0), np.full(nu, 200.0)
    for i in free:
        f = gait.steps[i]
        for d in (0, 1):
            a = np.zeros(nu)
            a[col[i] + d] = 1.0
            ls.append((W_STEP, a, f.xy[d]))
            lb[col[i] + d], ub[col[i] + d] = f.xy[d] - 0.3, f.xy[d] + 0.3
        px, py, pb = foot(i - 1)
        fx, fy, fb = foot(i)
        yaw = gait.steps[i - 1].yaw
        c, s_ = math.cos(yaw), math.sin(yaw)
        reach = gait.reach_y or REACH_Y
        ry = reach if f.side > 0 else (-reach[1], -reach[0])
        for (rx, rr), (l, h) in (((c, s_), REACH_X), ((-s_, c), ry)):
            a = rx * (fx - px) + rr * (fy - py)
            b = rx * (fb[0] - pb[0]) + rr * (fb[1] - pb[1])
            rows.append(a)
            lo.append(l - b)
            hi.append(h - b)
    qp = QP(nu, ls, np.array(rows), np.array(lo), np.array(hi), lb, ub)
    return qp, free


LEGS_OFF = ["boxes", "bnp", "cbnp", "ipm", "sa", "admm", "tighten", "points"]


def solve_rekin(qp: QP, budget=5.0):
    """The QP through Rekin's scalar model API: the variables with their bounds,
    each row as two linear constraints, the objective as Σ w (a·u − b)².
    Rekin recognises the convex QP and solves it on its certified conic path.
    (Its matrix API, addLinearRows + addQuadObjective, skipped that path on
    about 0.4% of these QPs and returned best-effort answers 2-14% above the
    optimum.)"""
    # Rekin 0.1.26.10.0 with its GPU plugin loaded segfaults after ~30-40
    # Problem solves in one process (a native worker thread).  These QPs never
    # use the GPU: keep the plugin unloaded.
    import os
    os.environ.setdefault("REKIN_GPU", "0")
    import rekin
    p = rekin.Problem(delta=1e-7)
    u = [p.addVar(f"u{i}", [float(qp.lb[i]), float(qp.ub[i])]) for i in range(qp.nu)]

    def lin(a):
        return sum(float(a[i]) * u[i] for i in np.nonzero(a)[0])

    for r, lo, hi in zip(qp.C, qp.lo, qp.hi):
        e = lin(r)
        p.addConstraint(e >= float(lo))
        p.addConstraint(e <= float(hi))
    obj = 0
    for w, a, b in qp.ls:
        e = lin(a) - float(b)
        obj = obj + float(w) * e * e
    p.addObjective(obj)
    # the conic gate certifies these before any search leg is needed; the
    # legs' threads cost 5.5x the CPU (measured), so they stay off
    r = p.solve(budget=budget, live=False, gpu=False, nthreads=1, stall_stop=0.5, threads_off=LEGS_OFF)
    sol = np.array([r["sol"][f"u{i}"] for i in range(qp.nu)])
    st = r.get("status_name", str(r.get("status")))
    if st != "delta-global" and os.environ.get("MPC_DUMP"):     # keep the QPs Rekin did not certify
        import pickle
        import uuid
        os.makedirs(os.environ["MPC_DUMP"], exist_ok=True)
        pickle.dump((qp, r), open(os.path.join(os.environ["MPC_DUMP"], f"{st}_{uuid.uuid4().hex[:8]}.pkl"), "wb"))
    return sol, st


_CASADI = {}


def solve_casadi(qp: QP):
    import casadi as ca
    Hm = np.zeros((qp.nu, qp.nu))
    g = np.zeros(qp.nu)
    for w, a, b in qp.ls:
        Hm += 2 * w * np.outer(a, a)
        g += -2 * w * b * a
    key = (qp.nu, qp.C.shape[0])
    if key not in _CASADI:                    # qpOASES prints a banner per instance
        import os
        import sys
        sys.stdout.flush()
        fd, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.close(null)
        try:
            _CASADI[key] = ca.conic("s", "qpoases", {"h": ca.Sparsity.dense(qp.nu, qp.nu),
                                                     "a": ca.Sparsity.dense(*qp.C.shape)},
                                    {"printLevel": "none", "error_on_fail": False})
            _CASADI[key](h=Hm, g=g, a=qp.C, lba=qp.lo, uba=qp.hi, lbx=qp.lb, ubx=qp.ub)
        finally:
            import ctypes
            ctypes.CDLL(None).fflush(None)    # the banner sits in C stdio's buffer
            os.dup2(fd, 1)
            os.close(fd)
    r = _CASADI[key](h=Hm, g=g, a=qp.C, lba=qp.lo, uba=qp.hi, lbx=qp.lb, ubx=qp.ub)
    return np.array(r["x"]).ravel()


def run(gait: Gait, ref_p, ref_v, c0, stats: QPStats | None = None, check_every: int = 0):
    """Receding-horizon MPC over the gait's ticks.  ref_p, ref_v: (T, 2); c0: the
    initial CoM position (at rest).  Returns the CoM state per tick (T, 2, 3)
    and the footsteps (positions updated in place)."""
    T = len(gait.support)
    X = np.zeros((T, 2, 3))
    X[0, :, 0] = c0
    stats = stats if stats is not None else QPStats()
    for k0 in range(T - 1):
        for f in gait.steps:
            if f.land <= k0:
                f.fixed = True
        n = N
        qp, free = build_qp(gait, k0, X[k0, 0], X[k0, 1], ref_p, ref_v, n)
        t0 = time.time()
        u, st = solve_rekin(qp)
        stats.secs += time.time() - t0
        stats.n += 1
        stats.statuses[st] = stats.statuses.get(st, 0) + 1
        if check_every and k0 % check_every == 0:
            t0 = time.time()
            uc = solve_casadi(qp)
            stats.check_secs += time.time() - t0
            stats.check_n += 1
            cr, cc = qp.cost(u), qp.cost(uc)
            stats.check_max_rel = max(stats.check_max_rel, (cr - cc) / max(abs(cc), 1e-9))
        for q, i in enumerate(free):
            if gait.steps[i].land > k0:
                gait.steps[i].xy = u[2 * n + 2 * q: 2 * n + 2 * q + 2].copy()
        A = np.array([[1, TICK, TICK ** 2 / 2], [0, 1, TICK], [0, 0, 1]])
        B = np.array([TICK ** 3 / 6, TICK ** 2 / 2, TICK])
        for ax_i in (0, 1):
            X[k0 + 1, ax_i] = A @ X[k0, ax_i] + B * u[ax_i * n]
    return X, stats


def com_at(X, t):
    """The CoM (position, velocity) at time t (s) from the per-tick states: exact
    within a tick (constant jerk, recovered from the next tick's acceleration)."""
    T = len(X)
    k = min(int(t / TICK), T - 2)
    tau = t - k * TICK
    p, v, a = X[k, :, 0], X[k, :, 1], X[k, :, 2]
    j = (X[k + 1, :, 2] - a) / TICK
    return (p + v * tau + a * tau ** 2 / 2 + j * tau ** 3 / 6,
            v + a * tau + j * tau ** 2 / 2)
