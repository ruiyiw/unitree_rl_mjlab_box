# Recording G1 23-DoF deployments

Add `--record <new-directory>` to the normal `g1_ctrl` command. It works with
the three new box policies, the original box policies/sequence, human-push,
balancing and velocity profiles. The selected YAML and gamepad controls stay
the same. C++ performs the recording; Python is only needed for offline reading.

## Build and record

From the repository root, use the usual build:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
mkdir -p ../recordings
./g1_ctrl --config ../config/box_three.yaml --network enp5s0 \
  --record ../recordings/box_three_001
```

For the original full sequence, launch instead:

```bash
./g1_ctrl --config ../config/box_sequence.yaml --network enp5s0 \
  --record ../recordings/box_sequence_001
```

Use your robot's network interface and a **new** session directory each time.
The parent directory must exist. An existing session is never overwritten, and
an invalid recording path fails before DDS startup. Run one controller at a time.
Without `--record`, no telemetry files or recording subscribers are created.
The existing `--log` flag still means the ordinary text diagnostic log.

Follow the selected policy's normal startup and shutdown procedure. With the
robot supported and in Passive, **Ctrl+C** stops the controller workers and
flushes the recording. SIGTERM also flushes. Stopping the process does not itself
command a safe posture. Power loss, crashes or SIGKILL can leave only a partial
recording; the reader detects a missing shutdown footer.

Recordings under `recordings/` are Git-ignored. Copy them separately when sharing
an experiment. Raw storage is roughly **1.5 MB/s (90 MB/min)** if both DDS topics
deliver 500 samples/s; actual size depends on topic rates. There is no automatic
rotation or deletion, so choose a disk with sufficient space.

## What is recorded

Each session contains:

- `records.bin`: versioned binary records with checksums and host monotonic
  nanosecond timestamps, without interpolation or sample-rate reduction.
- `session.yaml`: effective controller YAML, launch arguments, time anchors,
  the 23 active motor IDs, resolved policy directories and FNV-1a fingerprints
  of the executable and ONNX/reference/config assets under the configured policy
  directories. Keep the matching
  repository for policy re-execution; weights are not copied into the recording.

The binary file preserves all fields delivered by `rt/lowstate` and `rt/lowcmd`:

| Stream | Fields |
|---|---|
| Measured state | All 35 DDS motor slots: position, velocity, acceleration, estimated torque, temperatures, voltage, status, sensor and reserved fields; IMU quaternion (wxyz), gyro, acceleration, RPY and temperature; 40 raw wireless-remote bytes; firmware tick, modes, version, reserves and CRC |
| Published commands | All 35 DDS motor slots: mode, position/velocity targets, feedforward torque, kp/kd and reserves; message modes, reserves and CRC |
| FSM transitions | Previous and next state IDs; `-1` denotes controller startup/shutdown; names are in the saved YAML |

Command targets and measured motion are separate streams. Their timestamps mark
host observation, not the precise hardware sampling or actuation time. The file
sequence orders queue submissions; concurrent callbacks can have slightly
different timestamp ordering. NPZ exports sort each stream by timestamp.

The original box sequence's internal stage changes are not separate FSM
transitions. Its resulting measurements and commands are still captured.

The writer uses a bounded memory queue and a background disk thread. If it falls
behind or the disk fails, the controller reports the recording error and keeps
controlling the robot. Dropped samples are counted, and a failed write leaves an
incomplete recording. A clean footer with zero drops confirms that the recorder
saved its delivered samples; it does **not** prove DDS delivered every source
sample or that the robot received every command. Preserve raw firmware ticks to
investigate telemetry gaps.

## Inspect, export and replay offline

From the repository root, use Python with NumPy installed:

```bash
python scripts/replay_deployment.py deploy/robots/g1_23dof/recordings/box_three_001

python scripts/replay_deployment.py deploy/robots/g1_23dof/recordings/box_three_001 \
  --export /tmp/box_three_001.npz

python scripts/replay_deployment.py deploy/robots/g1_23dof/recordings/box_three_001 \
  --replay --speed 1
```

Inspection checks file integrity and prints sample counts, duration, transitions
and completeness. `--export` writes lossless structured NumPy arrays named
`lowstate`, `lowcmd` and `transitions`, each with `time_ns`, `sequence` and `data`
fields. Export loads the recording into memory; allow memory for the full file
and temporary arrays. Export refuses to overwrite a file.

`--replay` validates the file first, then emits every measured state as a JSON
line at the recorded relative timing for an offline consumer. It is a data
stream, not a graphical viewer. `--speed 0.5` uses half speed. OS scheduling,
decoding and a slow consumer can delay delivery; the original timestamps remain
authoritative. These commands never open DDS or send motor commands.

Example exported data access:

```python
import numpy as np

run = np.load('/tmp/box_three_001.npz', allow_pickle=False)
measured = run['lowstate']
time_s = measured['time_ns'] / 1e9
joint_q = measured['data']['motors']['q']       # shape (samples, 35), DDS order
pelvis_wxyz = measured['data']['imu']['quaternion']
command_q = run['lowcmd']['data']['motors']['q']
```

Use `--allow-incomplete` explicitly to recover intact records from an interrupted
file or one with reported drops. Corrupt checksums and unknown formats are always
rejected. Keep `session.yaml` beside `records.bin` for the profile and motor map.

## Limits of exact replay

This preserves the **observed joint states, pelvis orientation, commands and
timing** for later offline replay. LowState does not provide global base position,
base linear velocity, box pose or scene/contact state. Those require synchronized
external tracking to reconstruct the whole scene. The log also does not snapshot
internal policy histories or the simulator's hidden state. Re-running a neural
policy or playing targets on the physical robot is therefore not guaranteed to
reproduce the same motion. No automatic hardware command replay is provided.

## Binary layout (version 1)

All numbers are little-endian, with IEEE-754 float32 values and no padding.
The 16-byte file header is `G1REC001`, uint32 version `1`, uint32 reserved `0`.
Each record starts with `<IIQQQ`: payload bytes, type, submission sequence,
monotonic offset nanoseconds, FNV-1a-64 checksum. The checksum covers the preceding
24 header bytes followed by the payload. Types 1/2 are full LowState/LowCmd fields
in SDK declaration order, type 3 is two int32 state IDs, and type 4 is the shutdown
footer (uint64 attempted, written and dropped counts). The footer sequence equals
attempted. The precise payload dtypes are in `scripts/replay_deployment.py`.
