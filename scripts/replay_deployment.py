#!/usr/bin/env python3
"""Inspect, export, or replay G1 recordings offline. Never opens DDS."""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

MOTOR_STATE = np.dtype([
    ('mode', 'u1'), ('q', '<f4'), ('dq', '<f4'), ('ddq', '<f4'),
    ('tau_est', '<f4'), ('temperature', '<i2', 2), ('vol', '<f4'),
    ('sensor', '<u4', 2), ('motorstate', '<u4'), ('reserve', '<u4', 4),
])
MOTOR_COMMAND = np.dtype([
    ('mode', 'u1'), ('q', '<f4'), ('dq', '<f4'), ('tau', '<f4'),
    ('kp', '<f4'), ('kd', '<f4'), ('reserve', '<u4'),
])
IMU = np.dtype([
    ('quaternion', '<f4', 4), ('gyroscope', '<f4', 3),
    ('accelerometer', '<f4', 3), ('rpy', '<f4', 3), ('temperature', '<i2'),
])
PAYLOADS = {
    1: np.dtype([
        ('version', '<u4', 2), ('mode_pr', 'u1'), ('mode_machine', 'u1'),
        ('tick', '<u4'), ('imu', IMU), ('motors', MOTOR_STATE, 35),
        ('wireless_remote', 'u1', 40), ('reserve', '<u4', 4), ('crc', '<u4'),
    ]),
    2: np.dtype([
        ('mode_pr', 'u1'), ('mode_machine', 'u1'), ('motors', MOTOR_COMMAND, 35),
        ('reserve', '<u4', 4), ('crc', '<u4'),
    ]),
    3: np.dtype([('from_state', '<i4'), ('to_state', '<i4')]),
    4: np.dtype([('attempted', '<u8'), ('written', '<u8'), ('dropped', '<u8')]),
}
NAMES = {1: 'lowstate', 2: 'lowcmd', 3: 'transitions', 4: 'footer'}
HEADER = struct.Struct('<IIQQQ')


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def records(directory, allow_incomplete=False):
    """Yield (type, sequence, monotonic offset ns, decoded scalar).

    Accept only an intact prefix on truncation when explicitly requested.
    Corrupt/unknown records are always errors. No interpolation or resampling.
    """
    with (Path(directory) / 'records.bin').open('rb') as stream:
        if stream.read(16) != b'G1REC001' + struct.pack('<II', 1, 0):
            raise ValueError('Unknown recording format/version')
        previous = -1
        count = 0
        while True:
            header = stream.read(HEADER.size)
            if not header:
                if allow_incomplete:
                    return
                raise ValueError('Missing shutdown footer: recording was interrupted')
            if len(header) != HEADER.size:
                if allow_incomplete:
                    return
                raise ValueError('Truncated record header')
            size, kind, sequence, time_ns, checksum = HEADER.unpack(header)
            if kind not in PAYLOADS or size != PAYLOADS[kind].itemsize:
                raise ValueError(f'Invalid record type/size: {kind}/{size}')
            payload = stream.read(size)
            if len(payload) != size:
                if allow_incomplete:
                    return
                raise ValueError('Truncated record payload')
            if fnv1a(header[:24] + payload) != checksum:
                raise ValueError(f'Checksum mismatch at record {sequence}')
            if sequence <= previous:
                raise ValueError('Non-increasing record sequence')
            previous = sequence
            sample = np.frombuffer(payload, dtype=PAYLOADS[kind], count=1)[0]
            if kind == 4:
                attempted, written, dropped = (int(sample[key]) for key in PAYLOADS[4].names)
                if written != count or attempted != written + dropped or sequence != attempted:
                    raise ValueError('Footer counts do not match the recording')
                if stream.read(1):
                    raise ValueError('Unexpected data after shutdown footer')
                yield kind, sequence, time_ns, sample
                return
            count += 1
            yield kind, sequence, time_ns, sample


def inspect(directory, allow_incomplete=False):
    summary = {'counts': dict.fromkeys(('lowstate', 'lowcmd', 'transitions'), 0),
               'duration_s': 0., 'clean_shutdown': False, 'dropped': None, 'transitions': []}
    for kind, sequence, stamp, sample in records(directory, allow_incomplete):
        summary['duration_s'] = max(summary['duration_s'], stamp / 1e9)
        if kind == 4:
            summary['clean_shutdown'] = True
            summary['dropped'] = int(sample['dropped'])
        else:
            summary['counts'][NAMES[kind]] += 1
        if kind == 3:
            summary['transitions'].append({'time_s': stamp / 1e9,
                                          'from': int(sample['from_state']), 'to': int(sample['to_state'])})
    summary['complete'] = summary['clean_shutdown'] and summary['dropped'] == 0
    # Completeness here concerns recorder delivery, not DDS or the robot's receipt
    # of commands. A session with no measured states cannot replay robot motion.
    summary['has_motion'] = summary['counts']['lowstate'] > 0
    return summary


def export_npz(directory, output, allow_incomplete=False):
    groups = {kind: [] for kind in (1, 2, 3)}
    for kind, sequence, stamp, sample in records(directory, allow_incomplete):
        if kind in groups:
            groups[kind].append(struct.pack('<QQ', stamp, sequence) + sample.tobytes())
    arrays = {}
    for kind, chunks in groups.items():
        dtype = np.dtype([('time_ns', '<u8'), ('sequence', '<u8'), ('data', PAYLOADS[kind])])
        data = np.frombuffer(b''.join(chunks), dtype=dtype)
        arrays[NAMES[kind]] = data[np.argsort(data['time_ns'], kind='stable')]
    # Exclusive creation prevents accidentally destroying an earlier export.
    with Path(output).open('xb') as stream:
        np.savez(stream, **arrays)


def json_value(value):
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value]
    if isinstance(value, np.void):
        return {key: json_value(value[key]) for key in value.dtype.names}
    if isinstance(value, np.generic):
        return value.item()
    return value


def replay(directory, speed=1., allow_incomplete=False):
    """Stream every observed state as JSON at its recorded relative timing.

    This is an offline data stream for a viewer/consumer, never motor commands.
    Thread scheduling and stdout backpressure can delay delivery; timestamps are
    authoritative. Samples are emitted in recorder enqueue order.
    """
    start = time.monotonic()
    for kind, sequence, stamp, sample in records(directory, allow_incomplete):
        if kind != 1:
            continue
        remaining = start + stamp / 1e9 / speed - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        print(json.dumps({'time_ns': stamp, 'sequence': sequence, 'lowstate': json_value(sample)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--export', type=Path, metavar='FILE.npz')
    action.add_argument('--replay', action='store_true', help='stream measured states as JSON at recorded timing')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--allow-incomplete', action='store_true', help='use an intact prefix or a recording with drops')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    try:
        summary = inspect(args.directory, args.allow_incomplete)
        print(json.dumps(summary, indent=2), file=sys.stderr if args.replay else sys.stdout)
        if not summary['complete'] and not args.allow_incomplete:
            raise ValueError('Recording has dropped samples; use --allow-incomplete to recover it explicitly')
        if args.export:
            export_npz(args.directory, args.export, args.allow_incomplete)
        elif args.replay:
            if not summary['has_motion']:
                raise ValueError('Recording contains no measured robot states')
            replay(args.directory, args.speed, args.allow_incomplete)
    except (ValueError, OSError) as error:
        print(f'Recording error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
