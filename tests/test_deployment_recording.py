"""Exercise the real C++ recording codec/writer and SDK worker shutdown offline."""
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest

from scripts import replay_deployment as replay

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'deploy/robots/g1_23dof/build'


@pytest.fixture(scope='module')
def fixture_binary(tmp_path_factory):
    flags_file = BUILD / 'CMakeFiles/g1_ctrl.dir/flags.make'
    link_file = BUILD / 'CMakeFiles/g1_ctrl.dir/link.txt'
    if not flags_file.exists() or not link_file.exists():
        pytest.skip('Build g1_ctrl first to configure its SDK include/library paths')
    flags = {}
    for line in flags_file.read_text().splitlines():
        if ' = ' in line:
            name, value = line.split(' = ', 1)
            flags[name] = shlex.split(value)
    # Reuse configured architecture/dependency paths, not hardcoded local SDKs.
    link = shlex.split(link_file.read_text())
    libraries = [x for x in link if x.startswith(('-L', '-Wl,-rpath,'))]
    output = tmp_path_factory.mktemp('recorder_build') / 'recording_fixture'
    subprocess.run(['c++', *flags['CXX_FLAGS'], *flags['CXX_INCLUDES'],
                    str(ROOT / 'tests/fixtures/deployment_recording.cpp'), '-o', str(output),
                    *libraries, '-lunitree_sdk2', '-lyaml-cpp', '-lfmt', '-lpthread'],
                   check=True, cwd=BUILD, capture_output=True, text=True)
    return output


@pytest.fixture
def session(fixture_binary, tmp_path):
    directory = tmp_path / 'session'
    subprocess.run([str(fixture_binary), str(directory)], check=True, timeout=10)
    return directory


def test_full_message_roundtrip_and_export(session, tmp_path):
    info = replay.inspect(session)
    assert info['complete'] and info['has_motion']
    assert info['counts'] == dict(lowstate=200, lowcmd=200, transitions=3)
    assert [(x['from'], x['to']) for x in info['transitions']] == [(-1, 3), (3, 5), (5, -1)]
    out = tmp_path / 'run.npz'
    replay.export_npz(session, out)
    data = np.load(out, allow_pickle=False)
    state = data['lowstate']['data'][0]
    assert state['tick'] == 123456 and state['crc'] == 0xfeedabcd
    np.testing.assert_array_equal(state['version'], [7, 9])
    assert state['mode_pr'] == 1 and state['mode_machine'] == 4
    np.testing.assert_array_equal(state['imu']['quaternion'], [.5, -.5, .5, -.5])
    np.testing.assert_array_equal(state['imu']['gyroscope'], [1, 2, 3])
    np.testing.assert_array_equal(state['imu']['accelerometer'], [4, 5, 6])
    np.testing.assert_array_equal(state['imu']['rpy'], [7, 8, 9])
    assert state['imu']['temperature'] == -30
    np.testing.assert_array_equal(state['wireless_remote'], np.arange(40))
    np.testing.assert_array_equal(state['reserve'], [10, 11, 12, 13])
    for i, m in enumerate(state['motors']):
        assert (m['mode'], m['q'], m['dq'], m['ddq'], m['tau_est']) == (i, i+.25, -i-.5, i+.75, -i-.25)
        np.testing.assert_array_equal(m['temperature'], [i, -i])
        assert m['vol'] == 48+i and m['motorstate'] == 100+i
        np.testing.assert_array_equal(m['sensor'], [i, i+1])
        np.testing.assert_array_equal(m['reserve'], [i+2, i+3, i+4, i+5])
    cmd = data['lowcmd']['data'][0]
    assert (cmd['mode_pr'], cmd['mode_machine'], cmd['crc']) == (1, 4, 0xabcdef12)
    np.testing.assert_array_equal(cmd['reserve'], [20, 21, 22, 23])
    for i, m in enumerate(cmd['motors']):
        assert tuple(m) == (i, i+.125, -i-.125, i+.5, 40+i, 3+i, 200+i)
    np.testing.assert_array_equal(data['lowstate']['data']['tick'], np.arange(123456, 123656))
    for group in data.values():
        assert (np.diff(group['time_ns'].astype(np.int64)) >= 0).all()
    with pytest.raises(FileExistsError):
        replay.export_npz(session, out)


def test_overflow_is_explicit(fixture_binary, tmp_path):
    directory = tmp_path / 'overflow'
    subprocess.run([str(fixture_binary), str(directory), 'overflow'], check=True, timeout=10)
    info = replay.inspect(directory)
    assert info['clean_shutdown'] and not info['complete'] and info['dropped'] == 2


def test_fsm_stops_worker_before_exiting_state(fixture_binary, tmp_path):
    directory = tmp_path / 'fsm'
    subprocess.run([str(fixture_binary), str(directory), 'fsm'], check=True, timeout=10)
    transitions = replay.inspect(directory)['transitions']
    assert [(x['from'], x['to']) for x in transitions] == [(-1, 101), (101, 102), (102, -1)]


def test_disk_write_failure_leaves_incomplete_session(fixture_binary, tmp_path):
    directory = tmp_path / 'failed'
    subprocess.run([str(fixture_binary), str(directory), 'write_failure'], check=True, timeout=10)
    with pytest.raises(ValueError, match='Missing shutdown footer'):
        replay.inspect(directory)
    assert not replay.inspect(directory, allow_incomplete=True)['complete']


def test_existing_session_is_not_overwritten(fixture_binary, session):
    before = (session / 'records.bin').read_bytes()
    result = subprocess.run([str(fixture_binary), str(session)], capture_output=True, timeout=10)
    assert result.returncode != 0
    assert (session / 'records.bin').read_bytes() == before


@pytest.mark.parametrize('damage', ['truncated', 'missing_footer', 'payload', 'header', 'version'])
def test_detects_incomplete_and_corrupt_records(session, damage):
    path = session / 'records.bin'
    content = bytearray(path.read_bytes())
    if damage == 'truncated':
        content = content[:-10]
    elif damage == 'missing_footer':
        content = content[:-56]
    elif damage == 'payload':
        content[48] ^= 1
    elif damage == 'header':
        content[32] ^= 1
    else:
        content[8] = 2
    path.write_bytes(content)
    with pytest.raises(ValueError):
        replay.inspect(session)
    if damage in ('truncated', 'missing_footer'):
        recovered = replay.inspect(session, allow_incomplete=True)
        assert not recovered['complete'] and recovered['counts']['lowstate'] == 200
    else:
        with pytest.raises(ValueError):
            replay.inspect(session, allow_incomplete=True)


def test_replay_preserves_samples_and_timestamps(session, monkeypatch, capsys):
    import json
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    replay.replay(session, speed=2)
    result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    expected = [row for row in replay.records(session) if row[0] == 1]
    assert len(result) == len(expected) == 200
    for actual, (_, sequence, stamp, sample) in zip(result, expected):
        assert actual['time_ns'] == stamp and actual['sequence'] == sequence
        assert actual['lowstate']['tick'] == int(sample['tick'])
        assert actual['lowstate']['motors'][34]['q'] == float(sample['motors'][34]['q'])


def test_cli_help_and_record_path_rejected_before_dds(tmp_path):
    binary = BUILD / 'g1_ctrl'
    if not binary.exists():
        pytest.skip('Build g1_ctrl first')
    help_result = subprocess.run([str(binary), '--help'], capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0 and '--record' in help_result.stdout
    # Existing directory must fail before network initialization, even with an
    # impossible interface. Use the default non-box config to avoid inference.
    result = subprocess.run([str(binary), '--record', str(tmp_path), '--network', 'no-such-network'],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and 'Recording directory already exists' in result.stderr
    assert not (tmp_path / 'records.bin').exists()
