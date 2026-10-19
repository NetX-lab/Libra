"""Place rollout restarts inside the current allocation and stop only owned work."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid


def is_local_host(host, env=None):
    env = os.environ if env is None else env
    host = str(host or '')
    try:
        address = ipaddress.ip_address(host)
        if address.is_loopback or address.is_unspecified:
            return True
    except ValueError:
        pass
    names = {'', 'localhost', socket.gethostname(), socket.gethostname().split('.', 1)[0],
             str(env.get('SLURMD_NODENAME', ''))}
    return host in names or host.split('.', 1)[0] in {name.split('.', 1)[0] for name in names if name}


def allocation_nodes(env):
    expression = env.get('SLURM_JOB_NODELIST') or env.get('SLURM_NODELIST')
    if not env.get('SLURM_JOB_ID') or not expression:
        raise RuntimeError('Remote rollout launch requires the current Slurm allocation and node list')
    result = subprocess.run(['scontrol', 'show', 'hostnames', expression], text=True,
                            capture_output=True, check=True, env=env)
    nodes = result.stdout.split()
    if not nodes:
        raise RuntimeError('The current Slurm allocation contains no nodes')
    return nodes


def target_node(host, env, nodes=None):
    nodes = allocation_nodes(env) if nodes is None else nodes
    exact = [node for node in nodes if node == host or node.split('.', 1)[0] == host.split('.', 1)[0]]
    if not exact:
        try:
            address = str(ipaddress.ip_address(host))
            exact = [node for node in nodes if address in {
                result[4][0] for result in socket.getaddrinfo(node, None)}]
        except (ValueError, OSError):
            pass
    if len(exact) != 1:
        raise RuntimeError(f'Rollout host {host!r} is not one unambiguous node in the current Slurm allocation')
    return exact[0]


def slurm_command(command, host, gpus, env, *, step_name=None):
    """Use allocated node GPUs; never request an arbitrary replacement subset."""
    node = target_node(host, env)
    ids = ','.join(str(gpu) for gpu in gpus)
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Expected distinct assigned node GPU IDs')
    # With --overlap the step inherits the job's existing GPU allocation on
    # this node. Asking for --gpus=N could select a different physical subset.
    argv = ['srun', '--overlap', '--jobid', env['SLURM_JOB_ID'], '--nodes=1', '--ntasks=1',
            '--nodelist', node, '--cpus-per-task=4', '--cpu-bind=none']
    if step_name:
        argv += ['--job-name', step_name]
    argv += ['env', 'CUDA_VISIBLE_DEVICES=' + ids, 'PLANNED_CUDA_VISIBLE_DEVICES=' + ids,
             'bash', '-c', command]
    return shlex.join(argv)


def placement_command(command, host, gpus, env):
    return command if is_local_host(host, env) else slurm_command(command, host, gpus, env)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _process_record(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), parent=int(fields[1]), group=int(fields[2]),
                    state=fields[0], start=fields[19])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _owned_descendants(pid):
    records = {int(path.name): _process_record(int(path.name)) for path in Path('/proc').iterdir()
               if path.name.isdigit()}
    owned, frontier = {}, [pid]
    while frontier:
        current = frontier.pop()
        record = records.get(current)
        if record is not None and current not in owned:
            owned[current] = record
            frontier.extend(key for key, child in records.items() if child and child['parent'] == current)
    return owned


def _still_owned(record):
    current = _process_record(record['pid'])
    return bool(current and current['start'] == record['start'] and current['state'] != 'Z')


def stop_owned_session(proc, timeout=30.):
    """Include owned descendant sessions; a supervisor may start a new session."""
    real = any(base.__module__ == 'subprocess' and base.__name__ == 'Popen' for base in type(proc).__mro__)
    if not real:
        from .process_lifecycle import stop_local_worker
        return stop_local_worker(proc, timeout=timeout)
    records = _owned_descendants(proc.pid)
    root = records.get(proc.pid)
    if root is not None and root['group'] != proc.pid and proc.poll() is None:
        raise RuntimeError('Rollout launcher does not own a dedicated process session')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Capture all descendant sessions before removing the supervisor that
        # otherwise supplies their parent relationship. PID birth times fence
        # unrelated processes if an identifier is reused during cleanup.
        records.update(_owned_descendants(proc.pid))
        for record in reversed(list(records.values())):
            if _still_owned(record):
                try:
                    os.kill(record['pid'], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while any(_still_owned(record) for record in records.values()) and (deadline is None or time.monotonic() < deadline):
            time.sleep(.02)
        if not any(_still_owned(record) for record in records.values()):
            proc.wait(timeout=max(timeout, .1) if timeout > 0 else None)
            return
    raise RuntimeError('Owned rollout descendant processes remain live after termination')


def launch_rollout_process(meta, env, stdout=None, control_dir=None):
    remote = not is_local_host(meta.host, env)
    token = uuid.uuid4().hex
    owner = dict(token=token, instance_id=meta.instance_id, host=meta.host, gpus=list(meta.gpus), remote=remote,
                 job_id=env.get('SLURM_JOB_ID'), step_name='libroll-' + token[:16])
    command = meta.command
    if remote:
        root = Path(control_dir or env.get('LIBRA_ROLLOUT_PROCESS_DIR') or '')
        if not str(root) or str(root) == '.':
            raise ValueError('Remote rollout ownership requires a shared project control directory')
        path = root / 'process-owners' / (token + '.json')
        owner['state_path'] = str(path)
        argv = [sys.executable, '-m', 'RL_Framework.infra.elastic.rollout_process',
                '--state', str(path), '--token', token, '--', 'bash', '-c', command]
        command = slurm_command(shlex.join(argv), meta.host, meta.gpus, env, step_name=owner['step_name'])
    proc = subprocess.Popen(command, shell=True, env=env, stdout=stdout,
                            stderr=subprocess.STDOUT if stdout is not None else None, start_new_session=True)
    proc._libra_rollout_owner = owner
    return proc


def _owned_remote_step(owner, env):
    path = Path(owner['state_path'])
    if path.exists():
        record = json.loads(path.read_text())
        if record.get('token') != owner['token'] or str(record.get('job_id')) != str(owner['job_id']):
            raise RuntimeError('Remote rollout ownership receipt does not match this launcher')
        step = record.get('step_id')
        if not str(step).isdigit():
            raise RuntimeError('Remote rollout receipt lacks an actual Slurm step ID')
        return str(owner['job_id']) + '.' + str(step), record
    # srun may still be waiting to start. Its unique step name lets cancellation
    # remain scoped even before the remote command publishes its first receipt.
    result = subprocess.run(['squeue', '--steps', '--jobs', str(owner['job_id']), '--noheader',
                             '--format=%i|%j'], check=True, capture_output=True, text=True, env=env)
    matches = [line.split('|', 1)[0].strip() for line in result.stdout.splitlines()
               if '|' in line and line.split('|', 1)[1].strip() == owner['step_name']]
    if len(matches) > 1:
        raise RuntimeError('Remote rollout owner maps to more than one Slurm step')
    return (matches[0] if matches else None), None


def stop_rollout_process(proc, meta=None, timeout=30., control_dir=None):
    owner = getattr(proc, '_libra_rollout_owner', None)
    if not owner or not owner['remote']:
        return stop_owned_session(proc, timeout)
    env = os.environ.copy()
    if str(env.get('SLURM_JOB_ID')) != str(owner['job_id']):
        raise RuntimeError('Cannot stop a rollout from another Slurm allocation')
    step, record = _owned_remote_step(owner, env)
    if record and record.get('status') == 'exited' and record.get('owned_processes_exited') is True:
        return stop_owned_session(proc, timeout)
    if step:
        subprocess.run(['scancel', '--signal=TERM', step], check=True, env=env, capture_output=True, text=True)
    else:
        # Cancel a pending srun before it can create a remote step.
        stop_owned_session(proc, timeout)
    deadline = time.monotonic() + timeout if timeout > 0 else None
    while deadline is None or time.monotonic() < deadline:
        _, record = _owned_remote_step(owner, env)
        if record and record.get('status') == 'exited' and record.get('owned_processes_exited') is True:
            stop_owned_session(proc, timeout)
            return
        query = subprocess.run(['squeue', '--steps', '--jobs', str(owner['job_id']), '--noheader',
                                '--format=%i|%j'], check=True, capture_output=True, text=True, env=env)
        if not any('|' in line and line.split('|', 1)[1].strip() == owner['step_name']
                   for line in query.stdout.splitlines()) and proc.poll() is not None:
            return
        time.sleep(.05)
    if step:
        subprocess.run(['scancel', '--signal=KILL', step], check=True, env=env, capture_output=True, text=True)
    raise RuntimeError('Remote rollout Slurm step exit was not verified; retaining ownership')



def _process_command(pid):
    root = Path('/proc') / str(pid)
    try:
        if root.stat().st_uid != os.getuid():
            raise RuntimeError('Adopted rollout PID belongs to another user')
        argv = [part.decode() for part in (root / 'cmdline').read_bytes().split(b'\0') if part]
        environment = dict(part.decode().split('=', 1) for part in (root / 'environ').read_bytes().split(b'\0')
                           if b'=' in part)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError('Cannot verify adopted rollout PID identity') from exc
    return argv, environment


def _argument_value(argv, flag):
    for index, argument in enumerate(argv):
        if argument == flag and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith(flag + '='):
            return argument.split('=', 1)[1]
    return None


def _flatten_shell_arguments(argv):
    values = list(argv)
    for argument in argv:
        if '--instance-id' in argument or '--port' in argument:
            try:
                values.extend(shlex.split(argument))
            except ValueError:
                pass
    return values


def _verify_adopted_identity(pid, contract, *, launcher=False):
    record = _process_record(pid)
    if record is None or record['state'] == 'Z':
        raise RuntimeError('Adopted rollout PID is absent; its ownership cannot be proved')
    argv, environment = _process_command(pid)
    if str(environment.get('SLURM_JOB_ID') or environment.get('SLURM_JOBID')) != str(contract['job_id']):
        raise RuntimeError('Adopted rollout PID belongs to a different Slurm allocation')
    flattened = _flatten_shell_arguments(argv)
    supervisor = any(value in {'RL_Framework.scripts.restartable_vllm_server'} or
                     value.endswith('/restartable_vllm_server.py') for value in argv)
    server = any(value in {'RL_Framework.scripts.vllm_hot_reload_api_server',
                           'RL_Framework.tools.dependency_probe_server', 'vllm.entrypoints.openai.api_server'}
                 or value.endswith(('/vllm_hot_reload_api_server.py', '/dependency_probe_server.py')) for value in argv)
    if launcher:
        if not argv or Path(argv[0]).name != 'srun':
            raise RuntimeError('PID is not the expected local Slurm rollout launcher')
        node = _argument_value(flattened, '--nodelist') or _argument_value(flattened, '-w')
        if not node or node.split('.', 1)[0] != str(contract['host']).split('.', 1)[0]:
            raise RuntimeError('Adopted srun launcher targets a different rollout node')
    elif not supervisor and not server:
        raise RuntimeError('Adopted PID is not an expected vLLM service or supervisor')
    if (supervisor or launcher) and _argument_value(flattened, '--instance-id') != contract['instance_id']:
        raise RuntimeError('Adopted rollout instance identity does not match')
    port = _argument_value(flattened, '--port')
    health = _argument_value(flattened, '--health-url')
    if str(port) != str(contract['port']) and not (health and health.endswith(':' + str(contract['port']) + '/health')):
        raise RuntimeError('Adopted rollout port does not match')
    control = contract.get('control_dir')
    if control and supervisor and _argument_value(flattened, '--control-dir') != control:
        raise RuntimeError('Adopted rollout belongs to a different run control directory')
    return record


def _stop_verified_pid_tree(pid, expected_start, timeout):
    records = _owned_descendants(pid)
    root = records.get(pid)
    if root is None or root['start'] != expected_start:
        raise RuntimeError('Adopted rollout PID changed before termination')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = _process_record(pid)
        if current and current['start'] == expected_start:
            records.update(_owned_descendants(pid))
        for record in reversed(list(records.values())):
            if _still_owned(record):
                try:
                    os.kill(record['pid'], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while any(_still_owned(record) for record in records.values()) and (deadline is None or time.monotonic() < deadline):
            time.sleep(.02)
        if not any(_still_owned(record) for record in records.values()):
            return dict(pid=pid, process_start=expected_start, owned_processes_exited=True)
    raise RuntimeError('Adopted rollout process tree remains live after termination')


def _stop_adopted_on_host(contract, timeout):
    """Run only on the host owning the service; never interpret its PID locally."""
    if str(os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')) != str(contract['job_id']):
        raise RuntimeError('Adopted rollout stop is outside its Slurm allocation')
    if contract.get('discover_service'):
        candidates = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                record = _verify_adopted_identity(int(path.name), contract)
            except RuntimeError:
                continue
            candidates.append(record)
        ids = {record['pid'] for record in candidates}
        roots = [record for record in candidates if record['parent'] not in ids]
        # A server is usually a supervisor's child. Intermediate shell parents
        # still leave one root supervisor; multiple independent roots are unsafe.
        if len(roots) != 1:
            raise RuntimeError('Known launcher does not identify one unique owned rollout service tree')
        record = roots[0]
    else:
        record = _verify_adopted_identity(int(contract['pid']), contract)
    result = _stop_verified_pid_tree(record['pid'], record['start'], timeout)
    return dict(result, instance_id=contract['instance_id'], port=contract['port'], job_id=contract['job_id'])


def stop_adopted_rollout(meta, env=None, timeout=30., control_dir=None):
    """Validate host, role, run and job ownership before stopping an adopted PID.

    Old registries sometimes store the controller's local srun PID beside the
    remote service host. Only a verified launcher permits lookup of the unique
    matching service on that remote node; a bare PID never authorizes a scan.
    """
    env = dict(os.environ if env is None else env)
    job = env.get('SLURM_JOB_ID') or env.get('SLURM_JOBID')
    if not job or int(meta.pid) <= 0:
        raise RuntimeError('Adopted rollout stop requires a known PID and current Slurm allocation')
    contract = dict(pid=int(meta.pid), host=meta.host, instance_id=meta.instance_id,
                    port=int(meta.port), job_id=str(job), control_dir=str(control_dir) if control_dir else None)
    if is_local_host(meta.host, env):
        return _stop_adopted_on_host(contract, timeout)
    target_node(meta.host, env)  # Prove allocation ownership before touching any PID.
    local_launcher = None
    try:
        local_launcher = _verify_adopted_identity(int(meta.pid), contract, launcher=True)
    except RuntimeError:
        pass
    if local_launcher is not None:
        contract['discover_service'] = True
    argv = [sys.executable, '-m', 'RL_Framework.infra.elastic.rollout_process',
            '--stop-adopted', json.dumps(contract, separators=(',', ':')), '--timeout', str(timeout)]
    command = slurm_command(shlex.join(argv), meta.host, meta.gpus, env)
    result = subprocess.run(shlex.split(command), env=env, check=True, capture_output=True, text=True,
                            timeout=3*timeout+30 if timeout > 0 else None)
    try:
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError('Remote adopted rollout stop did not provide an exit receipt') from exc
    if (receipt.get('owned_processes_exited') is not True or receipt.get('job_id') != str(job)
            or receipt.get('instance_id') != meta.instance_id or receipt.get('port') != int(meta.port)):
        raise RuntimeError('Remote adopted rollout exit receipt differs from the requested owner')
    if local_launcher is not None and _still_owned(local_launcher):
        _stop_verified_pid_tree(local_launcher['pid'], local_launcher['start'], timeout)
    return receipt

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--token')
    parser.add_argument('--stop-adopted', type=json.loads)
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.stop_adopted is not None:
        result = _stop_adopted_on_host(args.stop_adopted, args.timeout)
        print(json.dumps(result), flush=True)
        return 0
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.state or not args.token or not args.command or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise RuntimeError('Owned remote rollout runner requires a real Slurm job step')
    owner = dict(token=args.token, job_id=os.environ['SLURM_JOB_ID'], step_id=os.environ['SLURM_STEP_ID'],
                 host=socket.gethostname(), status='starting')
    _write(args.state, owner)
    def terminate(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    proc = None
    try:
        proc = subprocess.Popen(args.command, start_new_session=True)
        _write(args.state, dict(owner, status='running', child_pid=proc.pid))
        return proc.wait()
    finally:
        if proc is not None:
            stop_owned_session(proc)
        _write(args.state, dict(owner, status='exited', owned_processes_exited=True))


if __name__ == '__main__':
    sys.exit(main())
