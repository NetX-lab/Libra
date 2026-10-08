"""Run one explicit command on the authorized Libra node list via jump host.

Run on the jump host. Password is read from its terminal and passed to expect
through an environment variable, never embedded in command arguments or files.
"""
import argparse
import concurrent.futures
import getpass
import json
import os
import subprocess
from pathlib import Path

EXPECT_SCRIPT = r'''
set timeout $env(LIBRA_TIMEOUT)
log_user 0
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@$env(LIBRA_NODE) $env(LIBRA_COMMAND)
expect {
    -re {[Pp]assword:} {send -- "$env(LIBRA_SSH_PASSWORD)\r"}
    eof {puts $expect_out(buffer); catch wait result; exit [lindex $result 3]}
    timeout {exit 124}
}
log_user 1
expect {
    -re {[Pp]assword:} {exit 77}
    eof {catch wait result; exit [lindex $result 3]}
    timeout {exit 124}
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', required=True, help='Comma separated full IPv4 addresses')
    parser.add_argument('--command-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    command = Path(args.command_file).read_text()
    password = getpass.getpass('Internal SSH password: ')
    def run(ip):
        env = dict(os.environ, LIBRA_NODE=ip, LIBRA_COMMAND=command,
                   LIBRA_SSH_PASSWORD=password, LIBRA_TIMEOUT=str(args.timeout))
        try:
            p = subprocess.run(['expect', '-c', EXPECT_SCRIPT], env=env,
                               capture_output=True, text=True, timeout=args.timeout+15)
            return dict(ip=ip, returncode=p.returncode, output=p.stdout+p.stderr)
        except subprocess.TimeoutExpired:
            return dict(ip=ip, returncode=124, output='timeout')
    nodes = list(dict.fromkeys(args.nodes.split(',')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, nodes))
    Path(args.output).write_text(json.dumps(results, indent=2))
    for r in results:
        print(r['ip'], r['returncode'], r['output'][-160:])


if __name__ == '__main__':
    main()
