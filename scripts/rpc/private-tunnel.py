#!/usr/bin/env python3
"""Supervise a separate, authenticated loopback RPC tunnel over a private route."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time


def receive(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError('RPC closed during health check')
        data.extend(part)
    return bytes(data)


def rpc_health(port):
    with socket.create_connection(('127.0.0.1', port), timeout=3) as sock:
        sock.settimeout(3)
        sock.sendall(bytes([14]) + struct.pack('<Q', 24) + bytes(24))
        size, = struct.unpack('<Q', receive(sock, 8))
        if size != 28:
            raise ValueError('Unexpected RPC HELLO size')
        hello = receive(sock, size)
        major, minor, patch = hello[:3]
        if major != 6 or minor > 2 or patch != 0:
            raise ValueError('Unsupported RPC protocol: %d.%d.%d' % (major, minor, patch))
        sock.sendall(bytes([15]) + struct.pack('<Q', 0))
        size, = struct.unpack('<Q', receive(sock, 8))
        if size != 4:
            raise ValueError('Unexpected device-count size')
        count, = struct.unpack('<I', receive(sock, size))
        if count != 1:
            raise ValueError('Expected a single-device worker')
        return {'protocol': '%d.%d.%d' % (major, minor, patch), 'devices': count}


def ssh_options(args):
    return ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UpdateHostKeys=no', '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no',
            '-o', 'ConnectTimeout=10',
            '-o', 'HostName=' + args.address, '-o', 'HostKeyAlias=' + args.host_key_alias,
            '-o', 'Compression=no', '-c', 'aes256-gcm@openssh.com']


def forward_command(args, control):
    return ['ssh', '-M', '-S', str(control), '-N', '-T',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', *ssh_options(args),
            '-L', '127.0.0.1:%d:127.0.0.1:%d' % (args.local_port, args.remote_port),
            '--', args.ssh_host]


def initialize_command(args, control):
    # Require this connection's master; do not silently open a different connection.
    return ['ssh', '-S', str(control), '-T', *ssh_options(args),
            '-o', 'ProxyCommand=false', '--', args.ssh_host, 'true']


def store(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({**state, 'updated_at_unix': time.time()}, indent=2) + '\n')
    temp.replace(path)


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run(args):
    directory = args.state_dir.expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_uid != os.getuid():
        raise ValueError('State directory must belong to this user')
    directory.chmod(0o700)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    with (directory / 'lock').open('a') as lock, tempfile.TemporaryDirectory(prefix='umm-rpc-ssh-') as temp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = Path(temp) / 'control'
        state = {'supervisor_pid': os.getpid(), 'ssh_pid': None, 'ready': False,
                 'endpoint': '127.0.0.1:%d' % args.local_port,
                 'remote_port': args.remote_port, 'address': args.address,
                 'host_key_alias': args.host_key_alias, 'control_socket': str(control), 'generation': 0}
        state_file = directory / 'state.json'
        try:
            while not stopped.is_set():
                state.update(ready=False, status='connecting', error=None, ssh_pid=None)
                store(state_file, state)
                child = subprocess.Popen(forward_command(args, control), stdin=subprocess.DEVNULL)
                try:
                    state['ssh_pid'] = child.pid
                    store(state_file, state)
                    deadline = time.monotonic() + 15
                    while not stopped.is_set() and child.poll() is None and not control.exists() and time.monotonic() < deadline:
                        stopped.wait(.1)
                    if stopped.is_set():
                        break
                    if child.poll() is not None or not control.exists():
                        raise ConnectionError('SSH master failed to start')
                    subprocess.run(initialize_command(args, control), check=True, timeout=12,
                                   stdin=subprocess.DEVNULL)
                    state.update(status='checking_rpc', session_initialized=True)
                    store(state_file, state)
                    # A busy serial worker may delay HELLO; keep this SSH connection.
                    while not stopped.is_set() and child.poll() is None:
                        try:
                            health = rpc_health(args.local_port)
                        except (OSError, ValueError) as error:
                            state.update(error=str(error), ready=False)
                            store(state_file, state)
                            stopped.wait(3)
                            continue
                        state.update(health=health, ready=True, status='ready', error=None,
                                     generation=state['generation'] + 1)
                        store(state_file, state)
                        # SSH keepalives check transport; do not probe an in-use worker.
                        while not stopped.wait(1) and child.poll() is None:
                            pass
                        break
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    state['error'] = str(error)
                finally:
                    stop_child(child)
                    control.unlink(missing_ok=True)
                    state.update(ready=False, status='reconnecting', ssh_pid=None, session_initialized=False)
                    store(state_file, state)
                stopped.wait(3)
        finally:
            state.update(ready=False, status='stopped', ssh_pid=None)
            store(state_file, state)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-host', default='hw19')
    parser.add_argument('--address', default='192.168.0.190')
    parser.add_argument('--host-key-alias', default='1.95.177.112')
    parser.add_argument('--local-port', type=int, default=50353)
    parser.add_argument('--remote-port', type=int, default=50053)
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1024 <= args.local_port <= 65535 or args.local_port in (50052, 50053):
        parser.error('Choose an independent local port; preserve 50052/50053')
    if not 1 <= args.remote_port <= 65535 or args.ssh_host.startswith('-'):
        parser.error('Invalid remote port or SSH host')
    return args


if __name__ == '__main__':
    run(parse_args())
