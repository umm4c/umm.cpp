import importlib.util
from pathlib import Path
import struct
import types
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('tunnel', Path(__file__).resolve().parents[2] / 'scripts' / 'rpc' / 'private-tunnel.py')
tunnel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tunnel)


class Socket:
    def __init__(self, data):
        self.data = data
        self.sent = bytearray()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def settimeout(self, _):
        pass
    def sendall(self, data):
        self.sent.extend(data)
    def recv(self, n):
        # Fragment headers and bodies, as a real stream may do.
        part = self.data[:min(n, 3)]
        self.data = self.data[len(part):]
        return part


def reply(major=6, minor=1, patch_version=0, devices=1):
    return struct.pack('<Q', 28) + bytes([major, minor, patch_version, 0]) + bytes(24) + struct.pack('<QI', 4, devices)


class Tests(unittest.TestCase):
    def health(self, data):
        with patch.object(tunnel.socket, 'create_connection', return_value=Socket(data)):
            return tunnel.rpc_health(50353)
    def test_fragmented_61(self):
        self.assertEqual(self.health(reply()), {'protocol': '6.1.0', 'devices': 1})
    def test_fragmented_62(self):
        self.assertEqual(self.health(reply(minor=2))['protocol'], '6.2.0')
    def test_unknown_protocol(self):
        for data in (reply(major=7), reply(minor=3), reply(patch_version=1)):
            with self.assertRaises(ValueError):
                self.health(data)
    def test_device_count(self):
        for count in (0, 2):
            with self.assertRaises(ValueError):
                self.health(reply(devices=count))
    def test_unbounded_response(self):
        with self.assertRaises(ValueError):
            self.health(struct.pack('<Q', 2**63))
    def test_truncated(self):
        with self.assertRaises(ConnectionError):
            self.health(reply()[:19])
    def test_connection_failure(self):
        with patch.object(tunnel.socket, 'create_connection', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                tunnel.rpc_health(50353)
    def test_state_failure_after_spawn_cleans_child(self):
        stopped = threading.Event()
        child = Mock(pid=1234)
        child.poll.return_value = None
        child.wait.side_effect = lambda **_: stopped.set()
        calls = 0
        def store(*_):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('disk full')
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(state_dir=Path(directory), ssh_host='hw19', address='192.168.0.190', host_key_alias='1.95.177.112', local_port=50353, remote_port=50053)
            with patch.object(tunnel, 'store', side_effect=store), patch.object(tunnel.subprocess, 'Popen', return_value=child), patch.object(tunnel.signal, 'signal'), patch.object(tunnel.threading, 'Event', return_value=stopped):
                tunnel.run(args)
        child.terminate.assert_called_once()
        child.wait.assert_called_once()

    def test_authenticated_loopback_and_same_master(self):
        args = types.SimpleNamespace(ssh_host='hw19', address='192.168.0.190', host_key_alias='1.95.177.112', local_port=50353, remote_port=50053)
        forward = tunnel.forward_command(args, '/tmp/owned/control')
        initialize = tunnel.initialize_command(args, '/tmp/owned/control')
        self.assertIn('127.0.0.1:50353:127.0.0.1:50053', forward)
        for command in (forward, initialize):
            for option in ('StrictHostKeyChecking=yes', 'HostKeyAlias=1.95.177.112', 'HostName=192.168.0.190', 'BatchMode=yes', 'ForwardAgent=no'):
                self.assertIn(option, command)
            self.assertEqual(command[command.index('-S')+1], '/tmp/owned/control')
        self.assertIn('ProxyCommand=false', initialize)
        self.assertEqual(initialize[-1], 'true')


if __name__ == '__main__':
    unittest.main()
