"""Reproduce Windows deny-delete sharing and verify background writer recovery."""
import ctypes
from ctypes import wintypes
import json
import tempfile
import time
from pathlib import Path
from camera import TelemetryWriter, write_telemetry


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'target.json'
        write_telemetry(path, {'valid': False})
        create = ctypes.windll.kernel32.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        close = ctypes.windll.kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        handle = create(str(path), 0x80000000, 3, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value
        writer = TelemetryWriter(path)
        try:
            for seq in range(5):
                writer.submit({'valid': False, 'seq': seq})
            time.sleep(.2)
            assert writer.thread.is_alive()
            assert 'seq' not in json.loads(path.read_text())
            close(handle)
            handle = None
            deadline = time.monotonic() + 3
            while json.loads(path.read_text()).get('seq') != 4:
                assert time.monotonic() < deadline
                time.sleep(.05)
        finally:
            if handle is not None:
                close(handle)
            writer.close()
        assert json.loads(path.read_text())['status'] == 'STOPPED'
        assert not list(Path(directory).glob('*.tmp'))
    print('Windows locked-file recovery, latest-only queue, STOPPED and temp cleanup PASS')


if __name__ == '__main__':
    main()
