"""Repeatable CUDA/video integration checks; saves measurements, not claims of accuracy."""
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--image-size', type=int, choices=(512, 768, 1024), default=1024)
    args = p.parse_args()
    folder = ROOT / 'test_results'
    folder.mkdir(exist_ok=True)
    tracker = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=args.image_size)
    cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    ok, first = cap.read()
    assert ok
    box = [483, 122, 583, 232]  # red Y block in the bundled video
    rows = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i in range(args.frames):
        if i == 0:
            frame = first
        else:
            ok, frame = cap.read()
            if not ok:
                break
        result = tracker.step(frame, i / 30, box if i == 0 else None)
        rows.append({k: v for k, v in result.items() if k != 'mask'})
        if i in (0, 30, 60, 90):
            from camera import draw
            cv2.imwrite(str(folder / f'video_{args.image_size}_{i:03}.jpg'), draw(frame, result))
        assert result['memory'] <= 24
    elapsed = time.perf_counter() - start
    cap.release()
    video_peak = torch.cuda.max_memory_allocated() / 2**20
    # Stable scene exercises eviction beyond capacity, regardless of video loss.
    alloc = []
    ram = []
    for i in range(80):
        result = tracker.step(first, i / 30, box if i == 0 else None)
        assert result['status'] == 'TRACKING', result
        assert len(tracker.outputs['cond_frame_outputs']) == 1
        assert result['memory'] <= 24
        if i in (30, 50, 79):
            alloc.append(torch.cuda.memory_allocated() / 2**20)
            if __import__('sys').platform == 'win32':
                import ctypes
                from ctypes import wintypes
                class Counters(ctypes.Structure):
                    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                        (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
                        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                get_process = ctypes.windll.kernel32.GetCurrentProcess
                get_process.restype = wintypes.HANDLE
                get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
                get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
                assert get_memory(get_process(), ctypes.byref(counters), counters.cb)
                ram.append(counters.WorkingSetSize / 2**20)
    blank = np.zeros_like(first)
    lost = tracker.step(blank, 80 / 30)
    assert lost['box'] is None and lost['dx'] is None, lost
    locked = tracker.step(first, 81 / 30)
    assert locked['box'] is None, 'Must not automatically reacquire after loss'
    reset = tracker.step(first, 82 / 30, box)
    assert reset['status'] == 'TRACKING'
    gap = tracker.step(first, 90 / 30 + 2)
    assert gap['box'] is None
    # Explicitly skipped source frames still produce one continuous model history.
    for i in range(24):
        result = tracker.step(first, i * 4 / 30, box if i == 0 else None)
        assert result['status'] == 'TRACKING'
    assert abs(tracker.model.kf.dt - 4) < 1e-5
    assert tracker.index == 24
    times = [r['inference_ms'] for r in rows[5:] if r['inference_ms'] > 0]
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, image_size=args.image_size,
                  source='sam2/demo/data/gallery/03_blocks.mp4', frames=len(rows),
                  tracking_frames=sum(r['box'] is not None for r in rows),
                  video_wall_seconds=elapsed, steady_mean_ms=float(np.mean(times)) if times else None,
                  steady_p50_ms=float(np.median(times)) if times else None,
                  steady_p95_ms=float(np.percentile(times, 95)) if times else None,
                  video_peak_allocated_mib=video_peak, stable_allocated_mib=alloc,
                  stable_process_working_set_mib=ram,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  tests='video, bounded memory, blank loss, loss latch, reset, long gap, skipped frames PASS',
                  rows=rows)
    (folder / f'streaming_report_{args.image_size}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
