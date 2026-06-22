"""
Lightweight resource profiler for CIMS model runs.

Samples RSS memory and CPU usage in a background thread while tracemalloc
tracks Python-level peak allocations. Cross-platform: mac, Windows, Linux.

Usage
-----
    with ResourceProfiler() as prof:
        model.run(...)
    prof.report()
"""

import os
import threading
import time
import tracemalloc

import psutil


class ResourceProfiler:
    """
    Context manager that measures wall-clock time, peak RAM, and CPU% during a block.

    Parameters
    ----------
    sample_interval : float
        Seconds between background memory/CPU samples (default 5s).
    """

    def __init__(self, sample_interval: float = 5.0, context: dict = None):
        self._interval = sample_interval
        self._context = context or {}
        self._process = psutil.Process(os.getpid())
        self._samples: list[tuple[float, float, float]] = []  # (elapsed_s, rss_mb, cpu_pct)
        self._start: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.elapsed: float = 0.0
        self.peak_tracemalloc_mb: float = 0.0

    def __enter__(self):
        tracemalloc.start()
        self._start = time.time()
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_tracemalloc_mb = peak / 1e6
        self.elapsed = time.time() - self._start

    def _sample_loop(self):
        while not self._stop.wait(self._interval):
            try:
                rss_mb = self._process.memory_info().rss / 1e6
                cpu_pct = self._process.cpu_percent(interval=None)
                self._samples.append((time.time() - self._start, rss_mb, cpu_pct))
            except psutil.NoSuchProcess:
                break

    @property
    def peak_rss_mb(self) -> float:
        return max((s[1] for s in self._samples), default=0.0)

    @property
    def mean_cpu_pct(self) -> float:
        values = [s[2] for s in self._samples if s[2] > 0]
        return sum(values) / len(values) if values else 0.0

    def report(self) -> None:
        mins, secs = divmod(self.elapsed, 60)
        hrs, mins = divmod(mins, 60)

        if hrs >= 1:
            time_str = f"{int(hrs)}h {int(mins)}m {secs:.1f}s"
        elif mins >= 1:
            time_str = f"{int(mins)}m {secs:.1f}s"
        else:
            time_str = f"{secs:.1f}s"

        config_rows = [(k, str(v)) for k, v in self._context.items()]
        resource_rows = [
            ("Wall-clock time",   time_str),
            ("Peak RSS memory",   f"{self.peak_rss_mb:.0f} MB"),
            ("Peak Python alloc", f"{self.peak_tracemalloc_mb:.0f} MB  (tracemalloc; Python objects only)"),
            ("Mean CPU",          f"{self.mean_cpu_pct:.1f}%"),
            ("Samples taken",     str(len(self._samples))),
        ]

        all_rows = config_rows + resource_rows
        width = max(len(label) for label, _ in all_rows)

        lines = ["\n=== Resource Usage ==="]
        if config_rows:
            lines.append("  -- Configuration --")
            for label, value in config_rows:
                lines.append(f"  {label:<{width}} : {value}")
            lines.append("  -- Timing & Memory --")
        for label, value in resource_rows:
            lines.append(f"  {label:<{width}} : {value}")
        lines.append("=" * 22 + "\n")
        print("\n".join(lines))
