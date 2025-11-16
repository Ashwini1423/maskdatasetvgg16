"""Performance timing utilities."""

import time
from typing import Dict, List
from contextlib import contextmanager


class PerformanceTimer:
    """Performance timer for tracking pipeline stage timings."""

    def __init__(self):
        """Initialize performance timer."""
        self.timings = {}
        self.current_stage = None
        self.start_time = None

    @contextmanager
    def measure(self, stage_name: str):
        """
        Context manager for measuring stage execution time.

        Args:
            stage_name: Name of the pipeline stage

        Usage:
            with timer.measure('preprocessing'):
                # preprocessing code
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            if stage_name not in self.timings:
                self.timings[stage_name] = []
            self.timings[stage_name].append(elapsed)

    def get_average_timings(self) -> Dict[str, float]:
        """
        Get average timing for each stage.

        Returns:
            Dictionary of stage names to average times (ms)
        """
        averages = {}
        for stage, times in self.timings.items():
            averages[stage] = sum(times) / len(times) if times else 0.0
        return averages

    def get_total_time(self) -> float:
        """
        Get total average pipeline time.

        Returns:
            Total time in milliseconds
        """
        averages = self.get_average_timings()
        return sum(averages.values())

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed statistics for each stage.

        Returns:
            Dictionary with min, max, avg, and count for each stage
        """
        stats = {}
        for stage, times in self.timings.items():
            if times:
                stats[stage] = {
                    'min': min(times),
                    'max': max(times),
                    'avg': sum(times) / len(times),
                    'count': len(times)
                }
        return stats

    def print_report(self):
        """Print performance report."""
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)

        stats = self.get_statistics()

        for stage, stage_stats in stats.items():
            print(f"\n{stage}:")
            print(f"  Average: {stage_stats['avg']:.2f} ms")
            print(f"  Min:     {stage_stats['min']:.2f} ms")
            print(f"  Max:     {stage_stats['max']:.2f} ms")
            print(f"  Frames:  {stage_stats['count']}")

        total = self.get_total_time()
        fps = 1000 / total if total > 0 else 0

        print("\n" + "-" * 60)
        print(f"Total Pipeline Time: {total:.2f} ms")
        print(f"Estimated FPS:       {fps:.2f}")
        print("=" * 60 + "\n")

    def reset(self):
        """Reset all timings."""
        self.timings = {}
