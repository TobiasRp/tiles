"""Timeline alignment and bubble budget on a synthetic, known trace."""

import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from test_decoder_planning import target

from tileplan.decoder_stack import make_stack_plan, read_stack_model, read_stack_weights
from tileplan.timeline import (
    align,
    budget,
    executed_barriers,
    layer_window,
    render_timeline_svg,
)

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m4"
TASK_NS = 1000.0
RELEASE_NS = 500.0


def synthetic_trace(plan: dict, position: int) -> tuple[dict, float]:
    """Each task takes TASK_NS; a barrier releases RELEASE_NS after the last
    arrival, and a skipped conditional barrier releases each worker at once.

    Every worker's cycle counter runs at 2 GHz from its own random origin, and
    the global timer ticks in 1024 ns steps, as on the RTX 3050.
    """
    workers = plan["target"]["worker_count"]
    rng = np.random.default_rng(5)
    cycle_origin = rng.integers(0, 10**9, workers)
    streams: list[list[tuple[int, float, float]]] = [[] for _ in range(workers)]
    clocks = [0.0] * workers
    executed = executed_barriers(plan, position)
    for wave_index, wave in enumerate(plan["waves"]):
        for worker, lane in enumerate(wave["workers"]):
            for task in lane:
                streams[worker].append((task, clocks[worker], clocks[worker] + TASK_NS))
                clocks[worker] += TASK_NS
        if wave_index + 1 < len(plan["waves"]):
            # A skipped barrier is a zero-length instruction on every worker.
            release = max(clocks) + RELEASE_NS
            for worker in range(workers):
                end = release if executed[wave_index] else clocks[worker]
                streams[worker].append((-1, clocks[worker], end))
                clocks[worker] = end
    now = max(clocks)
    record = {"start_cycle": [], "end_cycle": [], "start_ns": []}
    tasks, owners = [], []
    epoch = 7_000_000_000
    for worker, stream in enumerate(streams):
        for task, start, end in stream:
            tasks.append(task)
            owners.append(worker)
            record["start_cycle"].append(int(cycle_origin[worker] + 2 * start))
            record["end_cycle"].append(int(cycle_origin[worker] + 2 * end))
            record["start_ns"].append(int((epoch + start) // 1024 * 1024))
    trace = {
        "instruction_tasks": tasks,
        "instruction_workers": owners,
        "samples": [{"kernel_us": now / 1e3, **record}],
    }
    return trace, now


class TimelineTest(unittest.TestCase):
    def test_alignment_and_budget_recover_the_synthetic_schedule(self):
        spec = read_stack_model(FIXTURE / "model.json")
        weights = read_stack_weights(FIXTURE / "weights.npz")
        # Three chunks of the 6-step context; prefixes up to 2 steps run fused.
        plan, _ = make_stack_plan(spec, weights, target(4), "bfloat16", 2, 2)
        for position, skipped in ((1, 3), (4, 0)):
            with self.subTest(position=position):
                self.check_position(plan, position, skipped)

    def check_position(self, plan: dict, position: int, skipped: int) -> None:
        self.assertEqual(executed_barriers(plan, position).count(False), skipped)
        trace, total = synthetic_trace(plan, position)
        events, info = align(trace)
        self.assertAlmostEqual(info["traced_us"], total / 1e3, delta=1.0)
        # A short synthetic span leaves some 1024 ns timer quantization in the fit.
        for clock in info["sm_clock_ghz"]:
            self.assertAlmostEqual(clock, 2.0, delta=0.02)
        report = budget(plan, events, position=position, bandwidth_gbps=200.0)
        parts = report["ideal_stream_us"] + sum(report["components_us"].values())
        self.assertAlmostEqual(parts, report["traced_kernel_us"], places=6)
        barriers = len(plan["waves"]) - 1 - skipped
        self.assertAlmostEqual(
            report["components_us"]["barrier_sync"],
            barriers * RELEASE_NS / 1e3,
            delta=0.05 * barriers,  # fitted clocks: within 50 ns per barrier
        )
        self.assertEqual(
            sum(p["waves"] for p in report["phases"].values()), barriers + 1
        )
        svg = render_timeline_svg(
            plan, events, title="synthetic", window=layer_window(plan, events, 1)
        )
        self.assertEqual(
            ElementTree.fromstring(svg).tag, "{http://www.w3.org/2000/svg}svg"
        )


if __name__ == "__main__":
    unittest.main()
