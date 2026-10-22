"""Timeline alignment and bubble budget on a synthetic, known trace."""

import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from test_decoder_planning import target
from tileplan.decoder_stack import make_stack_plan, read_stack_model, read_stack_weights
from tileplan.timeline import align, budget, layer_window, render_timeline_svg

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m4"
TASK_NS = 1000.0
RELEASE_NS = 500.0


def synthetic_trace(plan: dict) -> tuple[dict, float]:
    """Each task takes TASK_NS; a barrier releases RELEASE_NS after the last arrival.

    Every worker's cycle counter runs at 2 GHz from its own random origin, and
    the global timer ticks in 1024 ns steps, as on the RTX 3050.
    """
    workers = plan["target"]["worker_count"]
    rng = np.random.default_rng(5)
    cycle_origin = rng.integers(0, 10**9, workers)
    streams: list[list[tuple[int, float, float]]] = [[] for _ in range(workers)]
    now = 0.0
    for wave_index, wave in enumerate(plan["waves"]):
        arrival = now
        finish = []
        for worker, lane in enumerate(wave["workers"]):
            clock = now
            for task in lane:
                streams[worker].append((task, clock, clock + TASK_NS))
                clock += TASK_NS
            finish.append(clock)
        arrival = max(finish)
        if wave_index + 1 < len(plan["waves"]):
            release = arrival + RELEASE_NS
            for worker in range(workers):
                streams[worker].append((-1, finish[worker], release))
            now = release
        else:
            now = arrival
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
        weights = read_stack_weights(FIXTURE / "weights.npz", spec)
        plan, _ = make_stack_plan(spec, weights, target(4), 7, "bfloat16")
        trace, total = synthetic_trace(plan)
        events, info = align(trace)
        self.assertAlmostEqual(info["traced_us"], total / 1e3, delta=1.0)
        # A short synthetic span leaves some 1024 ns timer quantization in the fit.
        for clock in info["sm_clock_ghz"]:
            self.assertAlmostEqual(clock, 2.0, delta=0.01)
        report = budget(plan, events, position=3, bandwidth_gbps=200.0)
        parts = report["ideal_stream_us"] + sum(report["components_us"].values())
        self.assertAlmostEqual(parts, report["traced_kernel_us"], places=6)
        waves = len(plan["waves"])
        self.assertAlmostEqual(
            report["components_us"]["barrier_sync"],
            (waves - 1) * RELEASE_NS / 1e3,
            delta=0.05 * (waves - 1),  # fitted clocks: within 50 ns per barrier
        )
        self.assertEqual(
            sum(p["waves"] for p in report["phases"].values()), len(plan["waves"])
        )
        svg = render_timeline_svg(
            plan, events, title="synthetic", window=layer_window(plan, events, 1)
        )
        self.assertEqual(
            ElementTree.fromstring(svg).tag, "{http://www.w3.org/2000/svg}svg"
        )


if __name__ == "__main__":
    unittest.main()
