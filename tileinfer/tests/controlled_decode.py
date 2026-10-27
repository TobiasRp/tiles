"""Controlled fixed-context decode timing: interleaved arms with telemetry.

Each TileInfer arm is a package served by a resident ``tile_runtime_test
--benchmark-serve`` process; ``--pytorch`` adds Transformers eager and CUDA
Graph replay, run in this process. For every context the arms fill their
prefix once, then run ``--blocks`` measurement blocks each, in alternating
order (forward, then reversed), so drift reaches every arm alike. A block is
``--rounds`` CUDA-event batches of ``--iterations`` launches. A background
``nvidia-smi`` log records clocks, temperature, power and clock-event reasons,
joined to every block by time.

The report keeps raw samples, per-block telemetry, hashes and the GPU state,
and compares every arm with the first through paired block differences:
blocks next to each other in time form a pair, and a bootstrap over pairs
gives the 95% interval of the mean difference. Two arms of one package (an
A/A run) measure the protocol's own resolution.

``--synthetic-prefix`` fills every arm's K/V prefix with deterministic values
instead of decoding it: attention cost does not depend on the values, so
timing is unchanged, but the next token is no longer compared with PyTorch.
"""

import argparse
import datetime
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TELEMETRY = (
    "timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
    "clocks_event_reasons.active"
)
BOOTSTRAP = 10_000
# Rounds this far from their arm's median at one context are hiccups (a
# delayed launch, not the kernel); they are excluded from the statistics and
# counted. Locked-clock rounds otherwise stay within about 0.05%.
OUTLIER_FRACTION = 0.0025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TileArm:
    """One package in a resident benchmark server."""

    def __init__(
        self, name: str, executor: Path, package: Path, synthetic: bool
    ) -> None:
        self.name = name
        self.executor = executor
        self.package = package
        command = [str(executor), "--benchmark-serve", str(package)]
        self.process = subprocess.Popen(
            command + (["--synthetic-prefix"] if synthetic else []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = json.loads(self._line())
        self.plan_sha256 = ready["plan_sha256"]
        self.weights_verified = ready["weights_verified"]

    def _line(self) -> str:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: benchmark server exited")
        return line

    def measure(
        self, position: int, token: int, warmup: int, iterations: int, rounds: int
    ) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(f"{position} {token} {warmup} {iterations} {rounds}\n")
        self.process.stdin.flush()
        return json.loads(self._line())

    def describe(self) -> dict:
        return {
            "kind": "tileinfer",
            "executor": str(self.executor),
            "executor_sha256": sha256(self.executor),
            "package": str(self.package),
            "plan_sha256": self.plan_sha256,
            # The server trusts the plan's weight checksum; run `tileplan verify` once.
            "weights_verified": self.weights_verified,
        }

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait(timeout=60) != 0:
            raise RuntimeError(f"{self.name}: benchmark server failed")


class PyTorchArms:
    """Transformers eager and CUDA Graph replay at one fixed context.

    The settings match benchmark_decode.py: cuBLASLt, no TF32 or reduced BF16
    reductions, eager attention, and a static cache of exactly the context
    length, refilled for every context, by decoding or with synthetic values.
    """

    def __init__(self, checkpoint: Path, dtype: str, synthetic: bool) -> None:
        os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
        import torch
        from transformers import AutoModelForCausalLM

        from tileplan.decoder_reference import pin_torch_numerics

        self.torch = torch
        pin_torch_numerics()
        self.scalar = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=self.scalar,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        self.checkpoint = checkpoint
        self.synthetic = synthetic
        self.graph = None

    def prepare(self, position: int, token: int, warmup: int) -> dict:
        from transformers import StaticCache

        torch = self.torch
        self.graph = None
        gc.collect()
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        with torch.inference_mode():
            cache = StaticCache(self.model.config, max_cache_len=position + 1)
            # One decoded step allocates the cache; synthetic rows replace the rest.
            for prefix in range(1 if self.synthetic else position):
                self.model(
                    input_ids=token_tensor,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([prefix], device=device),
                    logits_to_keep=1,
                )
            if self.synthetic:
                generator = torch.Generator(device=device).manual_seed(position)
                for layer in cache.layers:
                    for tensor in (layer.keys, layer.values):
                        assert tensor is not None
                        tensor[:, :, :position].uniform_(-1, 1, generator=generator)
            cache_position = torch.tensor([position], dtype=torch.long, device=device)
            mask = torch.zeros(
                (1, 1, 1, position + 1), device=device, dtype=self.scalar
            )

            def forward():
                return self.model(
                    input_ids=token_tensor,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    attention_mask=mask,
                    logits_to_keep=1,
                ).logits

            for _ in range(warmup):
                forward()
            logits = forward().detach().clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    forward()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_logits = forward()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.allclose(graph_logits, logits, atol=1e-3, rtol=1e-4):
                raise AssertionError(
                    f"graph/eager logits differ at position {position}"
                )
        self.forward, self.graph = forward, graph
        return {"next_token": int(logits.argmax()), "top_logit": float(logits.max())}

    def measure(self, name: str, iterations: int, rounds: int) -> dict:
        torch = self.torch
        assert self.graph is not None, "prepare() first"
        call = self.forward if name == "eager" else self.graph.replay
        samples = []
        with torch.inference_mode():
            for _ in range(rounds):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    call()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / iterations)
        return {"samples_us": samples, "mean_us": statistics.mean(samples)}

    def describe(self) -> dict:
        import transformers

        return {
            "kind": "pytorch",
            "checkpoint": str(self.checkpoint),
            "torch_version": self.torch.__version__,
            "transformers_version": transformers.__version__,
            "baseline": "LlamaForCausalLM eager attention, StaticCache of the context "
            "length, logits_to_keep=1, cuBLASLt, no TF32 or reduced BF16 reductions",
        }


class Telemetry:
    """A background nvidia-smi log, sampled every `interval_ms`."""

    def __init__(self, path: Path, interval_ms: int) -> None:
        self.path = path
        self.file = path.open("w")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={TELEMETRY}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={interval_ms}",
            ],
            stdout=self.file,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> list[dict]:
        self.process.terminate()
        self.process.wait(timeout=10)
        self.file.close()
        samples = []
        for line in self.path.read_text().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            try:
                stamp = datetime.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f")
                samples.append(
                    {
                        "time": stamp.timestamp(),
                        "sm_mhz": float(fields[1]),
                        "mem_mhz": float(fields[2]),
                        "temperature_c": float(fields[3]),
                        "power_w": float(fields[4]),
                        "reasons": int(fields[5], 16),
                    }
                )
            except ValueError:
                continue
        return samples


def block_telemetry(samples: list[dict], start: float, end: float) -> dict:
    inside = [s for s in samples if start <= s["time"] <= end]
    if not inside:
        return {"samples": 0}
    reasons = 0
    for sample in inside:
        reasons |= sample["reasons"]
    return {
        "samples": len(inside),
        "sm_mhz": [min(s["sm_mhz"] for s in inside), max(s["sm_mhz"] for s in inside)],
        "mem_mhz": [
            min(s["mem_mhz"] for s in inside),
            max(s["mem_mhz"] for s in inside),
        ],
        "temperature_c": max(s["temperature_c"] for s in inside),
        "power_w": statistics.mean(s["power_w"] for s in inside),
        "reasons": f"0x{reasons:x}",
    }


def paired(reference: list[float], other: list[float], seed: int) -> dict:
    """Mean of block-pair differences (other - reference) with a bootstrap 95% interval."""
    differences = np.array(other) - np.array(reference)
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, (BOOTSTRAP, len(differences))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    base = statistics.mean(reference)
    return {
        "pairs": len(differences),
        "mean_us": float(differences.mean()),
        "median_us": float(np.median(differences)),
        "stdev_us": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
        "ci95_us": [float(low), float(high)],
        "ci95_half_width_us": float(high - low) / 2,
        "mean_percent": 100 * float(differences.mean()) / base,
    }


def environment(repository: Path) -> dict:
    def query(*command: str) -> str:
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()

    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(),
        "commit": query("git", "-C", str(repository), "rev-parse", "HEAD"),
        "dirty": bool(query("git", "-C", str(repository), "status", "--porcelain")),
        "gpu": query(
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,"
            "persistence_mode",
            "--format=csv,noheader",
        ),
        "clocks": query("nvidia-smi", "-q", "-d", "CLOCK"),
        "compute_apps": query(
            "nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"
        ),
    }


def parse_arm(text: str) -> tuple[str, Path, Path]:
    name, _, paths = text.partition("=")
    executor, _, package = paths.partition(",")
    if not name or not executor or not package or name in ("eager", "graph"):
        raise argparse.ArgumentTypeError("arm must be NAME=EXECUTOR,PACKAGE")
    return name, Path(executor), Path(package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument(
        "--pytorch", type=Path, help="checkpoint for eager and graph arms"
    )
    parser.add_argument(
        "--positions", type=int, nargs="+", default=[0, 255, 1023, 4095]
    )
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--synthetic-prefix", action="store_true")
    parser.add_argument("--telemetry-ms", type=int, default=200)
    parser.add_argument("--note", default="", help="GPU state, e.g. locked clocks")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sorted(set(args.positions)) != args.positions or min(args.positions) < 0:
        parser.error("positions must be strictly increasing and nonnegative")
    if min(args.iterations, args.rounds, args.blocks) <= 0 or args.warmup < 0:
        parser.error("iterations, rounds and blocks must be positive")
    names = [name for name, _, _ in args.arm]
    if len(set(names)) != len(names):
        parser.error("arm names must be unique")

    args.out.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    report: dict = {
        "protocol": {
            "positions": args.positions,
            "token_id": args.token_id,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "blocks": args.blocks,
            "order": "per context: prefix and warmup per arm, then blocks in "
            "alternating forward and reversed arm order",
            "timing": "CUDA event batch average per round; kernel only for TileInfer",
            "comparison": "paired block differences against the first arm, "
            f"bootstrap 95% interval over pairs ({BOOTSTRAP} resamples); rounds "
            f"more than {100 * OUTLIER_FRACTION}% from the arm's median at a context "
            "are excluded and counted",
            "synthetic_prefix": args.synthetic_prefix,
            "note": args.note,
        },
        "environment": environment(repository),
    }
    telemetry = Telemetry(args.out / "telemetry.csv", args.telemetry_ms)
    tile = [
        TileArm(name, executor, package, args.synthetic_prefix)
        for name, executor, package in args.arm
    ]
    dtype = json.loads((tile[0].package / "plan.json").read_text())["model"][
        "weight_dtype"
    ]
    pytorch = (
        PyTorchArms(args.pytorch, dtype, args.synthetic_prefix)
        if args.pytorch
        else None
    )
    order = names + (["eager", "graph"] if pytorch else [])
    report["arms"] = {arm.name: arm.describe() for arm in tile}
    if pytorch:
        report["arms"] |= {name: pytorch.describe() for name in ("eager", "graph")}

    contexts: list[dict] = []
    try:
        for position in args.positions:
            print(f"context {position + 1}: preparing", file=sys.stderr, flush=True)
            reference = (
                pytorch.prepare(position, args.token_id, args.warmup)
                if pytorch
                else None
            )
            first = {}
            for arm in tile:
                result = arm.measure(position, args.token_id, args.warmup, 1, 1)
                first[arm.name] = (result["top_token"], result["top_logit"])
            top_token = first[names[0]][0]
            if any(token != top_token for token, _ in first.values()):
                raise AssertionError(
                    f"TileInfer arms disagree at position {position}: {first}"
                )
            if (
                reference
                and not args.synthetic_prefix
                and top_token != reference["next_token"]
            ):
                raise AssertionError(
                    f"next token differs from PyTorch at position {position}"
                )
            blocks = []
            for block in range(args.blocks):
                for name in order if block % 2 == 0 else order[::-1]:
                    start = time.time()
                    if name in names:
                        result = tile[names.index(name)].measure(
                            position, args.token_id, 0, args.iterations, args.rounds
                        )
                    else:
                        assert pytorch is not None
                        result = pytorch.measure(name, args.iterations, args.rounds)
                    blocks.append(
                        {
                            "arm": name,
                            "block": block,
                            "start": start,
                            "end": time.time(),
                            "samples_us": result["samples_us"],
                            "mean_us": statistics.mean(result["samples_us"]),
                        }
                    )
            contexts.append(
                {
                    "position": position,
                    "context_length": position + 1,
                    "top_token": top_token,
                    "top_logits": {name: logit for name, (_, logit) in first.items()},
                    "pytorch": reference,
                    "blocks": blocks,
                }
            )
    finally:
        samples = telemetry.stop()
        for arm in tile:
            arm.close()

    for context in contexts:
        for block in context["blocks"]:
            block["telemetry"] = block_telemetry(samples, block["start"], block["end"])
        means: dict[str, list[float]] = {name: [] for name in order}
        excluded = dict.fromkeys(order, 0)
        for name in order:
            blocks = [b for b in context["blocks"] if b["arm"] == name]
            center = statistics.median(x for b in blocks for x in b["samples_us"])
            for block in blocks:
                kept = [
                    x
                    for x in block["samples_us"]
                    if abs(x - center) <= OUTLIER_FRACTION * center
                ]
                excluded[name] += len(block["samples_us"]) - len(kept)
                # A block of only outliers keeps them rather than vanish from its pair.
                block["kept_mean_us"] = statistics.mean(kept or block["samples_us"])
                means[name].append(block["kept_mean_us"])
        context["summary"] = {
            name: {
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "cv_percent": 100 * statistics.pstdev(values) / statistics.mean(values),
                "excluded_rounds": excluded[name],
            }
            for name, values in means.items()
        }
        context["paired_vs_" + order[0]] = {
            name: paired(means[order[0]], means[name], context["position"])
            for name in order[1:]
        }
    report["contexts"] = contexts
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for context in contexts:
        line = [f"context {context['context_length']:5d}"]
        for name in order:
            summary = context["summary"][name]
            dropped = summary["excluded_rounds"]
            line.append(
                f"{name}={summary['median_us']:9.1f} us"
                + (f" ({dropped} rounds excluded)" if dropped else "")
            )
        print("  ".join(line))
        for name, stats in context["paired_vs_" + order[0]].items():
            low, high = stats["ci95_us"]
            print(
                f"    {name} - {order[0]}: {stats['mean_us']:+8.1f} us "
                f"[{low:+.1f}, {high:+.1f}] ({stats['mean_percent']:+.2f}%)"
            )
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
