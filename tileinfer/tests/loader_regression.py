"""Exercise structural package validation without probing or using a GPU."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

from tileplan.decoder_plan import build
from tileplan.decoder_stack import build_stack
from tileplan.probe import normalize


def rearrange(plan: dict) -> None:
    """Preserve dependencies while choosing a different legal worker assignment."""
    workers = plan["target"]["worker_count"]
    for task in plan["tasks"]:
        task["worker"] = (task["worker"] + 1) % workers
    for wave in plan["waves"]:
        wave["workers"] = [wave["workers"][-1]] + wave["workers"][:-1]
        wave["estimated_work"] = [wave["estimated_work"][-1]] + wave["estimated_work"][
            :-1
        ]
        for lane in wave["workers"]:
            lane.reverse()
    plan["tasks"].reverse()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, required=True)
    args = parser.parse_args()
    fixture_root = Path(__file__).parents[2] / "tileplan/tests/fixtures"

    def run(*command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(args.loader), *command], capture_output=True, text=True, timeout=20
        )

    raw = run("--describe-target-json")
    raw.check_returncode()
    with tempfile.TemporaryDirectory(prefix="cpu-loader-") as directory:
        root = Path(directory)
        target = root / "target.json"
        target.write_text(json.dumps(normalize(json.loads(raw.stdout))))
        cases = 0
        for kind, builder, fixture in (
            ("block", build, "decoder_m1"),
            ("stack", build_stack, "decoder_m4"),
        ):
            source = fixture_root / fixture
            for dtype in ("float32", "bfloat16"):
                package = root / f"{kind}-{dtype}"
                builder(
                    source / "model.json",
                    source / "weights.npz",
                    target,
                    7,
                    package,
                    dtype,
                )
                path = package / "plan.json"
                original = json.loads(path.read_text())

                def check(plan: dict, success: bool, label: str) -> None:
                    nonlocal cases
                    path.write_text(json.dumps(plan))
                    result = run(str(package))
                    if (
                        result.returncode == 0
                    ) != success or "runtime error:" in result.stderr:
                        raise AssertionError(f"{kind}/{dtype}/{label}: {result.stderr}")
                    cases += 1

                check(original, True, "canonical")
                legal = copy.deepcopy(original)
                rearrange(legal)
                check(legal, True, "alternative assignment and ordering")
                legal["model"]["tile_rows"] = 2**31 - 1
                check(legal, True, "large tile metadata does not expand a graph")
                if dtype == "float32":
                    legacy = copy.deepcopy(original)
                    legacy["header"].update(
                        format_version=2, executor_abi="tileinfer-decoder-v2"
                    )
                    legacy["target"]["kernel_fingerprint"] = (
                        "decoder-wave-abi2-ops-v3-stack-sm86-b128"
                    )
                    check(legacy, True, "version 2 compatibility")
                mutations = {
                    "derived frequency overflow": lambda p: p["model"].update(
                        rope_factor=1e-45
                    ),
                    "noninteger wave": lambda p: p["waves"][0].update(id=0.0),
                    "operand": lambda p: p["tasks"][0]["operands"].update(scale=15),
                    "noninteger operand": lambda p: p["tasks"][0]["operands"].update(
                        scale=1.0
                    ),
                    "extra operand": lambda p: p["tasks"][0]["operands"].update(
                        extra=0
                    ),
                    "dependency": lambda p: p["tasks"][-1].update(predecessors=[]),
                    "ID": lambda p: p["tasks"][1].update(id=0),
                    "wave": lambda p: p["tasks"][-1].update(wave=0),
                    "overlap": lambda p: p["tasks"].append(
                        copy.deepcopy(p["tasks"][1])
                    ),
                    "context overflow": lambda p: p["model"].update(
                        max_context=2**31 - 1
                    ),
                    "width overflow": lambda p: p["model"].update(
                        intermediate_size=2**31 - 1
                    ),
                    "layers overflow": lambda p: p["model"].update(layers=2**31 - 1),
                    "byte offset overflow": lambda p: p["storage"]["weights"][1].update(
                        offset=2**32 + 64
                    ),
                    "unsigned dimension": lambda p: p["model"].update(
                        hidden_size=2**64 - 1
                    ),
                    "epsilon underflow": lambda p: p["model"].update(
                        rms_norm_eps=1e-300
                    ),
                    "worker index": lambda p: p["waves"][0]["workers"][0].append(
                        2**64 - 1
                    ),
                    "missing task": lambda p: p["waves"][0]["workers"].__setitem__(
                        0, []
                    ),
                }
                for label, mutate in mutations.items():
                    invalid = copy.deepcopy(original)
                    mutate(invalid)
                    check(invalid, False, label)
                path.write_text(json.dumps(original))
        print(f"{cases} CPU package and UBSan validation cases passed")


if __name__ == "__main__":
    main()
