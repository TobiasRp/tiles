"""Exercise structural package validation without probing or using a GPU."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

from tileplan.decoder_block import build
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


def delay_producer(plan: dict) -> None:
    """Keep schedule membership valid but race the input normalization with GEMV."""
    task = plan["tasks"][0]
    worker = task["worker"]
    plan["waves"][task["wave"]]["workers"][worker].remove(task["id"])
    task["wave"] += 1
    plan["waves"][task["wave"]]["workers"][worker].append(task["id"])


def stale_scratch_read(plan: dict) -> None:
    """Run a layer-0 attention task after layer 1 has overwritten the shared Q."""
    tasks = plan["tasks"]
    early = next(
        t
        for t in tasks
        if t["layer"] == 0 and t["opcode"].startswith("attention_scores")
    )
    late = next(
        t
        for t in tasks
        if t["layer"] == 1 and t["opcode"].startswith("attention_scores")
    )
    plan["waves"][early["wave"]]["workers"][early["worker"]].remove(early["id"])
    early["wave"] = late["wave"]
    plan["waves"][late["wave"]]["workers"][early["worker"]].append(early["id"])


def qkv_tasks(plan: dict) -> list[dict]:
    """The first layer's Q/K/V tasks in row order."""
    tasks = [
        t
        for t in plan["tasks"]
        if t["layer"] in (None, 0) and t["opcode"].startswith("qkv")
    ]
    return sorted(tasks, key=lambda t: t["operands"]["rows"][0])


def cross_qk_boundary(plan: dict) -> None:
    """Move the Q/K boundary inside a task: legal, ownership stays complete."""
    tasks = qkv_tasks(plan)
    d = plan["model"]["hidden_size"]
    first = next(t for t in tasks if t["operands"]["rows"][1] == d)
    second = next(t for t in tasks if t["operands"]["rows"][0] == d)
    end = second["operands"]["rows"][1]
    split = d + (end - d) // 2
    q_id = first["output"][0]["tensor"]
    k_id = second["output"][0]["tensor"]
    first["operands"]["rows"][1] = split
    first["output"].append({"tensor": k_id, "rows": [0, split - d]})
    first["estimated_work"] += (split - d) * d
    second["operands"]["rows"][0] = split
    second["output"] = [{"tensor": k_id, "rows": [split - d, end - d]}]
    second["estimated_work"] -= (split - d) * d
    assert first["output"][0]["tensor"] == q_id


def drop_k_output(plan: dict) -> None:
    """A Q/K/V task that projects K rows but does not declare them."""
    task = next(
        t
        for t in qkv_tasks(plan)
        if t["operands"]["rows"][0] >= plan["model"]["hidden_size"]
    )
    task["output"] = []


def rotate_in_later_chunk(plan: dict) -> None:
    """Move rotated Q and the K cache row from chunk 0 to chunk 1."""
    scores = [t for t in plan["tasks"] if t["opcode"].startswith("attention_scores")]
    first = next(t for t in scores if t["operands"]["chunk"] == 0)
    later = next(
        t
        for t in scores
        if t["layer"] == first["layer"]
        and t["operands"] == {"kv_head": first["operands"]["kv_head"], "chunk": 1}
    )
    later["output"] += first["output"][1:]
    del first["output"][1:]


def ffn_tasks(plan: dict, opcode: str = "ffn") -> list[dict]:
    """The first layer's FFN chunk tasks (or reduction tasks) in plan order."""
    return [
        t
        for t in plan["tasks"]
        if t["layer"] == 0 and t["opcode"].rsplit("_", 1)[0] == opcode
    ]


def renumber_chunks(plan: dict) -> None:
    """Swap two FFN chunks' partial slots: a legal plan with another sum order."""
    first, second = ffn_tasks(plan)[:2]
    for task, chunk in ((first, 1), (second, 0)):
        task["operands"]["chunk"] = chunk
        task["output"][1]["chunk"] = chunk


def duplicate_partial(plan: dict) -> None:
    """Give two FFN chunks the same partial slot."""
    task = ffn_tasks(plan)[1]
    task["operands"]["chunk"] = 0
    task["output"][1]["chunk"] = 0


def extra_chunk(plan: dict) -> None:
    """Declare one more FFN chunk, with its workspace, than any task writes."""
    plan["model"]["ffn_chunks"] += 1
    plan["storage"]["ffn_partial_bytes"] += plan["model"]["hidden_size"] * 4


def reduce_in_ffn_wave(plan: dict) -> None:
    """Run the first layer's reduction in the FFN wave, before the partials exist."""
    for task in ffn_tasks(plan, "ffn_reduce"):
        plan["waves"][task["wave"]]["workers"][task["worker"]].remove(task["id"])
        task["wave"] -= 1
        plan["waves"][task["wave"]]["workers"][task["worker"]].append(task["id"])


def conditional_wave(plan: dict) -> dict:
    """The first scores wave, whose barrier fused prefixes skip."""
    return next(wave for wave in plan["waves"] if "conditional_barrier" in wave)


def move_condition(plan: dict) -> None:
    """Skip the barrier after the Q/K/V wave instead, before attention reads K."""
    wave = conditional_wave(plan)
    plan["waves"][wave["id"] - 1]["conditional_barrier"] = wave.pop(
        "conditional_barrier"
    )


def fuse_into_projection(plan: dict) -> None:
    """Move layer 0's values tasks into the Wo wave, so a skipped barrier
    would let a fused prefix's output projection race its scores tasks."""
    wave = conditional_wave(plan)
    values, projection = plan["waves"][wave["id"] + 1], plan["waves"][wave["id"] + 2]
    for worker, lane in enumerate(values["workers"]):
        for task_id in lane:
            plan["tasks"][task_id]["wave"] += 1
        projection["workers"][worker] = lane + projection["workers"][worker]
        values["workers"][worker] = []
    projection["conditional_barrier"] = wave.pop("conditional_barrier")


def logits_tasks(plan: dict) -> list[dict]:
    return [task for task in plan["tasks"] if task["opcode"].startswith("logits")]


def set_candidate_slot(task: dict, slot: int) -> None:
    task["operands"]["chunk"] = slot
    task["output"][-1]["chunk"] = slot


def swap_candidate_slots(plan: dict) -> None:
    """Logits tasks may own any permutation of the candidate slots."""
    first, second = logits_tasks(plan)[:2]
    slot = first["operands"]["chunk"]
    set_candidate_slot(first, second["operands"]["chunk"])
    set_candidate_slot(second, slot)


def share_candidate_slot(plan: dict) -> None:
    first, second = logits_tasks(plan)[:2]
    set_candidate_slot(second, first["operands"]["chunk"])


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
                # Two-step chunks: fused and split attention, a conditional barrier.
                builder(
                    source / "model.json",
                    source / "weights.npz",
                    target,
                    package,
                    dtype,
                    2,
                    2,
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
                execution_only = copy.deepcopy(original)
                for task in execution_only["tasks"]:
                    task.pop("predecessors")
                    task.pop("estimated_work")
                for wave in execution_only["waves"]:
                    wave.pop("estimated_work")
                check(
                    execution_only, True, "planner metadata is not needed for execution"
                )
                legal = copy.deepcopy(original)
                rearrange(legal)
                check(legal, True, "alternative assignment and ordering")
                crossing = copy.deepcopy(original)
                cross_qk_boundary(crossing)
                check(crossing, True, "Q/K/V rows across the Q/K boundary")
                renumbered = copy.deepcopy(original)
                renumber_chunks(renumbered)
                check(renumbered, True, "FFN partial slots out of row order")
                for version in (2, 3):
                    retired = copy.deepcopy(original)
                    retired["header"].update(
                        format_version=version,
                        executor_abi=f"tileinfer-decoder-v{version}",
                        model_kind=f"llama_decoder_{kind}_f32",
                    )
                    retired["model"]["dtype"] = "float32"
                    check(retired, False, f"retired version {version} package")
                for version in (6, 7, 8, 9):
                    previous = copy.deepcopy(original)
                    previous["header"].update(
                        format_version=version,
                        executor_abi=f"tileinfer-decoder-v{version}",
                    )
                    check(previous, False, f"version {version} package")
                mutations = {
                    "read before producer": delay_producer,
                    "mixed activation dtype": lambda p: p["model"].update(
                        activation_dtype="float32"
                        if dtype == "bfloat16"
                        else "bfloat16"
                    ),
                    "cache dtype": lambda p: p["model"].update(
                        kv_dtype="float32" if dtype == "bfloat16" else "bfloat16"
                    ),
                    "attention workspace": lambda p: p["storage"].update(
                        attention_bytes=3
                    ),
                    "opcode precision": lambda p: p["tasks"][0].update(
                        opcode=p["tasks"][0]["opcode"]
                        .replace("_bf16", "_x")
                        .replace("_f32", "_bf16")
                        .replace("_x", "_f32")
                    ),
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
                    "missing conditional barrier": lambda p: conditional_wave(p).pop(
                        "conditional_barrier"
                    ),
                    "conditional barrier threshold": lambda p: conditional_wave(p)[
                        "conditional_barrier"
                    ].update(skip_up_to_length=3),
                    "conditional barrier before attention": move_condition,
                    "conditional barrier before a projection": fuse_into_projection,
                    "chunk steps beyond shared memory": lambda p: p["model"].update(
                        attention_chunk_steps=1 << 20
                    ),
                    "fused beyond chunk steps": lambda p: p["model"].update(
                        attention_fused_steps=3
                    ),
                    "too few chunks": lambda p: p["model"].update(attention_chunks=1),
                    "unaligned head": lambda p: p["model"].update(head_dim=6),
                    "undeclared K rows": drop_k_output,
                    "FFN chunk outside range": lambda p: ffn_tasks(p)[0][
                        "operands"
                    ].update(chunk=p["model"]["ffn_chunks"]),
                    "FFN partial slot shared": duplicate_partial,
                    "FFN reduction before its partials": reduce_in_ffn_wave,
                    "FFN workspace": lambda p: p["storage"].update(
                        ffn_partial_bytes=p["storage"]["ffn_partial_bytes"] + 4
                    ),
                    "FFN chunks without tasks": extra_chunk,
                    "rotation in a later chunk": rotate_in_later_chunk,
                }
                if kind == "stack":
                    swapped = copy.deepcopy(original)
                    swap_candidate_slots(swapped)
                    check(swapped, True, "candidate slots out of row order")
                    layers = original["model"]["layers"]
                    mutations |= {
                        "candidate slot outside range": lambda p: logits_tasks(p)[0][
                            "operands"
                        ].update(chunk=len(logits_tasks(p))),
                        "candidate slot shared": share_candidate_slot,
                        "selection workspace size": lambda p: p["storage"].update(
                            selection_bytes=p["storage"]["selection_bytes"] + 8
                        ),
                        "unaligned selection workspace": lambda p: p["storage"].update(
                            selection_bytes=p["storage"]["selection_bytes"] + 4
                        ),
                        "missing selection workspace": lambda p: p["storage"].pop(
                            "selection_bytes"
                        ),
                        "undeclared candidate": lambda p: logits_tasks(p)[0][
                            "output"
                        ].pop(),
                        "layer outside range": lambda p: next(
                            t for t in p["tasks"] if t["layer"] is not None
                        ).update(layer=layers),
                        "stage tagged as a layer": lambda p: p["tasks"][0].update(
                            layer=0
                        ),
                        "scratch reused by a later layer": stale_scratch_read,
                    }
                else:
                    mutations["block task without a layer"] = lambda p: p["tasks"][
                        0
                    ].update(layer=None)
                    mutations["selection workspace in a block"] = lambda p: p[
                        "storage"
                    ].update(selection_bytes=8)
                for label, mutate in mutations.items():
                    invalid = copy.deepcopy(original)
                    mutate(invalid)
                    check(invalid, False, label)
                path.write_text(json.dumps(original))
        print(f"{cases} CPU package and UBSan validation cases passed")


if __name__ == "__main__":
    main()
