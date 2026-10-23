"""Normalize decoder worker facts from the compiled CUDA executable."""

import json
import subprocess
from pathlib import Path

ABI = "tileinfer-decoder-v6"
OPCODES = {
    "qkv_f32",
    "attention_scores_f32",
    "attention_values_f32",
    "output_projection_f32",
    "gate_up_f32",
    "down_f32",
    "embedding_f32",
    "final_rms_norm_f32",
    "logits_f32",
}

OPCODES |= {opcode.replace("_f32", "_bf16") for opcode in OPCODES}


def normalize(raw: dict) -> dict:
    try:
        device, executor = raw["device"], raw["executor"]
        sm = device["sm_count"]
        active = executor["active_blocks_per_sm"]
        block = executor["threads_per_block"]
        capability = device["compute_capability"]
        if (
            type(sm) is not int
            or sm < 1
            or type(active) is not int
            or active < 1
            or type(block) is not int
            or block < 1
        ):
            raise ValueError("invalid residency or block size")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(n) is not int or n < 0 for n in capability)
        ):
            raise ValueError("invalid compute capability")
        if (
            device["cooperative_launch"] is not True
            or executor["abi"] != ABI
            or not OPCODES.issubset(set(executor["opcodes"]))
        ):
            raise ValueError("executor lacks required decoder capabilities")
        if not isinstance(executor["fingerprint"], str) or not executor["fingerprint"]:
            raise ValueError("missing compiled-kernel fingerprint")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"incomplete target description: {exc}") from exc
    return {
        "schema_version": 2,
        "device": device,
        "executor": executor,
        "launch": {
            "worker_count": sm,
            "grid_blocks": sm,
            "threads_per_block": block,
            "cooperative_grid_limit": sm * active,
        },
    }


def probe(executable: Path, out: Path) -> dict:
    completed = subprocess.run(
        [str(executable), "--describe-target-json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"target probe failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        target = normalize(json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise ValueError("executor did not return target JSON") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(target, sort_keys=True, indent=2) + "\n")
    return target


def read_target(path: Path) -> dict:
    target = json.loads(path.read_text())
    if target.get("schema_version") != 2 or target != normalize(
        {"device": target["device"], "executor": target["executor"]}
    ):
        raise ValueError("target.json is invalid or has an inconsistent launch")
    return target
