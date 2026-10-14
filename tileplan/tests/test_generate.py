"""Check generation defaults and package reuse without a checkpoint or GPU."""

import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


class GenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.main = runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "generate.py")
        )["main"]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoints" / "Llama-3.2-1B"
        self.checkpoint.mkdir(parents=True)
        tokenizer = Mock(eos_token_id=None)
        tokenizer.return_value = {"input_ids": [1, 2]}
        tokenizer.decode.return_value = "continuation"
        self.transformers = ModuleType("transformers")
        self.transformers.__dict__["AutoTokenizer"] = Mock(
            from_pretrained=Mock(return_value=tokenizer)
        )

    def invoke(self, *arguments: str) -> list[list[str]]:
        commands = Mock()
        plan = self.root / "tileplan"
        infer = self.root / "tileinfer"
        environment = {
            "ROOT": self.root,
            "PLAN": plan,
            "INFER": infer,
            "CHECKPOINT": self.checkpoint,
            "EXECUTOR": infer / "build" / "release" / "tile_runtime",
            "VENV": plan / ".venv",
            "run": commands,
        }
        result = subprocess.CompletedProcess(
            [], 0, json.dumps({"generated": [3], "generated_count": 1, "compute_us": 1})
        )
        self.stderr = io.StringIO()
        with (
            patch.dict(self.main.__globals__, environment),
            patch.object(sys, "prefix", str(environment["VENV"])),
            patch.object(sys, "argv", ["generate.py", "prompt", *arguments]),
            patch.dict(sys.modules, {"transformers": self.transformers}),
            patch.object(subprocess, "run", return_value=result),
            redirect_stdout(io.StringIO()),
            redirect_stderr(self.stderr),
        ):
            self.main()
        return [
            [str(part) for part in call.args[0]] for call in commands.call_args_list
        ]

    def package(self, dtype: str | None, kernel: str, version: int = 4) -> Path:
        path = self.root / "existing"
        path.mkdir(exist_ok=True)
        model = {"matrix_kernel": kernel, "max_context": 64}
        if dtype is not None:
            model["weight_dtype"] = dtype
        (path / "plan.json").write_text(
            json.dumps(
                {
                    "header": {"format_version": version},
                    "source": {"repository": "meta-llama/Llama-3.2-1B"},
                    "model": model,
                }
            )
        )
        return path

    def test_new_package_precision(self) -> None:
        for arguments, dtype, kernel in (
            ((), "bfloat16", "simt"),
            (("--dtype", "float32"), "float32", "simt"),
            (("--matrix-kernel", "mma"), "bfloat16", "mma"),
        ):
            with self.subTest(arguments=arguments):
                commands = self.invoke(*arguments)
                command = next(item for item in commands if "import-llama32" in item)
                self.assertEqual(command[command.index("--dtype") + 1], dtype)
                self.assertEqual(command[command.index("--matrix-kernel") + 1], kernel)

    def test_existing_package_inherits_precision(self) -> None:
        for dtype, kernel, version, arguments in (
            ("float32", "simt", 4, ()),
            ("bfloat16", "simt", 4, ()),
            ("bfloat16", "mma", 4, ()),
            (None, "simt", 2, ("--dtype", "float32")),
            ("bfloat16", "mma", 4, ("--dtype", "bfloat16", "--matrix-kernel", "mma")),
        ):
            with self.subTest(dtype=dtype, kernel=kernel, version=version):
                package = self.package(dtype, kernel, version)
                commands = self.invoke("--package", str(package), *arguments)
                self.assertFalse(any("import-llama32" in item for item in commands))

    def test_existing_package_rejects_explicit_precision_conflicts(self) -> None:
        package = self.package("bfloat16", "mma")
        for option, value in (("--dtype", "float32"), ("--matrix-kernel", "simt")):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit) as error:
                    self.invoke("--package", str(package), option, value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(f"{option} conflicts", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
