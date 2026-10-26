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
        script = runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "generate.py")
        )
        self.main = script["main"]
        self.format = script["FORMAT"]
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

    def package(self, dtype: str | None, version: int | None = None) -> Path:
        path = self.root / "existing"
        path.mkdir(exist_ok=True)
        model = {"max_context": 64}
        if dtype is not None:
            model["weight_dtype"] = dtype
        (path / "plan.json").write_text(
            json.dumps(
                {
                    "header": {"format_version": version or self.format},
                    "source": {"repository": "meta-llama/Llama-3.2-1B"},
                    "model": model,
                }
            )
        )
        return path

    def test_new_package_precision(self) -> None:
        for arguments, dtype in (((), "bfloat16"), (("--dtype", "float32"), "float32")):
            with self.subTest(arguments=arguments):
                commands = self.invoke(*arguments)
                command = next(item for item in commands if "import-llama32" in item)
                self.assertEqual(command[command.index("--dtype") + 1], dtype)

    def test_existing_package_inherits_precision(self) -> None:
        for dtype, arguments in (
            ("float32", ()),
            ("bfloat16", ()),
            ("bfloat16", ("--dtype", "bfloat16")),
        ):
            with self.subTest(dtype=dtype, arguments=arguments):
                package = self.package(dtype)
                commands = self.invoke("--package", str(package), *arguments)
                self.assertFalse(any("import-llama32" in item for item in commands))

    def test_existing_package_rejects_explicit_precision_conflicts(self) -> None:
        package = self.package("bfloat16")
        with self.assertRaises(SystemExit) as error:
            self.invoke("--package", str(package), "--dtype", "float32")
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--dtype conflicts", self.stderr.getvalue())

    def test_retired_package_format_is_rejected(self) -> None:
        package = self.package("float32", version=2)
        with self.assertRaises(SystemExit) as error:
            self.invoke("--package", str(package))
        self.assertEqual(error.exception.code, 2)
        self.assertIn("regenerate", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
