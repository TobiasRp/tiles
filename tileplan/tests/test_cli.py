import subprocess
import sys
import unittest


class CliScaffoldTest(unittest.TestCase):
    def test_help_lists_pipeline(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tileplan", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("reference", "probe", "build", "benchmark", "demo"):
            self.assertIn(command, result.stdout)

    def test_demo_cannot_claim_success_yet(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tileplan", "demo", "--out", "unused"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("implementation pending", result.stderr)


if __name__ == "__main__":
    unittest.main()
