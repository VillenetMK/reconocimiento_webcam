import contextlib
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="webcam proyecto ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / ".venv_vision"
        self.env.mkdir()
        self.python = self.env / "python de prueba"
        self.python.touch()
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("opencv-python>=4.11\n", encoding="utf-8")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(bootstrap, "ROOT", self.root))
        self.stack.enter_context(patch.object(bootstrap, "ENV", self.env))
        self.stack.enter_context(patch.object(bootstrap, "environment_python", return_value=self.python))

    def test_ready_environment_does_not_reinstall(self):
        digest = hashlib.sha256(self.requirements.read_bytes()).hexdigest()
        (self.env / "webcam_requirements.sha256").write_text(digest, encoding="ascii")
        with patch.object(bootstrap.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
             patch.object(bootstrap, "run") as install:
            self.assertEqual(bootstrap.prepare(), self.python)
            install.assert_not_called()

    def test_failed_install_is_not_marked_ready(self):
        with patch.object(bootstrap, "run", side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap.prepare()
        self.assertFalse((self.env / "webcam_requirements.sha256").exists())

    def test_preserves_spaced_paths_arguments_and_exit_status(self):
        with patch.object(bootstrap, "prepare", return_value=self.python), \
             patch.object(sys, "argv", ["bootstrap.py", "--camara", "1"]), \
             patch.object(bootstrap.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as execute:
            self.assertEqual(bootstrap.main(), 7)
            self.assertEqual(execute.call_args.args[0],
                             [str(self.python), str(self.root / "PY1.PY"), "--camara", "1"])


if __name__ == "__main__":
    unittest.main()
