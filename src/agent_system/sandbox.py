from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int


class DockerSandboxRunner:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def run_tests(self, workspace_dir: Path) -> SandboxResult:
        if not self._config.sandbox_enabled:
            return SandboxResult(
                success=False,
                stdout="Sandbox disabled. Skipping real test run.",
                stderr="",
                exit_code=2,
            )

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--cpus=2",
            "--memory=4g",
            "-v",
            f"{workspace_dir.resolve()}:/workspace",
            "-w",
            "/workspace",
            self._config.sandbox_image,
            "sh",
            "-lc",
            self._config.sandbox_test_command,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return SandboxResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
