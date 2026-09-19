"""Prepara el entorno del proyecto y ejecuta la app, sin activar PowerShell."""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv_vision"


def environment_python(env=ENV):
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def prepare():
    if sys.version_info < (3, 10):
        raise RuntimeError("Necesitas Python 3.10 o superior; usa una version de 64 bits.")
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Instala Python de 64 bits para usar PyTorch.")
    python = environment_python()
    if not python.is_file():
        print("Creando el entorno de Python del proyecto...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    requirements = ROOT / "requirements.txt"
    stamp = ENV / "webcam_requirements.sha256"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    ready = stamp.is_file() and stamp.read_text(encoding="ascii").strip() == digest
    if ready:
        health = subprocess.run(
            [str(python), "-c", "import cv2, numpy, torch, ultralytics"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        ready = health.returncode == 0
    if not ready:
        print("Instalando dependencias. La primera vez puede tardar varios minutos...", flush=True)
        run([python, "-m", "pip", "install", "--upgrade", "pip"])
        if sys.platform != "darwin":
            # Descarga la version para CPU; esta app no necesita CUDA.
            run([python, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([python, "-m", "pip", "install", "--upgrade", "-r", requirements])
        run([python, "-m", "pip", "check"])
        run([python, ROOT / "PY1.PY", "--diagnostico"])
        stamp.write_text(digest + "\n", encoding="ascii")
    return python


def main():
    try:
        python = prepare()
        result = subprocess.run([str(python), str(ROOT / "PY1.PY"), *sys.argv[1:]], cwd=ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\nOperacion cancelada. Puedes volver a ejecutar INICIAR.bat.")
        return 130
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"No se pudo iniciar: {error}", file=sys.stderr)
        print("Revisa el mensaje anterior. En Windows usa Python 3.10-3.14 de 64 bits.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
