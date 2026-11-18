"""Implementación stub del descargador de modelos."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import shutil, threading, time, urllib.request, urllib.error
from pathlib import Path

@dataclass
class DownloadProgress:
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "idle"
    error: str = ""
    file_path: str = ""

@dataclass
class ModelInfo:
    repo_id: str = ""
    filename: str = ""
    url: str = ""
    size_bytes: int = 0
    size_mb: float = 0.0
    quantization: str = ""
    parameters: str = ""
    description: str = ""

RECOMMENDED_MODELS = [
    ModelInfo(
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        quantization="Q4_K_M", parameters="7B",
        description="Modelo general. Buen equilibrio calidad/tamaño.",
        size_mb=4700,
    ),
    ModelInfo(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        url="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quantization="Q4_K_M", parameters="7B",
        description="Especialista en código.",
        size_mb=4900,
    ),
    ModelInfo(
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        url="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        quantization="Q4_K_M", parameters="3B",
        description="Modelo compacto para hardware limitado.",
        size_mb=2200,
    ),
]

class ModelDownloader:
    def __init__(self):
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def get_recommended_models(self): return RECOMMENDED_MODELS

    def download(self, model, dest_dir, progress_callback=None):
        self._cancel.clear()
        dest = Path(dest_dir) / model.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        prog = DownloadProgress(total_bytes=model.size_bytes, status="downloading")
        try:
            if dest.exists() and (model.size_bytes == 0 or dest.stat().st_size == model.size_bytes):
                prog.status = "done"
                prog.file_path = str(dest)
                return prog
            start_byte = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "NovaAgent/1.0"}
            if start_byte > 0:
                headers["Range"] = f"bytes={start_byte}-"
            req = urllib.request.Request(model.url, headers=headers)
            start_time = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.getcode()
                mode = "ab" if start_byte > 0 and status == 206 else "wb"
                with open(part, mode) as f:
                    while not self._cancel.is_set():
                        chunk = resp.read(1024*256)
                        if not chunk: break
                        f.write(chunk)
                        prog.downloaded_bytes += len(chunk)
                        elapsed = time.perf_counter() - start_time
                        if elapsed > 0:
                            prog.speed_bps = prog.downloaded_bytes / elapsed
                        if progress_callback:
                            progress_callback(prog)
            shutil.move(str(part), str(dest))
            prog.status = "done"
            prog.file_path = str(dest)
        except Exception as e:
            prog.status = "error"
            prog.error = str(e)
        return prog

    def cancel(self): self._cancel.set()

    def list_local_models(self, models_dir):
        models = []
        path = Path(models_dir)
        if not path.exists(): return models
        for p in path.glob("*.gguf"):
            try:
                s = p.stat()
                models.append({
                    "filename": p.name, "path": str(p),
                    "size_bytes": s.st_size,
                    "size_mb": round(s.st_size / (1024*1024), 2),
                })
            except Exception: pass
        return sorted(models, key=lambda m: m["filename"])
