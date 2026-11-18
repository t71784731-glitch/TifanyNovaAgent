"""Implementación stub del motor de inferencia."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import platform, shutil, subprocess, threading, time
from pathlib import Path

@dataclass
class EngineInfo:
    backend: str = "stub"
    llama_cpp_version: str = ""
    cuda_available: bool = False
    cuda_devices: List[str] = field(default_factory=list)
    gpu_layers_used: int = 0
    model_path: str = ""
    model_size_mb: float = 0.0
    context_size: int = 0
    load_time_seconds: float = 0.0
    error: str = ""

@dataclass
class GenerationResult:
    text: str = ""
    tokens_prompt: int = 0
    tokens_generated: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0
    error: str = ""

class LlamaEngine:
    def __init__(self):
        self._llama = None
        self._loaded = False
        self._info = EngineInfo()
        self._lock = threading.RLock()

    def detect_hardware(self):
        info = {"cuda_available": False, "gpus": [], "gpu_count": 0}
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                proc = subprocess.run(
                    [smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    info["cuda_available"] = True
                    for line in proc.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            info["gpus"].append({
                                "name": parts[0],
                                "vram_total_mib": float(parts[1]),
                                "vram_free_mib": float(parts[2]),
                                "driver": parts[3] if len(parts) >= 4 else "",
                            })
                    info["gpu_count"] = len(info["gpus"])
            except Exception: pass
        return info

    def check_availability(self):
        try:
            import llama_cpp
            return {"available": True, "version": getattr(llama_cpp, "__version__", "?")}
        except ImportError:
            return {"available": False, "version": ""}

    def load_model(self, model_path, n_ctx=4096, n_gpu_layers=None):
        with self._lock:
            path = Path(model_path)
            self._info = EngineInfo(model_path=model_path)
            start = time.perf_counter()
            if not path.exists():
                self._info.error = f"Archivo no existe: {model_path}"
                return self._info
            self._info.model_size_mb = path.stat().st_size / (1024*1024)
            hw = self.detect_hardware()
            self._info.cuda_available = hw.get("cuda_available", False)
            if n_gpu_layers is None:
                n_gpu_layers = 0 if not self._info.cuda_available else 999
            self._info.gpu_layers_used = n_gpu_layers
            try:
                from llama_cpp import Llama
                self._llama = Llama(
                    model_path=str(path), n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers, verbose=False
                )
                self._loaded = True
                self._info.backend = "llama_cpp_cuda" if n_gpu_layers > 0 and self._info.cuda_available else "llama_cpp"
                self._info.context_size = n_ctx
                self._info.load_time_seconds = time.perf_counter() - start
            except Exception as e:
                self._info.error = str(e)
            return self._info

    def unload(self):
        with self._lock:
            if self._llama:
                del self._llama
                self._llama = None
            self._loaded = False

    @property
    def is_loaded(self): return self._loaded

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        result = GenerationResult()
        if not self._loaded:
            result.error = "Modelo no cargado"
            return result
        start = time.perf_counter()
        try:
            output = self._llama(
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, echo=False
            )
            if isinstance(output, dict):
                choices = output.get("choices", [])
                result.text = choices[0].get("text", "").strip() if choices else ""
                usage = output.get("usage", {})
                result.tokens_generated = usage.get("completion_tokens", 0)
            result.elapsed_seconds = time.perf_counter() - start
            if result.elapsed_seconds > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.elapsed_seconds
        except Exception as e:
            result.error = str(e)
        return result
