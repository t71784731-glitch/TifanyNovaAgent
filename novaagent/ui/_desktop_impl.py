"""Implementación stub de la interfaz gráfica."""
from __future__ import annotations
import os, queue, sys, threading, platform
from pathlib import Path
from typing import Any, Optional

try:
    from PySide6.QtCore import Qt, QObject, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QSpinBox, QDoubleSpinBox, QSlider, QListWidget, QTableWidget,
        QTableWidgetItem, QTabWidget, QStatusBar, QProgressBar,
        QFileDialog, QMessageBox, QHeaderView, QCheckBox, QFormLayout,
    )
    PYSIDE6_OK = True
except ImportError:
    PYSIDE6_OK = False

from novaagent.engine.llm_engine import LlamaEngine
from novaagent.engine.model_downloader import ModelDownloader, RECOMMENDED_MODELS

THEME_QSS = """
QMainWindow, QWidget { background-color: #1a1a2e; color: #e8e8e8; font-family: "Segoe UI"; }
QPushButton { background-color: #0f3460; color: #e8e8e8; padding: 8px 16px; border: 1px solid #2a2a4a; border-radius: 4px; }
QPushButton:hover { background-color: #00d4aa; color: #000; }
QPushButton#accent { background-color: #00d4aa; color: #000; }
QPushButton#danger { background-color: #e74c3c; color: #fff; }
QLineEdit, QTextEdit, QPlainTextEdit { background-color: #0d1b2a; color: #e8e8e8; border: 1px solid #2a2a4a; }
QTabBar::tab { background-color: #16213e; color: #a0a0b0; padding: 10px 20px; }
QTabBar::tab:selected { background-color: #0f3460; color: #00d4aa; }
QStatusBar { background-color: #16213e; color: #a0a0b0; }
QProgressBar::chunk { background-color: #00d4aa; }
"""

if PYSIDE6_OK:
    class NovaDesktopApp:
        def __init__(self, app_dir="", data_dir="", models_dir=""):
            self.app_dir = app_dir
            self.data_dir = data_dir
            self.models_dir = models_dir
            self.app = QApplication(sys.argv)
            self.app.setStyleSheet(THEME_QSS)
            self.engine = LlamaEngine()
            self.downloader = ModelDownloader()
            self.queue = queue.Queue()
            self.window = self._build_window()

        def _build_window(self):
            w = QMainWindow()
            w.setWindowTitle("NovaAgent")
            w.setMinimumSize(1200, 800)
            central = QWidget()
            w.setCentralWidget(central)
            layout = QVBoxLayout(central)
            tabs = QTabWidget()
            layout.addWidget(tabs)
            tabs.addTab(self._build_chat_tab(), "Chat")
            tabs.addTab(self._build_models_tab(), "Modelos")
            tabs.addTab(self._build_settings_tab(), "Ajustes")
            tabs.addTab(self._build_about_tab(), "Acerca de")
            sb = QStatusBar()
            w.setStatusBar(sb)
            self.status_label = QLabel("Listo")
            sb.addWidget(self.status_label)
            self.gpu_label = QLabel("GPU: --")
            sb.addPermanentWidget(self.gpu_label)
            hw = self.engine.detect_hardware()
            if hw.get("cuda_available"):
                g = hw["gpus"][0] if hw["gpus"] else {}
                self.gpu_label.setText(f"GPU: {g.get('name', '?')}")
            else:
                self.gpu_label.setText("GPU: CPU")
            return w

        def _build_chat_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            layout.addWidget(self.chat_display, 1)
            hl = QHBoxLayout()
            self.chat_input = QLineEdit()
            self.chat_input.setPlaceholderText("Escribe un mensaje...")
            self.chat_input.returnPressed.connect(self._send)
            hl.addWidget(self.chat_input, 1)
            send = QPushButton("Enviar")
            send.setObjectName("accent")
            send.clicked.connect(self._send)
            hl.addWidget(send)
            layout.addLayout(hl)
            return w

        def _send(self):
            msg = self.chat_input.text().strip()
            if not msg: return
            self.chat_display.append(f"Tú: {msg}")
            self.chat_input.clear()
            if not self.engine.is_loaded:
                self.chat_display.append("NovaAgent: Por favor, carga un modelo primero en la pestaña Modelos.")
                return
            def worker():
                try:
                    r = self.engine.generate(msg, max_tokens=256, temperature=0.7)
                    if r.error:
                        self.queue.put(("error", r.error))
                    else:
                        self.queue.put(("response", r.text))
                except Exception as e:
                    self.queue.put(("error", str(e)))
            threading.Thread(target=worker, daemon=True).start()
            self._poll()

        def _poll(self):
            try:
                while True:
                    kind, data = self.queue.get_nowait()
                    if kind == "response":
                        self.chat_display.append(f"NovaAgent: {data}")
                    elif kind == "error":
                        self.chat_display.append(f"Error: {data}")
                    elif kind == "download_progress":
                        self.progress_bar.setValue(int(data))
                    elif kind == "download_done":
                        self.status_label.setText(f"Descarga lista: {data}")
                        self._refresh_local()
            except queue.Empty: pass
            self.app.processEvents()
            if hasattr(self, "_polling") and self._polling:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, self._poll)

        def _build_models_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Modelos recomendados (descarga desde Internet)"))
            self.models_table = QTableWidget()
            self.models_table.setColumnCount(3)
            self.models_table.setHorizontalHeaderLabels(["Modelo", "Tamaño", "Descripción"])
            self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.models_table, 1)
            self._fill_models_table()
            hl = QHBoxLayout()
            dl = QPushButton("Descargar seleccionado")
            dl.setObjectName("accent")
            dl.clicked.connect(self._download_selected)
            hl.addWidget(dl)
            layout.addLayout(hl)
            self.progress_bar = QProgressBar()
            layout.addWidget(self.progress_bar)
            layout.addWidget(QLabel("Modelos descargados (locales)"))
            self.local_list = QListWidget()
            layout.addWidget(self.local_list)
            load = QPushButton("Cargar modelo seleccionado")
            load.clicked.connect(self._load_selected)
            layout.addWidget(load)
            self._refresh_local()
            return w

        def _fill_models_table(self):
            models = RECOMMENDED_MODELS
            self.models_table.setRowCount(len(models))
            for i, m in enumerate(models):
                self.models_table.setItem(i, 0, QTableWidgetItem(f"{m.parameters} {m.quantization}"))
                self.models_table.setItem(i, 1, QTableWidgetItem(f"{m.size_mb:.0f} MB"))
                self.models_table.setItem(i, 2, QTableWidgetItem(m.description))

        def _refresh_local(self):
            self.local_list.clear()
            for m in self.downloader.list_local_models(self.models_dir):
                self.local_list.addItem(f"{m['filename']} ({m['size_mb']:.0f} MB)")

        def _download_selected(self):
            row = self.models_table.currentRow()
            if row < 0:
                QMessageBox.information(None, "Info", "Selecciona un modelo")
                return
            model = RECOMMENDED_MODELS[row]
            self.status_label.setText(f"Descargando {model.filename}...")
            def cb(p):
                if p.total_bytes > 0:
                    pct = int(p.downloaded_bytes * 100 / p.total_bytes)
                    self.queue.put(("download_progress", pct))
            def worker():
                r = self.downloader.download(model, self.models_dir, progress_callback=cb)
                if r.status == "done":
                    self.queue.put(("download_done", r.file_path))
                else:
                    self.queue.put(("error", r.error or "Error"))
            self._polling = True
            self._poll()
            threading.Thread(target=worker, daemon=True).start()

        def _load_selected(self):
            item = self.local_list.currentItem()
            if not item:
                QMessageBox.information(None, "Info", "Selecciona un modelo")
                return
            filename = item.text().split()[0]
            path = Path(self.models_dir) / filename
            info = self.engine.load_model(str(path))
            if info.error:
                QMessageBox.critical(None, "Error", info.error)
            else:
                self.status_label.setText(f"Modelo cargado: {filename}")
                QMessageBox.information(None, "OK", f"Modelo cargado:\n{filename}")

        def _build_settings_tab(self):
            w = QWidget()
            layout = QFormLayout(w)
            self.ctx = QSpinBox(); self.ctx.setRange(512, 32768); self.ctx.setValue(4096)
            layout.addRow("Contexto máximo:", self.ctx)
            self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setValue(0.7)
            layout.addRow("Temperatura:", self.temp)
            self.offline = QCheckBox("Modo offline")
            layout.addRow(self.offline)
            return w

        def _build_about_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setPlainText(
                "NovaAgent v1.0\n\n"
                "Aplicación de escritorio con agentes de IA locales.\n\n"
                "Motor: llama.cpp (sin Ollama)\n"
                "Interfaz: PySide6\n\n"
                "Características:\n"
                "  • Inferencia 100% local\n"
                "  • Aceleración CUDA si hay GPU NVIDIA\n"
                "  • Funciona en CPU si no hay GPU\n"
                "  • Descarga de modelos desde Hugging Face\n"
                "  • Modo offline tras descargar el modelo\n"
            )
            layout.addWidget(txt)
            return w

        def run(self):
            self.window.show()
            return self.app.exec()
else:
    class NovaDesktopApp:
        def __init__(self, app_dir="", data_dir="", models_dir=""):
            self.app_dir = app_dir
            self.data_dir = data_dir
            self.models_dir = models_dir
        def run(self):
            print("ERROR: PySide6 no está instalado")
            print("Instala con: pip install PySide6")
            return 1
