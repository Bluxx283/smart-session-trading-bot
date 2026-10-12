"""SSAD's optional floating desktop AI copilot (local, opt-in prototype).

Run with: python desktop_copilot.py
Requires OPENAI_API_KEY in the environment to use vision analysis.
Captures only after the user checks consent and clicks Analyze. Images stay in
memory and are sent to the configured OpenAI API; they are not saved locally.
No broker orders are placed.
"""
from __future__ import annotations

import base64
import os
import sys
from datetime import datetime, timezone
from io import BytesIO

import mss
from PIL import Image
import requests
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)


SYSTEM_PROMPT = """You are SSAD's trading education copilot. Analyze only what is actually
visible in the supplied screenshot. First identify whether the image is a readable
market chart and state the visible instrument/timeframe only when legible. Discuss
visible structure, swing highs/lows, support/resistance and possible scenarios.
Never invent prices, candles, indicators or certainty. If unreadable or stale, say so.
If giving a hypothetical setup, label it EDUCATIONAL / UNCONFIRMED and include
directional bias, possible entry zone, invalidation/stop, targets and breakeven
condition only when the chart gives a defensible basis. State key uncertainty and
that this is not financial advice. Never claim a trade is guaranteed. Do not place
orders or tell the app to execute trades."""


class AnalysisWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, png: bytes, question: str):
        super().__init__()
        self.png = png
        self.question = question

    def run(self):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            self.failed.emit("Set OPENAI_API_KEY in your environment to enable AI chart analysis.")
            return
        try:
            encoded = base64.b64encode(self.png).decode("ascii")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": os.getenv("SSAD_VISION_MODEL", "gpt-4o-mini"),
                    "temperature": 0.2,
                    "max_tokens": 900,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": self.question or "Analyze this chart cautiously. Explain what is visible and any conditional setup."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"}},
                        ]},
                    ],
                },
                timeout=75,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            self.done.emit(answer if isinstance(answer, str) else str(answer))
        except requests.Timeout:
            self.failed.emit("The AI request timed out. Check your connection and try again.")
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("error", {}).get("message", str(exc))
            except Exception:
                detail = str(exc)
            self.failed.emit(f"AI provider error: {detail}")
        except Exception as exc:
            self.failed.emit(f"Analysis failed: {exc}")


class FloatingCopilot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SSAD • AI Companion")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.94)
        self.resize(370, 570)
        self._drag_offset = None
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.panel = QWidget()
        self.panel.setObjectName("panel")
        layout = QVBoxLayout(self.panel)
        layout.setSpacing(9)

        top = QHBoxLayout()
        mascot = QLabel("🤖")
        mascot.setFont(QFont("Sans Serif", 23))
        top.addWidget(mascot)
        title_col = QVBoxLayout()
        title = QLabel("SSAD Companion")
        title.setObjectName("title")
        subtitle = QLabel("Your little market copilot")
        subtitle.setObjectName("muted")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        top.addLayout(title_col, 1)
        self.opacity_btn = QPushButton("◐")
        self.opacity_btn.setToolTip("Toggle translucency")
        self.opacity_btn.clicked.connect(self.toggle_opacity)
        top.addWidget(self.opacity_btn)
        close_btn = QPushButton("×")
        close_btn.clicked.connect(self.close)
        top.addWidget(close_btn)
        layout.addLayout(top)

        layout.addWidget(QLabel("Choose the display that contains your chart"))
        self.display = QComboBox()
        with mss.mss() as sct:
            self.monitors = list(sct.monitors[1:])
        for i, monitor in enumerate(self.monitors, 1):
            self.display.addItem(f"Display {i} • {monitor['width']} × {monitor['height']}", i)
        if not self.monitors:
            self.display.addItem("No displays detected", 0)
        layout.addWidget(self.display)

        self.consent = QCheckBox("I permit one screenshot of this display for this analysis")
        layout.addWidget(self.consent)
        self.question = QLineEdit()
        self.question.setPlaceholderText("Ask about structure, risk, or a setup…")
        layout.addWidget(self.question)
        actions = QHBoxLayout()
        self.analyze_btn = QPushButton("🔎 Analyze screen")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self.analyze)
        actions.addWidget(self.analyze_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        actions.addWidget(self.clear_btn)
        layout.addLayout(actions)

        self.status = QLabel("Ready • No screen is captured until you consent and click Analyze.")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("I'll explain what I can actually see. No made-up prices or guaranteed signals.")
        layout.addWidget(self.output, 1)
        footer = QLabel("Advisory prototype · no order execution · screenshots are not saved")
        footer.setWordWrap(True)
        footer.setObjectName("muted")
        layout.addWidget(footer)
        root.addWidget(self.panel)

        self.setStyleSheet("""
            QWidget#panel { background: rgba(13, 22, 38, 238); border: 1px solid #38516f;
                            border-radius: 18px; color: #eaf2ff; }
            QLabel { color: #eaf2ff; }
            QLabel#title { font-size: 16px; font-weight: 700; }
            QLabel#muted { color: #a9bad1; font-size: 11px; }
            QLineEdit, QTextEdit, QComboBox { background: rgba(25, 39, 59, 245); color: #eef5ff;
                border: 1px solid #354d6b; border-radius: 9px; padding: 8px; }
            QPushButton { background: #243650; color: #eaf2ff; border: 1px solid #415a79;
                border-radius: 9px; padding: 8px; }
            QPushButton#primary { background: #2678d5; border-color: #4c9cf2; font-weight: 600; }
            QPushButton:hover { border-color: #75b7ff; }
            QCheckBox { color: #cbd8e9; }
        """)

    def toggle_opacity(self):
        self.setWindowOpacity(0.82 if self.windowOpacity() > 0.9 else 0.96)

    def clear_chat(self):
        self.output.clear()

    def analyze(self):
        if not self.consent.isChecked():
            QMessageBox.information(self, "Permission required", "Check the consent box before capturing a screen.")
            return
        index = self.display.currentData()
        if not index:
            QMessageBox.warning(self, "No display", "No physical display is available.")
            return
        try:
            with mss.mss() as sct:
                if index >= len(sct.monitors):
                    raise RuntimeError("Selected display is no longer available.")
                shot = sct.grab(sct.monitors[index])
                image = Image.frombytes("RGB", shot.size, shot.rgb)
                buf = BytesIO()
                image.save(buf, format="PNG")
                png = buf.getvalue()
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception as exc:
            QMessageBox.warning(self, "Capture failed", str(exc))
            return

        question = self.question.text().strip()
        self.output.append(f"<b>You · {stamp}</b><br>Analyzing one consented screenshot…")
        self.status.setText("Analyzing screenshot. Please wait…")
        self.analyze_btn.setEnabled(False)
        self.worker = AnalysisWorker(png, question)
        self.worker.done.connect(self.analysis_done)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.start()

    def analysis_done(self, answer):
        self.output.append(f"<b>SSAD Companion</b><br>{answer.replace(chr(10), '<br>')}")
        self.status.setText("Analysis complete. Verify every detail against your live chart.")
        self.analyze_btn.setEnabled(True)

    def analysis_failed(self, error):
        self.output.append(f"<b>Could not analyze</b><br>{error}")
        self.status.setText("Analysis unavailable.")
        self.analyze_btn.setEnabled(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = FloatingCopilot()
    widget.show()
    sys.exit(app.exec())
