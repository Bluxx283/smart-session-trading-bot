"""Opt-in, one-shot local display capture for the SSAD prototype.

No background capture, image persistence, OCR, or trading actions are performed.
Run Streamlit locally on the same computer as the TradingView window.
"""
from datetime import datetime, timezone
from io import BytesIO


class ScreenCaptureError(RuntimeError):
    """A safe, user-displayable screen-capture error."""


def capture_screen_once(display_index: int = 1):
    """Capture one display frame and return (PNG bytes, UTC ISO timestamp).

    The caller must invoke this only after an explicit user action.
    """
    try:
        import mss
    except ImportError as exc:
        raise ScreenCaptureError(
            "Screen capture dependency is missing. Install the project requirements (including mss) and restart SSAD."
        ) from exc

    try:
        with mss.mss() as capture:
            # mss monitor 0 is the virtual combined desktop; 1..N are physical displays.
            if not isinstance(display_index, int) or display_index < 1:
                raise ScreenCaptureError("Choose a physical display numbered 1 or higher.")
            if display_index >= len(capture.monitors):
                available = max(0, len(capture.monitors) - 1)
                raise ScreenCaptureError(
                    f"Display {display_index} is not available. This computer reports {available} physical display(s)."
                )
            shot = capture.grab(capture.monitors[display_index])
            # Encode in memory; do not write the image to disk.
            from PIL import Image
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            return buffer.getvalue(), timestamp
    except ScreenCaptureError:
        raise
    except Exception as exc:
        raise ScreenCaptureError(
            "Could not capture that display. Check local display permissions and whether SSAD is running on your own computer."
        ) from exc
