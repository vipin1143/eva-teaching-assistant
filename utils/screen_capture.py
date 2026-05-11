"""
Screen Capture Module
Captures teacher's Zoom/Meet window every 3 seconds.
Auto page-cycles through gallery pages to cover all visible students.
"""
import threading, time, base64, logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval = 3  # seconds between captures
        self._page_interval = 12  # seconds per gallery page before cycling
        self._callback: Optional[Callable] = None
        self._mss_ready = False
        self._pyautogui_ready = False
        self._init()

    def _init(self):
        try:
            import mss  # noqa
            self._mss_ready = True
            logger.info("✅ Screen capture (mss) ready")
        except ImportError:
            logger.warning("⚠️  mss not installed: pip install mss")
        try:
            import pyautogui  # noqa
            self._pyautogui_ready = True
            logger.info("✅ PyAutoGUI (page cycling) ready")
        except ImportError:
            logger.warning("⚠️  pyautogui not installed (page cycling disabled)")

    def start(self, session_id: int, callback: Optional[Callable] = None):
        """Start background capture thread"""
        if self._running:
            return
        self._session_id = session_id
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()
        logger.info("🎥 Screen capture started")

    def stop(self):
        self._running = False
        logger.info("🛑 Screen capture stopped")

    def capture_frame(self) -> Optional[str]:
        """Capture one frame of the screen and return as base64"""
        if not self._mss_ready:
            return None
        try:
            import mss, mss.tools
            from PIL import Image
            import io
            with mss.mss() as sct:
                # Capture primary monitor
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                # Resize to reduce bandwidth
                img = img.resize((1280, 720), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    def _capture_loop(self):
        """Background loop: capture + optional page cycling"""
        page_timer = 0
        while self._running:
            frame = self.capture_frame()
            if frame and self._callback:
                self._callback(frame, self._session_id)

            page_timer += self._interval
            if page_timer >= self._page_interval and self._pyautogui_ready:
                self._cycle_page()
                page_timer = 0

            time.sleep(self._interval)

    def _cycle_page(self):
        """
        Press Page Down in the Zoom/Meet window to show the next tile grid page.
        This allows detection of students not visible on the current page.
        """
        try:
            import pyautogui
            pyautogui.press("pagedown")
            logger.debug("📄 Cycled to next gallery page")
            time.sleep(0.5)  # Wait for page to load
        except Exception as e:
            logger.warning(f"Page cycle failed: {e}")