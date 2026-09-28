from __future__ import annotations
import threading
import cv2
import numpy as np
MAX_PROBE = 10000
def open_camera(index: int):
    capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_PROBE)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_PROBE)
    return capture
class CameraStream:
    def __init__(self, index: int) -> None:
        self.capture = open_camera(index)
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = self.capture.isOpened()
        self._thread = None
        if self._running:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
    @property
    def opened(self) -> bool:
        return self.capture.isOpened()
    def _loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                self._running = False
                break
            with self._lock:
                self._frame = frame
                self._seq += 1
    def read(self):
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame, self._seq
    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()
def shape_frame(frame: np.ndarray, width: int, height: int,
                mode: str = "fit", mirror: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    if mode == "fill":
        scale = max(width / w, height / h)
        rw = max(1, int(w * scale + 0.5))
        rh = max(1, int(h * scale + 0.5))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        x0 = (rw - width) // 2
        y0 = (rh - height) // 2
        out = resized[y0:y0 + height, x0:x0 + width]
    else:
        scale = min(width / w, height / h)
        rw = max(1, int(w * scale + 0.5))
        rh = max(1, int(h * scale + 0.5))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if rw == width and rh == height:
            out = resized
        else:
            out = np.zeros((height, width, 3), dtype=frame.dtype)
            x0 = (width - rw) // 2
            y0 = (height - rh) // 2
            out[y0:y0 + rh, x0:x0 + rw] = resized
    if mirror:
        out = cv2.flip(out, 1)
    return np.ascontiguousarray(out)
