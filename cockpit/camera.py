import glob
import os
import time
from queue import Empty, Full

import cv2
import numpy as np

from .config import FONT


def resolve_camera_source(preferred_index, by_id_pattern):
    matches = sorted(glob.glob(by_id_pattern))
    if matches:
        return matches[0]
    return preferred_index


def open_camera(preferred_index, by_id_pattern):
    candidates = []

    for match in sorted(glob.glob(by_id_pattern)):
        candidates.append(match)
        real_path = os.path.realpath(match)
        if real_path != match:
            candidates.append(real_path)

    candidates.append(preferred_index)

    for source in candidates:
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture, source
            capture.release()

    return None, candidates[0] if candidates else preferred_index


def put_latest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)


def get_latest_or_last(q, last_value):
    latest = last_value
    while True:
        try:
            latest = q.get_nowait()
        except Empty:
            break
    return latest


def create_error_frame(shape, error_text, color=(50, 50, 50)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :] = color
    cv2.putText(frame, error_text, (20, shape[0] // 2), FONT, 0.8, (0, 0, 255), 2)
    return frame


def capture_worker(capture, output_queues, stop_event):
    while not stop_event.is_set():
        ret, frame = capture.read()
        if not ret:
            time.sleep(0.005)
            continue

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for q in output_queues:
            put_latest(q, frame)
