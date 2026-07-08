import math

import cv2
import numpy as np

from .config import FONT, OSD_COLOR


def draw_artificial_horizon(canvas, roll_deg, pitch_deg, cx, cy, radius):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)

    full_cx, full_cy = canvas.shape[1] // 2, canvas.shape[0] // 2
    pixels_per_degree = 4
    pitch_shift_y = int(pitch_deg * pixels_per_degree)

    overlay = np.zeros_like(canvas)
    horizon_center_y = full_cy - pitch_shift_y
    cv2.line(overlay, (0, horizon_center_y), (canvas.shape[1], horizon_center_y), OSD_COLOR, 2)

    for p in range(10, 91, 10):
        line_y = full_cy - int(p * pixels_per_degree) - pitch_shift_y
        if line_y < 0:
            continue
        line_length = 60 if abs(p) % 20 == 0 else 30
        cv2.line(overlay, (full_cx - line_length, line_y), (full_cx + line_length, line_y), OSD_COLOR, 1)
        cv2.putText(overlay, str(p), (full_cx + line_length + 5, line_y + 5), FONT, 0.6, OSD_COLOR, 1)

    for p in range(-10, -91, -10):
        line_y = full_cy - int(p * pixels_per_degree) - pitch_shift_y
        if line_y > canvas.shape[0]:
            continue
        line_length = 60 if abs(p) % 20 == 0 else 30
        cv2.line(overlay, (full_cx - line_length, line_y), (full_cx + line_length, line_y), OSD_COLOR, 1)
        cv2.putText(overlay, str(p), (full_cx + line_length + 5, line_y + 5), FONT, 0.6, OSD_COLOR, 1)

    matrix = cv2.getRotationMatrix2D((full_cx, full_cy), -roll_deg, 1)
    rotated_overlay = cv2.warpAffine(overlay, matrix, (canvas.shape[1], canvas.shape[0]))
    canvas[:] = cv2.add(canvas, rotated_overlay)

    symbol_arm_length = 50
    symbol_gap = 10
    cv2.line(canvas, (full_cx - symbol_arm_length, full_cy), (full_cx - symbol_gap, full_cy), OSD_COLOR, 3)
    cv2.line(canvas, (full_cx + symbol_gap, full_cy), (full_cx + symbol_arm_length, full_cy), OSD_COLOR, 3)
    cv2.line(canvas, (full_cx, full_cy - symbol_gap), (full_cx, full_cy + symbol_gap), OSD_COLOR, 3)

    roll_indicator_y = 10
    roll_arrow_x = full_cx + int(math.sin(roll) * (canvas.shape[1] // 2 - 20))
    cv2.line(canvas, (roll_arrow_x, roll_indicator_y), (roll_arrow_x, roll_indicator_y + 10), OSD_COLOR, 2)


def draw_tape(canvas, value, x_pos, y_pos, width, height, is_vertical=True, color=(0, 255, 0), tick_range=50, step=10):
    center_y = y_pos + height // 2
    center_x = x_pos + width // 2
    alpha = 0.3
    beta = 1.0 - alpha

    y1, y2 = max(0, y_pos), min(canvas.shape[0], y_pos + height)
    x1, x2 = max(0, x_pos), min(canvas.shape[1], x_pos + width)

    if y1 < y2 and x1 < x2:
        roi_tape = canvas[y1:y2, x1:x2]
        black_overlay = np.zeros_like(roi_tape)
        blended_roi = cv2.addWeighted(roi_tape, beta, black_overlay, alpha, 0)
        canvas[y1:y2, x1:x2] = blended_roi

    cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + width, y_pos + height), color, 1)

    alpha_text = 0.5
    beta_text = 1.0 - alpha_text

    if is_vertical:
        y1_val, y2_val = center_y - 15, center_y + 15
        x1_val, x2_val = x_pos, x_pos + width + 20
        y1_val, y2_val = max(0, y1_val), min(canvas.shape[0], y2_val)
        x1_val, x2_val = max(0, x1_val), min(canvas.shape[1], x2_val)

        if y1_val < y2_val and x1_val < x2_val:
            roi_val = canvas[y1_val:y2_val, x1_val:x2_val]
            black_overlay_val = np.zeros_like(roi_val)
            blended_val = cv2.addWeighted(roi_val, beta_text, black_overlay_val, alpha_text, 0)
            canvas[y1_val:y2_val, x1_val:x2_val] = blended_val

        cv2.putText(canvas, f"{int(value):>3}", (x_pos + 5, center_y + 10), FONT, 0.8, color, 2)
        pixels_per_unit = height / tick_range
        int_val = int(value)

        for i in range(int_val - tick_range, int_val + tick_range):
            if i % step == 0:
                y = center_y - int((i - value) * pixels_per_unit)
                if y > y_pos and y < y_pos + height:
                    cv2.line(canvas, (x_pos + width - 20, y), (x_pos + width, y), color, 2)
                    cv2.putText(canvas, str(i), (x_pos + 5, y + 5), FONT, 0.5, color, 1)
    else:
        y1_comp, y2_comp = y_pos - 30, y_pos
        x1_comp, x2_comp = center_x - 20, center_x + 20
        y1_comp, y2_comp = max(0, y1_comp), min(canvas.shape[0], y2_comp)
        x1_comp, x2_comp = max(0, x1_comp), min(canvas.shape[1], x2_comp)

        if y1_comp < y2_comp and x1_comp < x2_comp:
            roi_comp = canvas[y1_comp:y2_comp, x1_comp:x2_comp]
            black_overlay_comp = np.zeros_like(roi_comp)
            blended_comp = cv2.addWeighted(roi_comp, beta_text, black_overlay_comp, alpha_text, 0)
            canvas[y1_comp:y2_comp, x1_comp:x2_comp] = blended_comp

        cv2.putText(canvas, f"{int(value):03}", (center_x - 18, y_pos - 8), FONT, 0.8, color, 2)
        cv2.line(canvas, (center_x, y_pos), (center_x, y_pos + 10), color, 2)
        pixels_per_unit = width / tick_range
        int_val = int(value)

        for i in range(int_val - tick_range, int_val + tick_range):
            if i % 10 == 0:
                x = center_x - int((i - value) * pixels_per_unit)
                if x > x_pos and x < x_pos + width:
                    i_norm = i % 360
                    lbl = str(i_norm)
                    if i_norm == 0:
                        lbl = "N"
                    elif i_norm == 90:
                        lbl = "E"
                    elif i_norm == 180:
                        lbl = "S"
                    elif i_norm == 270:
                        lbl = "W"

                    cv2.line(canvas, (x, y_pos), (x, y_pos + 10), color, 2)
                    if i_norm % 30 == 0:
                        cv2.putText(canvas, lbl, (x - 10, y_pos + 30), FONT, 0.6, color, 1)


def draw_status_banner(canvas, text, color=OSD_COLOR):
    banner_width = min(canvas.shape[1] - 40, max(280, len(text) * 12 + 40))
    banner_height = 44
    x1 = (canvas.shape[1] - banner_width) // 2
    y1 = 20
    x2 = x1 + banner_width
    y2 = y1 + banner_height

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, text, (x1 + 18, y1 + 29), FONT, 0.8, color, 2)
