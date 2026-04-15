import time
import os
import gc
import math

from time import ticks_ms
from media.sensor import *
from media.display import *
from media.media import *
from machine import UART, FPIOA
import cv_lite


# ----------------------
# 1. 硬件与参数
# ----------------------
fpioa = FPIOA()
fpioa.set_function(3, FPIOA.UART1_TXD)
fpioa.set_function(4, FPIOA.UART1_RXD)

uart = UART(
    UART.UART1,
    baudrate=115200,
    bits=UART.EIGHTBITS,
    parity=UART.PARITY_NONE,
    stop=UART.STOPBITS_ONE,
)

DETECT_WIDTH = ALIGN_UP(800, 16)
DETECT_HEIGHT = 480

FRAME_WIDTH = 480
FRAME_HEIGHT = 320
IMAGE_SHAPE = [FRAME_HEIGHT, FRAME_WIDTH]

CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
APPROX_EPSILON = 0.04
AREA_MIN_RATIO = 0.001
MAX_ANGLE_COS = 0.5
GAUSSIAN_BLUR_SIZE = 5

AREA_THRESHOLD = 2000

TEXT_COLOR = (255, 255, 0)
GOOD_COLOR = (0, 255, 0)
BAD_COLOR = (255, 0, 0)

sensor = None


# ----------------------
# 2. 工具函数
# ----------------------
def format_coord(coord):
    return f"{int(coord):+04d}"


def send_offset(dx, dy):
    payload = "[" + format_coord(dx) + format_coord(dy) + "*]"
    uart.write(payload)


def send_lost():
    uart.write("(x=999,y=999)")


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    ab = (x2 - x1, y2 - y1)
    ac = (x3 - x1, y3 - y1)
    cd = (x4 - x3, y4 - y3)

    d = det(ab, cd)
    if abs(d) < 1e-6:
        return None

    t = det(ac, cd) / d
    return (x1 + t * ab[0], y1 + t * ab[1])


def rect_to_corners(rect):
    return [
        (rect[4], rect[5]),
        (rect[6], rect[7]),
        (rect[8], rect[9]),
        (rect[10], rect[11]),
    ]


def find_max_rect(rects):
    best = None
    best_area = 0
    for rect in rects:
        area = rect[2] * rect[3]
        if area > best_area:
            best = rect
            best_area = area
    return best


# ----------------------
# 3. 初始化与释放
# ----------------------
def camera_init():
    global sensor

    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    sensor.set_pixformat(Sensor.RGB888)

    Display.init(Display.ST7701, width=DETECT_WIDTH, height=DETECT_HEIGHT, fps=100, to_ide=True)
    MediaManager.init()
    sensor.run()


def camera_deinit():
    global sensor

    if sensor is not None:
        sensor.stop()

    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()


# ----------------------
# 4. 主循环
# ----------------------
def capture_picture():
    fps = time.clock()

    while True:
        fps.tick()

        try:
            os.exitpoint()

            img = sensor.snapshot()
            img_np = img.to_numpy_ref()

            rects = cv_lite.rgb888_find_rectangles_with_corners(
                IMAGE_SHAPE,
                img_np,
                CANNY_THRESH1,
                CANNY_THRESH2,
                APPROX_EPSILON,
                AREA_MIN_RATIO,
                MAX_ANGLE_COS,
                GAUSSIAN_BLUR_SIZE,
            )

            best = find_max_rect(rects)

            if best is not None:
                corners = rect_to_corners(best)

                for i in range(4):
                    x1, y1 = corners[i]
                    x2, y2 = corners[(i + 1) % 4]
                    img.draw_line(int(x1), int(y1), int(x2), int(y2), color=GOOD_COLOR, thickness=3)

                center = find_intersection(
                    corners[0][0], corners[0][1], corners[2][0], corners[2][1],
                    corners[1][0], corners[1][1], corners[3][0], corners[3][1],
                )

                if center is not None:
                    cx = int(center[0])
                    cy = int(center[1])

                    img.draw_cross(cx, cy, color=BAD_COLOR, thickness=3)

                    dx = FRAME_WIDTH // 2 - cx
                    dy = FRAME_HEIGHT // 2 - cy
                    send_offset(dx, dy)
                else:
                    send_lost()
            else:
                send_lost()

            img.draw_string_advanced(0, 0, 24, "fps=%d" % int(fps.fps()), color=TEXT_COLOR)
            img.draw_string_advanced(0, 28, 24, "rects=%d" % len(rects), color=TEXT_COLOR)

            Display.show_image(img)

            gc.collect()

        except KeyboardInterrupt as e:
            print("user stop:", e)
            break
        except BaseException as e:
            print("Exception", e)
            break


def main():
    os.exitpoint(os.EXITPOINT_ENABLE)

    camera_is_init = False
    try:
        print("camera init")
        camera_init()
        camera_is_init = True

        print("camera capture")
        capture_picture()
    finally:
        if camera_is_init:
            print("camera deinit")
            camera_deinit()


if __name__ == "__main__":
    main()
