import time
import os, gc, math

from time import ticks_ms
from media.sensor import *
from media.display import *
from media.media import *
from machine import UART, FPIOA
import cv_lite

#------------
#参数区
#------------

fpioa = FPIOA()
fpioa.set_function(3, FPIOA.UART1_TXD)
fpioa.set_function(4, FPIOA.UART1_RXD)

uart = UART(

    UART.UART1,
    baudrate = 115200,
    bits = UART.EIGHTBITS,
    parity = UART.PARITY_NONE,
    stop = UART.STOPBITS_ONE

)

DETECT_WIDTH = ALIGN_UP(800,16)
DETECT_HEIGHT = ALIGN_UP(480,16)

FRAME_WIDTH = 480
FRAME_HEIGHT = 320
IMAGE_SHAPE = [FRAME_HEIGHT, FRAME_WIDTH]


#canny边缘检测
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150

# 多边形拟合参数
# 多边形拟合精度（越小越精确）
APPROX_EPSILON = 0.04
# 最小面积比例（过滤噪点）
AREA_MIN_RATIO = 0.001
#最大角度余弦值（判断是否为直角）
MAX_ANGLE_COS = 0.5
# 高斯模糊核大小（去噪）
GAUSSIAN_BLUR_SIZE = 5

# 矩形有效性判断参数
# 面积阈值
AREA_MINI_THRESHOLD = 2000
AREA_MAX_THRESHOLD = 50000
# 最小边长
SIDE_MIN_LENGTH = 30
# 对边最大差值
LENGTH_DIFF_THRESHOLD = 120
# 角度容忍值
ANGLE_TOLERANCE = 10

#显示颜色:文字 好矩形 坏矩形
TEXT_COLOR = (255, 255, 0)
GOOD_COLOR = (0, 255, 0)
BAD_COLOR = (255, 0, 0)
POINT_COLOR = (0, 0, 255)

#全局变量
sensor = None
last_valid_corners = None #缓存上一次的有效矩形角点 防抖

#--------
#工具函数
#--------

#格式化坐标
def format_coord(coord):
    return f"{int(coord):+04d}"

#串口发送
def send_offset(x, y):
    payload = "[" + format_coord(x) + format_coord(y) + "*]"
    uart.write(payload)

#失去目标
def send_lost():
    uart.write("[LOST*]")

#欧式距离
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

#计算夹角
def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

#计算两个角度的最小夹角差
def angle_diff_deg(a, b):
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

#判断两线段是否近似平行
def are_segments_parallel(theta1, theta2, tolerance=30):
    diff = angle_diff_deg(theta1, theta2)
    return diff <= tolerance or abs(diff - 180) <= tolerance #接近0°或者接近180°

#判断两线段是否近似垂直
def are_segments_vertical(theta1, theta2, tolerance=30):
    diff = angle_diff_deg(theta1, theta2)
    return abs(diff - 90) <= tolerance or abs(diff - 270) <= tolerance #接近90°或者接近270°

#向量法行列式求两线段交点
def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    #计算向量叉积
    def cross_product(a, b):
        return a[0] * b[1] - a[1] * b[0]

    ab = (x2 - x1, y2 - y1)
    cd = (x4 - x3, y4 - y3)
    ac = (x3 - x1, y3 - y1)

    deno = cross_product(ab, cd)
    if abs(deno) < 1e-6: #浮点数难以精确等于0，用一极小值代替
        return None  # 平行或重合

    t = cross_product(ac, cd) / deno #克莱姆法则

    return (x1 + t * ab[0], y1 + t * ab[1])

def rect_to_corners(rect):
    return [
        (int(rect[4]), int(rect[5])),
        (int(rect[6]), int(rect[7])),
        (int(rect[8]), int(rect[9])),
        (int(rect[10]), int(rect[11])),
    ]

def find_max_rect(rects):
    max_area = 0
    best_rect = None
    for rect in rects:
        area = rect[2] * rect[3]
        if area > max_area:
            max_area = area
            best_rect = rect
    return best_rect

def draw_rect_outline(img, corners, color):
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        img.draw_line(int(x1), int(y1), int(x2), int(y2), color=color, thickness=3)
        img.draw_circle(int(x1), int(y1), 2, color=POINT_COLOR, thickness=-1) #绘制角点

# 筛选矩形：面积 边长 平行垂直
def analyze_rect(rect):

    corners = rect_to_corners(rect)

    # 计算边长
    len1 = distance(corners[0], corners[1])
    len2 = distance(corners[2], corners[3])
    len3 = distance(corners[0], corners[3])
    len4 = distance(corners[1], corners[2])

    # 面积
    area = rect[2] * rect[3]

    #对边长
    err1 = abs(len1 - len2)
    err2 = abs(len3 - len4)

    # 角度
    theta1 = angle_deg(corners[0], corners[1])
    theta2 = angle_deg(corners[2], corners[3])
    theta3 = angle_deg(corners[0], corners[3])
    theta4 = angle_deg(corners[1], corners[2])


    #分层筛选
    area_allow = area >= AREA_MINI_THRESHOLD and area <= AREA_MAX_THRESHOLD

    length_allow = (
        len1 >= SIDE_MIN_LENGTH and
        len2 >= SIDE_MIN_LENGTH and
        len3 >= SIDE_MIN_LENGTH and
        len4 >= SIDE_MIN_LENGTH 
    )
    
    shape_allow = err1 <= LENGTH_DIFF_THRESHOLD and err2 <= LENGTH_DIFF_THRESHOLD

    parallel_allow = (
        are_segments_parallel(theta1, theta2, ANGLE_TOLERANCE) and
        are_segments_parallel(theta3, theta4, ANGLE_TOLERANCE)
    )

    vertical_allow = are_segments_vertical(theta1, theta3, ANGLE_TOLERANCE) and \
                     are_segments_vertical(theta2, theta4, ANGLE_TOLERANCE)
    
    valid = area_allow and length_allow and shape_allow and parallel_allow and vertical_allow
    
    return {
        "corners": corners,
        "area": int(area),
        "len1": len1,
        "len2": len2,
        "len3": len3,
        "len4": len4,
        "err1": err1,
        "err2": err2,
        "valid": valid,
    }


#------
#初始化
#------
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


#------
#主循环
#------

def capture_picture():
    global last_valid_corners

    fps = time.clock()
    frame_count = 0

    while True:
        fps.tick()
        frame_count += 1

        rect_flag = 0 #0无1有

        best_area = 0
        err1 = 0
        err2 = 0
        center = None


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
                GAUSSIAN_BLUR_SIZE

            )

            best = find_max_rect(rects)

            if best is not None:
                info = analyze_rect(best)
                candidate_corners = info["corners"]
                best_area = info["area"]
                err1 = info["err1"]
                err2 = info["err2"]

                #建议先画成红色 避免闪烁
                draw_rect_outline(img, candidate_corners, BAD_COLOR)

                if info["valid"]:
                    rect_flag = 1
                    last_valid_corners = candidate_corners

            if rect_flag == 1 and last_valid_corners is not None:
                draw_rect_outline(img, last_valid_corners, GOOD_COLOR)               
                center = find_intersection(
                    last_valid_corners[0][0], last_valid_corners[0][1],
                    last_valid_corners[2][0], last_valid_corners[2][1],
                    last_valid_corners[1][0], last_valid_corners[1][1],
                    last_valid_corners[3][0], last_valid_corners[3][1],
                )

                if center is not None:
                    cx = int(center[0])
                    cy = int(center[1])
                    dx = FRAME_WIDTH // 2 - cx
                    dy = FRAME_HEIGHT // 2 - cy

                    img.draw_cross(cx, cy, color=BAD_COLOR, thickness=3)
                    
                    send_offset(dx, dy)

                    img.draw_string_advanced(
                        10, 89, 24,
                        "center: (%d, %d)" % (cx, cy),
                        color=TEXT_COLOR
                    )
                else:
                    send_lost()

            else:
                send_lost()

            img.draw_string_advanced(10, 5, 24, "fps=%d" % int(fps.fps()), color=TEXT_COLOR)
            img.draw_string_advanced(10, 33, 24, "rects=%d" % len(rects), color=TEXT_COLOR)
            img.draw_string_advanced(10, 61, 24, "area=%d flag=%d" % (best_area, rect_flag), color=TEXT_COLOR)
            img.draw_string_advanced(10, 117, 24, "err1=%.1f err2=%.1f" % (err1, err2), color=TEXT_COLOR)

            Display.show_image(img)

            #每20帧回收一次
            if frame_count % 20 == 0:
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
        print("camera init...")
        camera_init()
        camera_is_init = True

        print("capture picture...")
        capture_picture()
    finally:
        if camera_is_init:
            print("camera deinit...")
            camera_deinit()

if __name__ == "__main__":
    main()
