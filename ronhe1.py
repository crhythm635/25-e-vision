# Find Rects Example
# 这个文件最初是“查找矩形”的示例程序。
#
# Vision program for the 2025 NUEDC E target board.
# 这是一份用于 2025 年电赛 E 题靶板的视觉程序。
# This version keeps the original UART offset protocol,
# 这个版本保留了原来的串口偏移量通信协议，
# and upgrades the detection pipeline with:
# 同时把目标检测流程升级成了下面这些能力：
# - grayscale rectangle detection
# - 使用灰度图进行矩形检测
# - candidate scoring instead of single max-area pick
# - 不再只选“面积最大”的矩形，而是对候选目标打分后再选择
# - ordered corners + perspective center projection
# - 对角点进行排序，并用透视投影估计目标中心
# - center smoothing and short-term target hold
# - 对中心点做平滑，并在短时间丢失目标时继续保持输出
# - cleaner runtime structure for match use
# - 运行时结构更清晰，更适合比赛现场使用

# 导入垃圾回收模块，用于定期回收内存，避免长时间运行后内存不足。
import gc
# 导入数学模块，后面会用到三角函数、平方根等数学运算。
import math
# 导入操作系统接口模块，这里主要用到退出点控制等功能。
import os
# 导入时间模块，用于延时、计时、消抖等。
import time
# 从 time 模块中单独导入 ticks_ms，用来获取毫秒级时间戳。
from time import ticks_ms

# 导入轻量级计算机视觉库，用来进行灰度矩形检测。
import cv_lite
# 从 machine 模块导入 FPIOA，用于配置芯片引脚复用功能。
from machine import FPIOA
# 从 machine 模块导入 UART，用于串口通信。
from machine import UART
# 从 machine 模块导入 TOUCH，用于读取触摸屏坐标和触摸事件。
from machine import TOUCH
# 导入显示相关接口，后面会把摄像头画面显示到屏幕上。
from media.display import *
# 导入媒体管理接口，用于初始化显示和摄像头相关资源。
from media.media import *
# 导入传感器接口，这里主要用于创建和配置摄像头对象。
from media.sensor import *


# 创建 FPIOA 对象，用于配置引脚功能映射。
fpioa = FPIOA()
# 把物理引脚 3 映射为 UART1 的发送引脚 TXD。
fpioa.set_function(3,FPIOA.UART1_TXD)
# 把物理引脚 4 映射为 UART1 的接收引脚 RXD。
fpioa.set_function(4,FPIOA.UART1_RXD)
# 创建 UART 串口对象。
uart = UART(
    # 使用 UART1 这个串口外设。
    UART.UART1,
    # 串口波特率设为 115200。
    baudrate=115200,
    # 数据位为 8 位。
    bits=UART.EIGHTBITS,
    # 不使用奇偶校验。
    parity=UART.PARITY_NONE,
    # 停止位为 1 位。
    stop=UART.STOPBITS_ONE,
)

# 显示输出宽度按 16 对齐，很多底层图像接口有这种要求。
DETECT_WIDTH = ALIGN_UP(800, 16)
# 显示输出高度设置为 480。
DETECT_HEIGHT = 480
#3.5寸mipi屏分辨率定义
lcd_width = 800
lcd_height = 480

# 摄像头采集图像的宽度设为 480。
FRAME_WIDTH = 480
# 摄像头采集图像的高度设为 320。
FRAME_HEIGHT = 320
# 按 [高, 宽] 的格式保存图像形状，供 cv_lite 检测函数使用。
IMAGE_SHAPE = [FRAME_HEIGHT, FRAME_WIDTH]

# Camera orientation. Change these if the preview is mirrored or upside-down.
# 下面两个参数用于控制摄像头画面方向。
# 如果预览画面左右镜像了，或者上下颠倒了，可以改这两个值。
# 是否开启水平镜像，False 表示不镜像。
USE_HMIRROR = False
# 是否开启垂直翻转，False 表示不翻转。
USE_VFLIP = False

# Physical target size. The paper is A4.
# 目标纸张的真实物理尺寸，题目中靶纸按 A4 纸来处理。
# A4 纸短边长度，单位是毫米。
TARGET_SHORT_MM = 210.0
# A4 纸长边长度，单位是毫米。
TARGET_LONG_MM = 297.0
# 长边和短边的比例，后面会用来判断候选矩形的长宽比是否合理。
TARGET_LONG_SHORT_RATIO = TARGET_LONG_MM / TARGET_SHORT_MM

# cv_lite rectangle detection
# 下面这些是 cv_lite 做矩形检测时需要的参数。
# Canny 边缘检测的低阈值。
CANNY_THRESH1 = 50
# Canny 边缘检测的高阈值。
CANNY_THRESH2 = 150
# 多边形逼近精度参数，值越小越贴近原始轮廓。
APPROX_EPSILON = 0.04
# 候选区域最小面积占整幅图像面积的比例，太小的区域会被忽略。
AREA_MIN_RATIO = 0.0012
# 判断四边形是否接近直角时允许的最大角余弦值。
MAX_ANGLE_COS = 0.5
# 高斯模糊核大小，用于预处理降噪。
GAUSSIAN_BLUR_SIZE = 5

# Runtime tuning
# 下面这些参数是程序运行时可以调节或影响追踪表现的参数。
# 初始面积阈值，小于这个面积的矩形直接不要。
AREA_THRESHOLD_INIT = 2200
# 面积阈值允许调到的最小值。
AREA_THRESHOLD_MIN = 600
# 面积阈值允许调到的最大值。
AREA_THRESHOLD_MAX = 12000
# 每次调参时，面积阈值增减的步长。
AREA_THRESHOLD_STEP = 200
# 调参消抖时间，单位毫秒。
KEY_DEBOUNCE_MS = 180

# 触摸调参相关参数。当前 480x320 图像是居中显示在 800x480 屏幕上的。
# 触摸屏设备编号，当前使用 0 号触摸设备。
TOUCH_DEVICE_INDEX = 0
# 触摸坐标旋转参数，当前固件主要走默认方向，这里先保留配置位。
TOUCH_ROTATION = 0
# 只有触摸点位于屏幕 y=240 以下时，才认为是在调参区。
TOUCH_UI_MIN_Y = 240
# 左侧调参区的最大 x 坐标，小于这个值表示“减小阈值”。
TOUCH_LEFT_MAX_X = 160
# 右侧调参区的最小 x 坐标，大于等于这个值表示“增大阈值”。
TOUCH_RIGHT_MIN_X = 640
# 计算 480 宽图像居中显示在 800 宽屏幕上时的 x 偏移量。
IMAGE_OFFSET_X = (lcd_width - FRAME_WIDTH) // 2
# 计算 320 高图像居中显示在 480 高屏幕上时的 y 偏移量。
IMAGE_OFFSET_Y = (lcd_height - FRAME_HEIGHT) // 2

# 平滑系数 alpha，越大越偏向新值，越小越平滑。
SMOOTH_ALPHA = 0.65
# 目标暂时丢失后，最多继续保持多少帧的历史中心点。
MAX_HOLD_FRAMES = 4
# 每隔多少帧打印一次调试信息。
DEBUG_PRINT_EVERY = 30

# 文字绘制颜色，RGB = 黄色。
TEXT_COLOR = (255, 255, 0)
# 检测到有效矩形时，绘制轮廓的颜色，RGB = 绿色。
GOOD_COLOR = (0, 255, 0)
# 中心十字和连线使用的颜色，RGB = 红色。
BAD_COLOR = (255, 0, 0)
# 顶点小圆点使用的颜色，RGB = 青色。
POINT_COLOR = (0, 255, 255)

# 先把 sensor 置为 None，后面初始化摄像头时再真正赋值。
sensor = None
# 触摸设备句柄，在 camera_init() 里初始化。
# 先把触摸设备对象设为 None，后面初始化成功后再赋值。
touch = None


# 限幅函数：把 value 限制在 low 和 high 之间。
def clamp(value, low, high):
    # 如果值小于下限，就直接返回下限。
    if value < low:
        return low
    # 如果值大于上限，就直接返回上限。
    if value > high:
        return high
    # 如果值在范围内，就原样返回。
    return value

# 按给定步长调整面积阈值，并把结果限制在合法范围内。
def apply_area_threshold_delta(area_threshold, delta):
    # 先把当前面积阈值和本次希望增减的步长相加。
    new_threshold = area_threshold + delta
    # 再把结果限制在最小值和最大值之间，避免调参越界。
    return clamp(new_threshold, AREA_THRESHOLD_MIN, AREA_THRESHOLD_MAX)

# 根据触摸点所在的屏幕区域，返回本次应当增减的面积阈值步长。
def get_touch_area_threshold_delta(screen_x, screen_y):
    # 如果触摸点还在屏幕上半部分，说明不属于触摸调参区。
    if screen_y < TOUCH_UI_MIN_Y:
        # 返回 0 表示这次触摸不调整面积阈值。
        return 0
    # 如果触摸点落在左边黑边区域，就执行“减小阈值”。
    if screen_x < TOUCH_LEFT_MAX_X:
        # 返回一个负步长，供主循环把阈值往下调。
        return -AREA_THRESHOLD_STEP
    # 如果触摸点落在右边黑边区域，就执行“增大阈值”。
    if screen_x >= TOUCH_RIGHT_MIN_X:
        # 返回一个正步长，供主循环把阈值往上调。
        return AREA_THRESHOLD_STEP
    # 如果触摸点落在中间图像区域，就不触发这组调参逻辑。
    return 0

# 读取当前触摸点坐标；如果当前没有触摸或触摸不可用，则返回 None。
def read_touch_point():
    # 如果触摸设备还没有初始化成功，就直接返回空结果。
    if touch is None:
        # 返回 None，表示当前没有可用的触摸点。
        return None
    # 尝试从触摸设备里读取 1 个触摸点，调参场景只需要单点即可。
    try:
        # read(1) 表示最多读取 1 个触摸点。
        points = touch.read(1)
    # 如果读取过程中报错，就把它当作“当前无触摸”处理。
    except Exception:
        # 返回 None，避免异常影响主循环继续运行。
        return None
    # 如果返回的是空元组或空列表，说明当前没人按屏幕。
    if not points:
        # 返回 None，交给上层按“未触摸”处理。
        return None
    # 取出本次读取到的第一个触摸点。
    point = points[0]
    # 把触摸点坐标转成整数后返回，便于后续做区域判断。
    return (int(point.x), int(point.y))

# 用兼容方式初始化触摸设备，适配不同固件对 TOUCH 构造函数的支持差异。
def init_touch_device():
    # 先尝试使用“设备号 + rotation 关键字”的方式初始化触摸设备。
    try:
        # 按带 rotation 参数的写法创建触摸设备对象。
        device = TOUCH(TOUCH_DEVICE_INDEX, rotation=TOUCH_ROTATION)
        # 返回创建好的设备对象，并附带当前采用的初始化方式说明。
        return device, "rotation关键字参数"
    # 如果这里抛出 TypeError，通常表示当前固件不支持这个参数写法。
    except TypeError:
        # 不立刻报错，而是继续尝试更兼容的初始化方式。
        pass

    # 再尝试只传设备号的写法，这种写法和你当前板子的示例更接近。
    try:
        # 按最基础的 TOUCH(设备号) 方式创建设备对象。
        device = TOUCH(TOUCH_DEVICE_INDEX)
        # 返回创建好的设备对象，并说明当前采用的是默认旋转方式。
        return device, "默认旋转"
    # 如果这种方式也失败，就把异常继续抛给上层统一打印。
    except Exception:
        # 原样抛出异常，方便串口里看到真实的失败原因。
        raise


# 把坐标格式化成固定长度的带符号字符串，方便串口发送。
def format_coord(coord):
    # 先把坐标转成整数，再限制在 -999 到 999 之间，避免超范围。
    coord = clamp(int(coord), -999, 999)
    # 例如会格式化成 +012、-045 这样的 4 位有符号字符串。
    return f"{coord:+04d}"


# 发送“正在跟踪目标”的偏移量数据。
def send_tracking(dx, dy):
    # 按原协议拼接字符串，例如 [+012-008*]。
    payload = "[" + format_coord(dx) + format_coord(dy) + "*]"
    # 通过串口发给下位机。
    uart.write(payload)


# 发送“目标丢失”的标志数据。
def send_lost():
    # Keep the original "lost target" protocol for compatibility
    # 保留原来的“目标丢失”协议格式，
    # with the lower controller already used in your system.
    # 这样就能兼容你当前已经在使用的下位机程序。
    uart.write("(x=999,y=999)")


# 计算两点之间的欧氏距离。
def calculate_distance(p1, p2):
    # 使用二维平面距离公式 sqrt((x2-x1)^2 + (y2-y1)^2)。
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 使用多边形面积公式，计算一个顶点序列围成的面积。
def polygon_area(points):
    # 先把面积累加器初始化为 0。
    area = 0.0
    # 依次遍历每一个顶点。
    for i in range(len(points)):
        # 取当前点坐标。
        x1, y1 = points[i]
        # 取下一个点坐标，使用取模让最后一个点自动连回第一个点。
        x2, y2 = points[(i + 1) % len(points)]
        # 累加叉积项，这是鞋带公式的一部分。
        area += x1 * y2 - x2 * y1
    # 返回绝对值的一半，就是多边形面积。
    return abs(area) * 0.5


# 计算一组点的几何中心，也就是平均坐标。
def calculate_center(points):
    # 如果点列表为空，就返回原点，防止后面报错。
    if not points:
        return (0.0, 0.0)
    # x 坐标和初始化为 0。
    sum_x = 0.0
    # y 坐标和初始化为 0。
    sum_y = 0.0
    # 遍历所有点。
    for x, y in points:
        # 累加 x。
        sum_x += x
        # 累加 y。
        sum_y += y
    # 记录点的总个数。
    count = len(points)
    # 返回平均后的中心坐标。
    return (sum_x / count, sum_y / count)


# 把四个角点按顺时针或逆时针顺序排好，方便后续统一处理。
def sort_corners(corners):
    # 先计算四个角点的中心点。
    center = calculate_center(corners)
    # 按每个点相对中心点的极角进行排序。
    ordered = sorted(corners, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    # 找到左上角点，简单做法是选 x+y 最小的点。
    left_top = min(ordered, key=lambda p: p[0] + p[1])
    # 找到左上角在排序列表中的下标。
    index = ordered.index(left_top)
    # 让结果从左上角开始，后面的点保持原先环绕顺序。
    return ordered[index:] + ordered[:index]


# 计算两条直线的交点。
def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # 定义一个小工具函数，用来计算二维向量行列式。
    def determinant(a, b):
        # 行列式 a.x*b.y - a.y*b.x，常用于判断平行和求交点。
        return a[0] * b[1] - a[1] * b[0]

    # 直线 1 的方向向量 AB。
    ab = (x2 - x1, y2 - y1)
    # 从 A 指向 C 的向量 AC。
    ac = (x3 - x1, y3 - y1)
    # 直线 2 的方向向量 CD。
    cd = (x4 - x3, y4 - y3)
    # 计算 AB 和 CD 的行列式，若接近 0 说明两条线几乎平行。
    det = determinant(ab, cd)
    # 如果两条线几乎平行，就认为没有稳定交点。
    if abs(det) < 1e-6:
        return None
    # 根据直线交点公式求出参数 t。
    t = determinant(ac, cd) / det
    # 返回交点坐标。
    return (x1 + t * ab[0], y1 + t * ab[1])


# 根据 4 对对应点，求出透视变换矩阵。
def get_perspective_matrix(src_pts, dst_pts):
    # 系数矩阵 A，后面会构造 8x8 线性方程组。
    a = []
    # 常数项向量 b。
    b = []
    # 遍历 4 组点对应关系。
    for i in range(4):
        # 取源坐标系中的点 (x, y)。
        x, y = src_pts[i]
        # 取目标坐标系中的点 (u, v)。
        u, v = dst_pts[i]
        # 添加关于 u 的方程。
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        # 添加关于 v 的方程。
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        # 添加常数项 u。
        b.append(u)
        # 添加常数项 v。
        b.append(v)

    # 未知数一共有 8 个。
    n = 8
    # 使用高斯消元法求解线性方程组。
    for i in range(n):
        # 假设当前主元所在行是 i。
        max_row = i
        # 在当前列 i 中寻找绝对值最大的主元，提高数值稳定性。
        for j in range(i, len(a)):
            if abs(a[j][i]) > abs(a[max_row][i]):
                max_row = j
        # 交换当前行和主元行。
        a[i], a[max_row] = a[max_row], a[i]
        # 常数向量对应元素也要一起交换。
        b[i], b[max_row] = b[max_row], b[i]

        # 取当前主元。
        pivot = a[i][i]
        # 如果主元过小，说明矩阵可能退化，无法稳定求解。
        if abs(pivot) < 1e-8:
            return None

        # 把当前行从第 i 列开始都除以主元，让主元变成 1。
        for j in range(i, n):
            a[i][j] /= pivot
        # 常数项也同步归一化。
        b[i] /= pivot

        # 消去其它行在第 i 列上的值。
        for j in range(len(a)):
            # 跳过当前主元所在行。
            if j != i and a[j][i] != 0:
                # 记录要消掉的倍数。
                factor = a[j][i]
                # 对这一行做线性消元。
                for k in range(i, n):
                    a[j][k] -= factor * a[i][k]
                # 常数项也同步更新。
                b[j] -= factor * b[i]

    # 按单应矩阵 H 的形式重组结果，最后一个元素固定为 1.0。
    return [
        [b[0], b[1], b[2]],
        [b[3], b[4], b[5]],
        [b[6], b[7], 1.0],
    ]


# 用透视变换矩阵把一组点映射到另一坐标系。
def transform_points(points, matrix):
    # 保存所有变换后的点。
    transformed = []
    # 逐点进行变换。
    for x, y in points:
        # 计算齐次坐标中的 x 分量。
        x_hom = x * matrix[0][0] + y * matrix[0][1] + matrix[0][2]
        # 计算齐次坐标中的 y 分量。
        y_hom = x * matrix[1][0] + y * matrix[1][1] + matrix[1][2]
        # 计算齐次坐标中的 w 分量。
        w_hom = x * matrix[2][0] + y * matrix[2][1] + matrix[2][2]
        # 只有 w 不接近 0 时，才能安全地做归一化。
        if abs(w_hom) > 1e-8:
            # 转回普通二维坐标后加入结果列表。
            transformed.append((x_hom / w_hom, y_hom / w_hom))
    # 返回所有成功变换后的点。
    return transformed


# 对中心点做指数平滑，减少抖动。
def smooth_point(last_point, new_point, alpha):
    # 如果上一次没有点，就直接返回当前新点。
    if last_point is None:
        return new_point
    # 否则按指数平滑公式进行更新。
    return (
        last_point[0] + alpha * (new_point[0] - last_point[0]),
        last_point[1] + alpha * (new_point[1] - last_point[1]),
    )


# 把 cv_lite 返回的矩形数据结构转换成更好处理的 4 个角点坐标。
def rect_from_cv_lite(rect):
    # cv_lite 返回结果中，第 4~11 位保存四个角点坐标，这里重新组织成 [(x1,y1), ...]。
    return [
        (rect[4], rect[5]),
        (rect[6], rect[7]),
        (rect[8], rect[9]),
        (rect[10], rect[11]),
    ]


# 评估单个候选矩形，判断它像不像我们要找的靶纸，并计算评分。
def evaluate_candidate(rect, last_center, area_threshold):
    # 先取出角点并排序，保证后续边长、对角线等计算顺序统一。
    corners = sort_corners(rect_from_cv_lite(rect))
    # 计算该四边形面积。
    area = polygon_area(corners)
    # 如果面积小于当前阈值，直接淘汰。
    if area < area_threshold:
        return None

    # 计算四条边的长度。
    edges = [calculate_distance(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    # 找出最短边。
    min_edge = min(edges)
    # 如果最短边太短，说明目标太小或噪声太多，直接淘汰。
    if min_edge < 18:
        return None

    # 上边和下边的平均长度。
    avg_top_bottom = (edges[0] + edges[2]) * 0.5
    # 左边和右边的平均长度。
    avg_left_right = (edges[1] + edges[3]) * 0.5
    # 取较长的一组边作为“长边”。
    long_edge = max(avg_top_bottom, avg_left_right)
    # 取较短的一组边作为“短边”，并保证至少为 1，防止除零。
    short_edge = max(min(avg_top_bottom, avg_left_right), 1.0)
    # 计算候选矩形的长宽比。
    aspect = long_edge / short_edge
    # 如果长宽比太离谱，就不是 A4 纸那种矩形，淘汰。
    if aspect < 1.0 or aspect > 2.3:
        return None

    # 求外接包围框左上角 x。
    bbox_x = min(p[0] for p in corners)
    # 求外接包围框左上角 y。
    bbox_y = min(p[1] for p in corners)
    # 求包围框宽度。
    bbox_w = max(p[0] for p in corners) - bbox_x
    # 求包围框高度。
    bbox_h = max(p[1] for p in corners) - bbox_y
    # 求包围框面积，并保证至少为 1，防止除零。
    bbox_area = max(bbox_w * bbox_h, 1.0)
    # 计算多边形面积占包围框面积的比例。
    fill_ratio = area / bbox_area
    # 如果填充率太低，说明形状过于扭曲或像噪声，淘汰。
    if fill_ratio < 0.45:
        return None

    # 用两条对角线求交点，作为更稳定的中心点估计。
    center = find_intersection(
        corners[0][0], corners[0][1],
        corners[2][0], corners[2][1],
        corners[1][0], corners[1][1],
        corners[3][0], corners[3][1],
    )
    # 如果对角线求交失败，就退化成四个角点的平均中心。
    if center is None:
        center = calculate_center(corners)

    # 计算候选框离画面边缘的最小边距。
    margin = min(
        bbox_x,
        bbox_y,
        FRAME_WIDTH - (bbox_x + bbox_w),
        FRAME_HEIGHT - (bbox_y + bbox_h),
    )
    # 连续性惩罚项初始化为 0。
    continuity_penalty = 0.0
    # 如果上一帧有中心点，就计算这次中心与上一帧中心的距离。
    if last_center is not None:
        continuity_penalty = calculate_distance(center, last_center)

    # 计算对边长度差，用于衡量这个四边形是否规整。
    opposite_penalty = abs(edges[0] - edges[2]) + abs(edges[1] - edges[3])
    # 长宽比偏离 A4 纸真实比例的程度，也作为惩罚项。
    aspect_penalty = abs(aspect - TARGET_LONG_SHORT_RATIO)
    # 如果矩形贴近图像边缘，通常不完整，所以给较大惩罚。
    edge_penalty = 120.0 if margin < 2 else (25.0 if margin < 8 else 0.0)

    # 综合各种因素计算总评分。
    # 面积越大通常越可信，所以给正分。
    # 长宽比越接近 A4、对边越相等、与上一帧越连续、离边缘越远，则分数越高。
    score = (
        area * 0.05
        - aspect_penalty * 140.0
        - opposite_penalty * 0.55
        - continuity_penalty * 2.0
        - edge_penalty
    )

    # 把后续会用到的数据一起打包返回。
    return {
        "corners": corners,
        "center": center,
        "area": area,
        "aspect": aspect,
        "score": score,
    }


# 在所有候选矩形中，选出得分最高的那个目标。
def find_best_target(rects, last_center, area_threshold):
    # 先假设当前还没有最佳目标。
    best = None
    # 把最佳分数初始化成一个很小的值。
    best_score = -1e12
    # 遍历所有检测到的矩形。
    for rect in rects:
        # 对当前矩形进行打分评估。
        candidate = evaluate_candidate(rect, last_center, area_threshold)
        # 如果评估结果为空，说明它被淘汰了，继续下一个。
        if candidate is None:
            continue
        # 如果当前候选分数更高，就更新最佳目标。
        if candidate["score"] > best_score:
            best = candidate
            best_score = candidate["score"]
    # 返回最终选中的最佳目标，如果一个都没有则返回 None。
    return best


# 结合透视关系，更准确地估算目标纸张中心在图像中的位置。
def project_target_center(corners):
    # 计算上边和下边长度的平均值，当作目标在图像中的“宽度”估计。
    width_avg = (calculate_distance(corners[0], corners[1]) + calculate_distance(corners[2], corners[3])) * 0.5
    # 计算左边和右边长度的平均值，当作目标在图像中的“高度”估计。
    height_avg = (calculate_distance(corners[1], corners[2]) + calculate_distance(corners[3], corners[0])) * 0.5

    # 如果图像里看起来宽大于高，认为当前角点顺序对应“横放”的 A4 纸。
    if width_avg >= height_avg:
        # 构造一个虚拟的 A4 纸矩形坐标系，单位是毫米。
        virtual_rect = [
            (0.0, 0.0),
            (TARGET_LONG_MM, 0.0),
            (TARGET_LONG_MM, TARGET_SHORT_MM),
            (0.0, TARGET_SHORT_MM),
        ]
        # 这个虚拟矩形的中心点就是纸张中心。
        virtual_center = (TARGET_LONG_MM * 0.5, TARGET_SHORT_MM * 0.5)
    # 否则认为当前目标更像“竖放”的 A4 纸。
    else:
        # 构造竖放时的虚拟矩形。
        virtual_rect = [
            (0.0, 0.0),
            (TARGET_SHORT_MM, 0.0),
            (TARGET_SHORT_MM, TARGET_LONG_MM),
            (0.0, TARGET_LONG_MM),
        ]
        # 计算竖放时的虚拟中心点。
        virtual_center = (TARGET_SHORT_MM * 0.5, TARGET_LONG_MM * 0.5)

    # 求出从虚拟纸张坐标到图像角点坐标的透视变换矩阵。
    matrix = get_perspective_matrix(virtual_rect, corners)
    # 如果矩阵求成功了，就尝试把虚拟中心投影到图像里。
    if matrix is not None:
        # 对中心点做透视变换。
        mapped = transform_points([virtual_center], matrix)
        # 如果变换结果有效，就直接返回这个投影中心。
        if mapped:
            return mapped[0]

    # 如果透视法失败了，就退化为使用对角线交点。
    fallback = find_intersection(
        corners[0][0], corners[0][1],
        corners[2][0], corners[2][1],
        corners[1][0], corners[1][1],
        corners[3][0], corners[3][1],
    )
    # 如果对角线交点有效，就返回它。
    if fallback is not None:
        return fallback
    # 最后再不行，就返回四个角点的平均中心。
    return calculate_center(corners)


# 尝试设置摄像头方向，如果当前驱动不支持也不要让程序崩掉。
def try_configure_orientation(device):
    # 尝试设置水平镜像。
    try:
        device.set_hmirror(USE_HMIRROR)
    # 如果当前摄像头驱动不支持这个接口，就忽略异常。
    except Exception:
        pass
    # 尝试设置垂直翻转。
    try:
        device.set_vflip(USE_VFLIP)
    # 同样忽略不支持时的异常。
    except Exception:
        pass


# 创建摄像头对象，优先尝试使用 id=2。
def create_sensor():
    # 某些板卡上摄像头可能需要指定 id=2。
    try:
        return Sensor(id=2)
    # 如果失败，就退回默认构造方式。
    except Exception:
        return Sensor()


# 初始化摄像头和显示模块。
def camera_init():
    # 声明这里要修改全局变量 sensor。
    # 声明这里除了要修改全局 sensor 外，还要修改全局 touch。
    global sensor, touch

    # 创建摄像头对象。
    sensor = create_sensor()
    # 复位摄像头，确保进入干净状态。
    sensor.reset()
    # 应用画面方向配置。
    try_configure_orientation(sensor)
    # 设置摄像头输出分辨率。
    sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    # 设置像素格式为 RGB888，便于后续显示和转灰度。
    sensor.set_pixformat(Sensor.RGB888)

    # 初始化显示屏。
    Display.init(
        # 使用 ST7701 这类显示驱动芯片。
        Display.ST7701,
        # 显示宽度。
        width=DETECT_WIDTH,
        # 显示高度。
        height=DETECT_HEIGHT,
        # 目标刷新率。
        fps=100,
        # 允许同时把画面输出到 IDE 预览。
        to_ide=True,
    )
    # 初始化媒体资源管理器。
    MediaManager.init()
    # 启动摄像头开始采集。
    sensor.run()

    # 触摸功能在运行时是可选的，初始化失败也不要影响视觉主流程继续运行。
    # 尝试初始化触摸屏设备，即使失败也不要影响视觉主流程运行。
    try:
        # 按兼容逻辑初始化触摸设备，并拿到本次采用的初始化方式说明。
        touch, touch_init_mode = init_touch_device()
        # 在串口打印触摸初始化成功信息，方便上板调试。
        print("touch init ok:", touch_init_mode)
    # 如果触摸初始化失败，就只关闭触摸功能，不让主程序退出。
    except Exception as e:
        # 把全局 touch 重新置空，表示当前没有可用的触摸设备。
        touch = None
        # 把失败原因打印到串口，方便继续排查。
        print("touch init failed:", e)


# 反初始化摄像头和显示模块，释放硬件资源。
def camera_deinit():
    # 声明这里要访问并可能修改全局变量 sensor。
    # 声明这里除了可能修改 sensor 外，也可能修改 touch。
    global sensor, touch
    # 如果摄像头已经创建成功，就先停止采集。
    if sensor is not None:
        sensor.stop()
    # 如果触摸设备曾经初始化成功过，就尝试在退出前释放它。
    if touch is not None:
        # 用 try 包一下，避免某些固件没有正确实现 deinit 时影响退出流程。
        try:
            # 主动释放触摸设备资源。
            touch.deinit()
        # 如果释放时报错，就忽略，让后续退出流程继续。
        except Exception:
            # 这里什么都不做，目的是保证程序能顺利收尾。
            pass
        # 无论释放过程是否报错，最后都把 touch 句柄清空。
        touch = None
    # 关闭显示模块。
    Display.deinit()
    # 开启允许睡眠的退出点模式。
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    # 稍微等待一下，让硬件有时间完成关闭动作。
    time.sleep_ms(100)
    # 释放媒体管理器资源。
    MediaManager.deinit()


# 主循环：不断采集图像、识别目标、发送偏移量、显示结果。
def capture_picture():
    # 创建 FPS 计时对象，用来统计帧率。
    fps = time.clock()
    # 当前使用的面积阈值，从初始值开始。
    area_threshold = AREA_THRESHOLD_INIT
    # 记录上一次调参触发时间，用于触摸调参的消抖判断。
    last_adjust_time = 0
    # 记录触摸屏当前是否仍处于按下状态，避免一次长按连续触发多次调参。
    touch_was_down = False
    # 保存平滑后的目标中心点，初始为 None。
    filtered_center = None
    # 保存上一帧或最近一次成功识别到的四个角点。
    last_corners = None
    # 记录已经丢失目标多少帧，初始设为“超过保持范围”。
    lost_frames = MAX_HOLD_FRAMES + 1
    # 帧计数器，用于调试打印和周期性垃圾回收。
    frame_id = 0

    # 进入无限循环，持续处理视频流。
    while True:
        # 更新 FPS 统计。
        fps.tick()
        # 帧编号加 1。
        frame_id += 1
        # 读取当前毫秒时间。
        current_time = ticks_ms()

        # 用 try 包住每一帧处理，防止异常直接把系统打死。
        try:
            # 检查系统退出点，便于安全退出程序。
            os.exitpoint()
            # 从摄像头抓取一帧彩色图像。
            img = sensor.snapshot()
            # 把彩色图转成灰度图，后面矩形检测会更稳、更快。
            gray_img = img.to_grayscale()
            # 获取灰度图对应的 numpy 风格引用，传给 cv_lite。
            img_np = gray_img.to_numpy_ref()

            # 调用 cv_lite 检测灰度图中的矩形，并返回带角点的结果。
            rects = cv_lite.grayscale_find_rectangles_with_corners(
                IMAGE_SHAPE,
                img_np,
                CANNY_THRESH1,
                CANNY_THRESH2,
                APPROX_EPSILON,
                AREA_MIN_RATIO,
                MAX_ANGLE_COS,
                GAUSSIAN_BLUR_SIZE,
            )

            # 读取当前这一帧的触摸点坐标；如果没有触摸，会得到 None。
            touch_point = read_touch_point()
            # 判断距离上次调参是否已经超过消抖时间。
            adjust_ready = time.ticks_diff(current_time, last_adjust_time) > KEY_DEBOUNCE_MS

            # 如果这一帧没有读到触摸点，说明手指已经离开屏幕。
            if touch_point is None:
                # 把“正在按下”标志复位，为下一次单击做准备。
                touch_was_down = False
            # 如果这一帧读到了触摸点，并且这是一次新的按下动作。
            elif not touch_was_down:
                # 只有消抖时间到了，才允许本次触摸真正参与调参。
                if adjust_ready:
                    # 根据触摸点所在区域，计算本次应该增减多少阈值。
                    delta = get_touch_area_threshold_delta(touch_point[0], touch_point[1])
                    # 只有区域判断结果不为 0，才真的去调整阈值。
                    if delta != 0:
                        # 按计算出的步长更新面积阈值，并自动做边界限制。
                        area_threshold = apply_area_threshold_delta(area_threshold, delta)
                        # 记录这次调参的发生时间，用于后续消抖。
                        last_adjust_time = current_time
                # 无论这次触摸有没有触发调参，都记为“当前正处于按下状态”。
                touch_was_down = True

            # 从所有检测到的矩形里选出最佳目标。
            best = find_best_target(rects, filtered_center, area_threshold)
            # 默认状态先设成 LOST，表示“当前未追踪到目标”。
            status_text = "LOST"

            # 如果成功找到了最佳目标。
            if best is not None:
                # 根据四个角点估算出更准确的目标中心。
                raw_center = project_target_center(best["corners"])
                # 对中心进行平滑，减小抖动。
                filtered_center = smooth_point(filtered_center, raw_center, SMOOTH_ALPHA)
                # 保存当前目标的四个角点，用于画框。
                last_corners = best["corners"]
                # 既然已经重新找到目标，就把丢失帧数清零。
                lost_frames = 0
                # 当前状态切换为 TRACK，表示正常追踪。
                status_text = "TRACK"

                # 取平滑后中心的 x 坐标并四舍五入成整数。
                cx = int(round(filtered_center[0]))
                # 取平滑后中心的 y 坐标并四舍五入成整数。
                cy = int(round(filtered_center[1]))
                # 计算目标相对画面中心的 x 方向偏移。
                # 这里是“画面中心减目标中心”，因此目标在右边时 dx 为负。
                dx = FRAME_WIDTH // 2 - cx
                # 计算目标相对画面中心的 y 方向偏移。
                dy = FRAME_HEIGHT // 2 - cy
                # 把偏移量通过串口发给下位机。
                send_tracking(dx, dy)

            # 如果这一帧没找到目标，但前面还有历史中心，并且丢失帧数还在允许保持范围内。
            elif filtered_center is not None and lost_frames < MAX_HOLD_FRAMES:
                # 丢失帧计数加 1。
                lost_frames += 1
                # 状态设为 HOLD，表示“短暂丢失，但继续保持输出”。
                status_text = "HOLD"

                # 继续使用上一次平滑后的中心点。
                cx = int(round(filtered_center[0]))
                # 继续使用上一次平滑后的 y 坐标。
                cy = int(round(filtered_center[1]))
                # 计算 x 偏移。
                dx = FRAME_WIDTH // 2 - cx
                # 计算 y 偏移。
                dy = FRAME_HEIGHT // 2 - cy
                # 继续发送旧中心对应的偏移量，让下位机动作更平稳。
                send_tracking(dx, dy)

            # 如果既没找到目标，历史保持时间也已经耗尽。
            else:
                # 清空历史平滑中心。
                filtered_center = None
                # 清空历史角点。
                last_corners = None
                # 丢失帧数重置到超范围状态。
                lost_frames = MAX_HOLD_FRAMES + 1
                # 通知下位机：目标已经彻底丢失。
                send_lost()

            # 如果当前还有最近一次有效角点，就在画面上把目标轮廓画出来。
            if last_corners is not None:
                # 依次绘制四条边。
                for i in range(4):
                    # 当前边起点。
                    x1, y1 = last_corners[i]
                    # 当前边终点，最后一条边会自动连回第一个点。
                    x2, y2 = last_corners[(i + 1) % 4]
                    # 在图像上画边线。
                    img.draw_line(int(x1), int(y1), int(x2), int(y2), color=GOOD_COLOR, thickness=3)
                    # 在每个角点位置画一个小圆点，方便观察角点是否正确。
                    img.draw_circle(int(x1), int(y1), 3, color=POINT_COLOR, fill=True)

            # 如果当前存在中心点，就画出中心十字和指向画面中心的线。
            if filtered_center is not None:
                # 计算当前中心整数坐标。
                cx = int(round(filtered_center[0]))
                # 计算当前中心整数坐标。
                cy = int(round(filtered_center[1]))
                # 画目标中心十字。
                img.draw_cross(cx, cy, color=BAD_COLOR, thickness=4)
                # 画一条从目标中心指向图像中心的连线，方便观察偏差。
                img.draw_line(cx, cy, FRAME_WIDTH // 2, FRAME_HEIGHT // 2, color=BAD_COLOR, thickness=2)
                # 在左上角显示当前目标中心坐标。
                img.draw_string_advanced(
                    0,
                    0,
                    24,
                    f"target=({cx},{cy})",
                    color=TEXT_COLOR,
                )

            # 不管有没有目标，都在画面中心画一个白色十字，作为参考基准。
            img.draw_cross(FRAME_WIDTH // 2, FRAME_HEIGHT // 2, color=(255, 255, 255), thickness=2)
            # 显示当前追踪状态。
            img.draw_string_advanced(0, 28, 24, f"state={status_text}", color=TEXT_COLOR)
            # 显示当前面积阈值，便于观察当前调参结果。
            img.draw_string_advanced(0, 56, 24, f"area_th={area_threshold}", color=TEXT_COLOR)
            # 显示当前检测到的矩形数量。
            img.draw_string_advanced(0, 84, 24, f"rects={len(rects)}", color=TEXT_COLOR)
            # 显示当前帧率 FPS。
            img.draw_string_advanced(0, 112, 24, f"fps={int(fps.fps())}", color=TEXT_COLOR)
            # 如果当前这一帧存在触摸点，就把触摸坐标显示在图像上，方便校准方向。
            if touch_point is not None:
                # 在图像上打印当前触摸坐标，便于观察触摸坐标和屏幕区域是否对应正确。
                img.draw_string_advanced(0, 188, 20, f"touch=({touch_point[0]},{touch_point[1]})", color=TEXT_COLOR)

            # 把当前处理后的图像按居中偏移量显示到屏幕上。
            Display.show_image(img, x=IMAGE_OFFSET_X, y=IMAGE_OFFSET_Y)

            # 每隔固定帧数打印一次串口调试信息到控制台。
            if frame_id % DEBUG_PRINT_EVERY == 0:
                print(
                    "fps=%d rects=%d th=%d state=%s"
                    % (int(fps.fps()), len(rects), area_threshold, status_text)
                )

            # 每 50 帧主动触发一次垃圾回收，降低内存碎片风险。
            if frame_id % 50 == 0:
                gc.collect()

        # 如果是用户主动中断，例如在终端里停止程序。
        except KeyboardInterrupt as e:
            # 打印停止信息。
            print("user stop:", e)
            # 跳出主循环。
            break
        # 其它异常也打印出来，然后退出循环，避免程序卡死。
        except BaseException as e:
            print("Exception", e)
            break


# 主入口函数，负责组织初始化、运行和善后释放。
def main():
    # 开启退出点功能，让程序在需要时可以响应系统退出请求。
    os.exitpoint(os.EXITPOINT_ENABLE)
    # 记录摄像头是否已经成功初始化，便于 finally 里安全反初始化。
    camera_is_init = False

    # 用 try/finally 保证即使中途报错，也能尽量释放资源。
    try:
        # 打印初始化提示。
        print("camera init")
        # 初始化摄像头和显示。
        camera_init()
        # 标记初始化成功。
        camera_is_init = True
        # 打印进入采集阶段提示。
        print("camera capture")
        # 进入图像采集与追踪主循环。
        capture_picture()
    # 如果主流程中有异常，就打印出来。
    except Exception as e:
        print("Exception", e)
    # 无论是否异常，最后都要尝试做资源释放。
    finally:
        # 只有真正初始化成功过，才执行反初始化。
        if camera_is_init:
            # 打印反初始化提示。
            print("camera deinit")
            # 关闭摄像头和显示，释放相关资源。
            camera_deinit()


# 只有当这个文件被“直接运行”时，才会执行 main()。
# 如果它被别的文件 import 进来，就不会自动执行，方便复用。
if __name__ == "__main__":
    # 调用主函数，启动整个程序。
    main()
