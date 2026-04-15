from maix import camera, display, image, nn, app
import cv2
import numpy as np
import time

# ==================== 配置 ====================
CONTOUR_MIN_AREA = 300
MAX_CONTOURS_PER_OBJ = 2
BIRDVIEW_SIZE = (128, 96)
# =============================================

DST_PTS = np.float32([[0,0], [BIRDVIEW_SIZE[0]-1,0], 
                      [BIRDVIEW_SIZE[0]-1,BIRDVIEW_SIZE[1]-1], [0,BIRDVIEW_SIZE[1]-1]])

def fast_order_points(pts):
    ctr = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-ctr[1], pts[:,0]-ctr[0])
    return pts[np.argsort(angles)][[1,0,3,2]]

def fast_perspective(roi, contour):
    eps = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) != 4:
        return None, None
    src = approx.reshape(4, 2).astype(np.float32)
    src = fast_order_points(src)
    M = cv2.getPerspectiveTransform(src, DST_PTS)
    warped = cv2.warpPerspective(roi, M, BIRDVIEW_SIZE, flags=cv2.INTER_NEAREST)
    return warped, src

# 初始化
detector = nn.YOLOv5(model="/root/models/model_3356.mud", dual_buff=True)
w, h = detector.input_width(), detector.input_height()
cam = camera.Camera(w, h, detector.input_format())
disp = display.Display()

frame_count = 0
start_time = time.time()

while not app.need_exit():
    img = cam.read()
    frame_count += 1
    current_time = time.time()
    
    # FPS计算
    if current_time - start_time >= 1.0:
        fps = frame_count / (current_time - start_time)
        print(f"\n[FPS] {fps:.2f}")
        frame_count, start_time = 0, current_time
    
    objs = detector.detect(img, conf_th=0.7)
    img_cv = image.image2cv(img, ensure_bgr=True, copy=False)
    
    print(f"\n===== Frame {frame_count} | Objects: {len(objs)} =====")
    
    for obj_idx, obj in enumerate(objs):
        x, y, ow, oh = obj.x, obj.y, obj.w, obj.h
        x, y = max(0, x), max(0, y)
        ow, oh = min(ow, w-x), min(oh, h-y)
        
        print(f"[Obj{obj_idx}] Class={obj.class_id} Score={obj.score:.3f} Box=({x},{y},{ow},{oh})")
        
        if ow < 40 or oh < 40:
            print(f"  -> SKIP: too small ({ow}x{oh})")
            continue
        
        roi = img_cv[y:y+oh, x:x+ow]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  -> Contours found: {len(contours)}")
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS_PER_OBJ]
        
        for cnt_idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print(f"  [Cnt{cnt_idx}] Area={area:.1f}", end="")
            
            if area < CONTOUR_MIN_AREA:
                print(f" < {CONTOUR_MIN_AREA} -> SKIP")
                break
            
            warped, src_pts = fast_perspective(roi, cnt)
            
            if warped is None:
                print(f" -> Not quadrilateral -> SKIP")
                continue
            
            bh, bw = warped.shape[:2]
            print(f" -> Perspective OK ({bw}x{bh})")
            print(f"    SrcPts: {src_pts.astype(int).tolist()}")
            
            ox = w - (cnt_idx+1)*(bw+5) - 5
            oy = 5
            if ox > 0:
                img_cv[oy:oy+bh, ox:ox+bw] = warped
                print(f"    Display at ({ox},{oy})")
            else:
                print(f"    Display SKIP: ox={ox}")
    
    disp.show(image.cv2image(img_cv, bgr=True, copy=False))