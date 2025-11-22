import cv2
import numpy as np
import time

# (Windows라면) 비프음 사용
try:
    import winsound
    USE_BEEP = True
except ImportError:
    USE_BEEP = False

# 셔틀콕(하얀색) HSV 범위 (대략적인 값, 꼭 카메라 환경에 맞게 조정 필요)
LOWER_WHITE = np.array([0,   0, 200])   # H, S, V
UPPER_WHITE = np.array([180, 60, 255])

# 최소 셔틀콕 크기 (노이즈 제거용)
MIN_AREA = 80   # 필요에 따라 조정

# 연속 비프 방지용
last_fault_time = 0
FAULT_COOLDOWN = 1.0  # 초 단위

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    height, width = frame.shape[:2]

    # 카메라 중심선 (가로선)
    center_line_y = height // 2  # 프레임 중앙 높이
    # 필요하면 기준선을 조금 위로 올리고 싶을 때:
    # center_line_y = int(height * 0.55)  # 예: 아래쪽으로 55% 지점

    # BGR -> HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 하얀 셔틀콕 마스크
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    # 노이즈 제거: 조금 깎고, 다시 부풀리기
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shuttle_center = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        cx, cy = int(x), int(y)

        # 셔틀콕 후보를 표시
        cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        shuttle_center = (cx, cy)
        break  # 가장 큰 하나만 사용 (필요하면 area로 최대값 선택 가능)

    # 중심선 그리기
    cv2.line(frame, (0, center_line_y), (width, center_line_y), (0, 255, 0), 2)
    cv2.putText(frame, "Center line", (10, center_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === 반칙 판정 로직 (단순 버전) ===
    # y 좌표는 위로 갈수록 작아짐 → cy < center_line_y 이면 중심선보다 "높은" 위치
    if shuttle_center is not None:
        cx, cy = shuttle_center

        if cy < center_line_y:
            # 화면에 표시
            cv2.putText(frame, "FAULT: High serve!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 비프음 (쿨다운 적용)
            now = time.time()
            if now - last_fault_time > FAULT_COOLDOWN:
                if USE_BEEP:
                    winsound.Beep(1000, 300)  # 1000 Hz, 0.3초
                else:
                    print("FAULT: High serve!")
                last_fault_time = now

    cv2.imshow("Badminton High Serve Detection", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
