import cv2
import mediapipe as mp
import numpy as np
import math
import time

# 하트 PNG 로드 & 리사이즈
heart_img = cv2.imread("pngwing.com.png", cv2.IMREAD_UNCHANGED)
heart_img = cv2.resize(heart_img, (200, 200))

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HAND_LANDMARK = mp_hands.HandLandmark

# 배경에 PNG 오버레이 함수
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if y + i >= background.shape[0] or x + j >= background.shape[1]:
                continue
            alpha = overlay[i, j, 3] / 255.0
            for c in range(3):
                background[y + i, x + j, c] = (
                    alpha * overlay[i, j, c] +
                    (1 - alpha) * background[y + i, x + j, c]
                )
    return background

# 파티클 클래스
class EmojiParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-3, 3)
        self.vy = np.random.uniform(-5, -1)
        self.life = 30

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # 중력
        self.life -= 1

    def draw(self, frame):
        if self.life > 0:
            overlay_image(frame, heart_img, int(self.x), int(self.y))

# 하트 감지 조건 완화: 동그랗고 닫히기만 하면 OK
def is_heart(traj):
    if len(traj) < 5:
        return False
    xs = [pt[0] for pt in traj]
    ys = [pt[1] for pt in traj]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    closed = math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) < 150
    round_enough = width > 10 and height > 10
    return closed and round_enough

# 중심 좌표 계산
def get_center(traj):
    xs = [pt[0] for pt in traj]
    ys = [pt[1] for pt in traj]
    return int(np.mean(xs)), int(np.mean(ys))

# 초기화
cap = cv2.VideoCapture(0)
canvas = None
prev_points = {}
trajectory = []
particles = []
last_draw_time = time.time()
emoji_cooldown = 2

# Mediapipe 실행
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("카메라 프레임 없음")
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                index_tip = hand_landmarks.landmark[HAND_LANDMARK.INDEX_FINGER_TIP]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)

                if label not in prev_points:
                    prev_points[label] = (None, None)

                prev_x, prev_y = prev_points[label]

                if label == 'Left':
                    trajectory.append((x, y))

                    # 궤적 시각화 (더 진하게)
                    for pt in trajectory:
                        cv2.circle(canvas, pt, 4, (0, 255, 255), -1)

                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                    prev_points[label] = (x, y)

                    # 속도 측정 → 감지
                    if len(trajectory) > 10:
                        dx = x - trajectory[-2][0]
                        dy = y - trajectory[-2][1]
                        speed = math.hypot(dx, dy)

                        if speed < 8 and time.time() - last_draw_time > emoji_cooldown:
                            if is_heart(trajectory):
                                cx, cy = get_center(trajectory)
                                print("💖 하트 감지됨!")
                                particles.append(EmojiParticle(cx, cy))
                                last_draw_time = time.time()
                            trajectory.clear()

                elif label == 'Right':
                    cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                    prev_points[label] = (None, None)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_points = {}

        # 파티클 업데이트 + 그리기
        for p in particles[:]:
            p.update()
            p.draw(frame)
            if p.life <= 0:
                particles.remove(p)

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow('💖 Heart Particle with PNG', combined)

        key = cv2.waitKey(5)
        if key == ord('r'):
            canvas = np.zeros_like(frame)
            particles.clear()
            trajectory.clear()
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
