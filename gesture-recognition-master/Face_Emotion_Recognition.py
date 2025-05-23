import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from deepface import DeepFace
import os

# 얼굴 인식의 기준이 될 이미지 로드 및 인코딩
known_face_encodings = []
known_face_names = []

face_image_dirs = {
    'Person1': ['./boy.jpeg'],
    'Person2': ['./girl.jpeg'],
}

for name, image_paths in face_image_dirs.items():
    encodings = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"이미지 경로 없음: {image_path}")
            continue
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings.append(face_encodings[0])

    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(mean_encoding)
        known_face_names.append(name)

# 영상 장치 초기화
cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

TOLERANCE = 0.6

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)

        face_locations = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                face_locations.append((top_left[1], bottom_right[0], bottom_right[1], top_left[0]))

        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        face_names = []
        emotions = []

        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # 감정 분석
            try:
                face_img = image_rgb[face_locations[i][0]:face_locations[i][2],
                                     face_locations[i][3]:face_locations[i][1]]
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']

                label = {
                    "happy": "smile",
                    "angry": "angry",
                    "sad": "cry",
                    "disgust": "cry"
                }.get(emotion, emotion)

            except Exception as e:
                print("감정 분석 오류:", e)
                label = ""

            # Unknown 제외하고 출력 리스트에 추가
            if name != "Unknown":
                face_names.append(name)
                emotions.append(label)
            else:
                face_names.append(None)
                emotions.append(None)

        # 결과 출력
        if mesh_results.multi_face_landmarks:
            for i, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                if i < len(face_names) and face_names[i] is not None:
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

                    # 얼굴 위 텍스트 출력
                    ih, iw, _ = image.shape
                    x = int(face_landmarks.landmark[1].x * iw)
                    y = int(face_landmarks.landmark[1].y * ih) - 10
                    label_text = f"{face_names[i]} {emotions[i]}"
                    cv2.putText(image, label_text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Face Recognition + Emotion', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
