import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load MoveNet MultiPose model
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet_input_size = 256

def preprocess_frame(frame):
    img = cv2.resize(frame, (movenet_input_size, movenet_input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.int32)  # MoveNet expects int32
    return img[np.newaxis, ...]

def extract_faces_from_keypoints(frame, people, threshold=0.3):
    faces_with_sizes = []
    h, w, _ = frame.shape

    for person in people[0]:
        keypoints = np.reshape(person[:51], (17, 3))

        nose_confidence = keypoints[0][2]
        left_shoulder_confidence = keypoints[5][2]
        right_shoulder_confidence = keypoints[6][2]

        # use only high-confidence detections
        if nose_confidence < threshold or left_shoulder_confidence < threshold or right_shoulder_confidence < threshold:
            continue

        nose_x, nose_y = int(keypoints[0][1] * w), int(keypoints[0][0] * h)
        left_shoulder_x, left_shoulder_y = int(keypoints[5][1] * w), int(keypoints[5][0] * h)
        right_shoulder_x, right_shoulder_y = int(keypoints[6][1] * w), int(keypoints[6][0] * h)

        # dynamic face size: proportional to shoulder width
        shoulder_width = int(np.linalg.norm([right_shoulder_x - left_shoulder_x, right_shoulder_y - left_shoulder_y]))
        face_size = max(shoulder_width, 60)  # minimum face size of 60 to avoid too small faces

        x1, y1 = max(0, nose_x - face_size // 2), max(0, nose_y - face_size // 2)
        x2, y2 = min(w, nose_x + face_size // 2), min(h, nose_y + face_size // 2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face_resized = cv2.resize(face_gray, (100, 100))

        actual_face_size = y2 - y1

        faces_with_sizes.append((face_resized, actual_face_size))

    return faces_with_sizes


def initialize_players_faces_multipose(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    filter_bank = []
    next_id = 1

    print("Initializing... Press 'c' when all players are in frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.imshow("Press 'c' to capture", display)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            input_img = preprocess_frame(frame)
            outputs = movenet.signatures["serving_default"](tf.constant(input_img))
            people = outputs["output_0"].numpy()  # shape: (1, 6, 56)

            faces_with_sizes = extract_faces_from_keypoints(frame, people)
            print(f"{len(faces_with_sizes)} faces detected via MoveNet MultiPose.")

            for face, face_size in faces_with_sizes:
                filter_bank.append({
                    "id": f"Player_{next_id}",
                    "face": face,
                    "face_size": face_size  # store the face size
                })
                next_id += 1

            break

    cap.release()
    cv2.destroyAllWindows()
    return filter_bank

# Run the function for testing
if __name__ == "__main__":
    filter_bank = initialize_players_faces_multipose()

    print(f"\nCaptured {len(filter_bank)} player faces:\n")
    for player in filter_bank:
        print(f"- {player['id']}")
        cv2.imshow(player["id"], player["face"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
