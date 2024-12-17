from numbers import Number
import cv2
import numpy as np
import asyncio
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play
import tensorflow as tf
import tensorflow_hub as hub


cap = cv2.VideoCapture(0)
cnt = 0
prevWidth = 0

roi = (300, 200, 900, 680)

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
# model = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights="imagenet")


labels = {0: "Background", 1: "crosswalk", 2: "road", 3: "pathway"} 

def preprocess_frame_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def preprocess_frame_classification(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = np.array(resized_frame) / 255.0  # Normalizing pixel values
    return np.expand_dims(normalized_frame, axis=0)

def detect_path(edges, roi):
    mask = np.zeros_like(edges)
    mask[roi[1]:roi[3], roi[0]:roi[2]] = edges[roi[1]:roi[3], roi[0]:roi[2]]
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines

def classify_frame(frame):
    preprocessed_frame = preprocess_frame_classification(frame)
    predictions = model(preprocessed_frame).numpy()
    predicted_class = np.argmax(predictions[0])
    return labels.get(predicted_class, "Background")

def calculate_width(lines):
    if lines is None:
        return None
    
    left_line = min(lines, key=lambda line: line[0][0])
    right_line = max(lines, key=lambda line: line[0][0])
    
    width_in_pixels = right_line[0][0] - left_line[0][0]
    return width_in_pixels

async def generate_tts(message):
    tts = gTTS(text=message, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

async def play_audio(audio_file):
    audio = AudioSegment.from_file(audio_file, format="mp3")
    play(audio)

async def notify_user(message):
    audio_file = await generate_tts(message)
    await play_audio(audio_file)

async def main():
    global roi, cnt, prevWidth

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame. Exiting ...")
            break

        color_frame = frame.copy()
        
        # Edge detection
        edges = preprocess_frame_edges(frame)
        lines = detect_path(edges, roi)

        # Classification
        label = classify_frame(frame)

        # Classification result
        cv2.putText(color_frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Processing width measurement if lines are detected
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            width_in_pixels = calculate_width(lines)
            if width_in_pixels:
                cnt += 1
                width_in_feet = width_in_pixels * 0.01
                cv2.putText(color_frame, f"Width: {width_in_feet:.2f} feet", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if cnt == 10:
                    if abs(prevWidth - width_in_feet) >= 3:
                        await notify_user(f"You have {width_in_feet:.2f} feet of width available.")
                        prevWidth = width_in_feet
                    cnt = 0
        
        # Notifying users if a crosswalk is detected
        if "crosswalk" in label.lower():
            await notify_user("A crosswalk is detected ahead!")
        
        # Show the video frame with overlays
        cv2.imshow('Path Assistance', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())
