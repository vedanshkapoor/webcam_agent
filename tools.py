import cv2
import base64
from dotenv import load_dotenv
import os
import time as time_module
import logging
from groq import Groq

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

webcam_index = None
webcam_backend = None


def capture_image() -> str:
    """Captures one frame from the webcam, resizes it, and encodes it as base64."""
    global webcam_index, webcam_backend
    start_time = time_module.time()

    if webcam_index is None or webcam_backend is None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for idx in range(5):
            for backend in backends:
                logging.info(f"Trying webcam index {idx} with backend {backend}")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    webcam_index = idx
                    webcam_backend = backend
                    logging.info(f"Webcam initialized: index {idx}, backend {backend}")
                    break
                cap.release()
            if webcam_index is not None:
                break
        if webcam_index is None:
            logging.error("No webcam found after trying all indices and backends")
            return ""

    cap = None
    try:
        cap = cv2.VideoCapture(webcam_index, webcam_backend)
        if not cap.isOpened():
            logging.error("Failed to open cached webcam")
            return ""

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)

        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                break
        if not ret:
            logging.error("Failed to capture frame after retries")
            return ""

        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            logging.error("Failed to encode frame")
            return ""

        logging.info(f"capture_image took {time_module.time() - start_time:.2f} seconds")
        return base64.b64encode(buf).decode('utf-8')

    except Exception as e:
        logging.error(f"Error in capture_image: {e}")
        return ""
    finally:
        if cap is not None:
            cap.release()


last_call = 0


def analyze_image_with_query(query: str) -> str:
    """Analyzes a webcam image with the provided query using Groq's vision API."""
    global last_call
    start_time = time_module.time()

    if time_module.time() - last_call < 1:
        return "Please wait before analyzing again"
    last_call = time_module.time()

    if not query:
        return "Error: Query is required"

    try:
        img_b64 = capture_image()
        if not img_b64:
            return "Error: Failed to capture image"
        model = "meta-llama/llama-4-maverick-17b-128e-instruct"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )

        logging.info(f"analyze_image_with_query took {time_module.time() - start_time:.2f} seconds")
        return chat_completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in image analysis: {e}")
        return f"Error: {str(e)}"