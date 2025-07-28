import os
import gradio as gr
import cv2
import numpy as np
import time as time_module
import logging
import threading
from speech_to_text import record_audio, transcribe_with_groq
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs, text_to_speech_with_gtts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
audio_filepath = "audio_question.mp3"

webcam_lock = threading.Lock()

def process_audio_and_chat():
    """Process audio input, transcribe, get AI response, and convert to speech."""
    chat_history = []
    while True:
        try:
            start_time = time_module.time()
            record_audio(file_path=audio_filepath)
            user_input = transcribe_with_groq(audio_filepath)
            if user_input and "goodbye" in user_input.lower():
                break
            if user_input:
                response = ask_agent(user_query=user_input)
                text_to_speech_with_elevenlabs(input_text=response, output_filepath="final.mp3")
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})
            logging.info(f"Audio loop took {time_module.time() - start_time:.2f} seconds")
            yield chat_history
            time_module.sleep(0.5)
        except Exception as e:
            logging.error(f"Error in continuous recording: {e}")
            break

camera = None
is_running = False
last_frame = None
webcam_index = 0
webcam_backend = None

def initialize_camera():
    """Initialize the camera with optimized settings and test multiple indices/backends."""
    global camera, webcam_index, webcam_backend
    with webcam_lock:
        if camera is not None and camera.isOpened():
            return True
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for idx in range(5):
            for backend in backends:
                logging.info(f"Trying webcam index {idx} with backend {backend}")
                try:
                    camera = cv2.VideoCapture(idx, backend)
                    if camera.isOpened():
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        camera.set(cv2.CAP_PROP_FPS, 15)
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        webcam_index = idx
                        webcam_backend = backend
                        logging.info(f"Webcam initialized: index {idx}, backend {backend}")
                        return True
                except Exception as e:
                    logging.warning(f"Failed to open webcam index {idx} with backend {backend}: {e}")
                if camera is not None:
                    camera.release()
        camera = None
        logging.error("No webcam found after trying all indices and backends")
        return False

def start_webcam():
    """Start the webcam feed."""
    global is_running, last_frame
    is_running = True
    if not initialize_camera():
        logging.error("Failed to start webcam")
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Webcam", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        last_frame = placeholder
        return placeholder
    with webcam_lock:
        for _ in range(3):
            ret, frame = camera.read()
            if ret:
                break
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame
            return frame
        logging.error("Failed to capture initial frame")
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Webcam", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        last_frame = placeholder
        return placeholder

def stop_webcam():
    """Stop the webcam feed."""
    global is_running, camera, webcam_index, webcam_backend
    is_running = False
    with webcam_lock:
        if camera is not None:
            camera.release()
            camera = None
            webcam_index = 0
            webcam_backend = None
        logging.info("Webcam stopped")
    return None

def get_webcam_frame():
    """Get current webcam frame with optimized performance."""
    global camera, is_running, last_frame
    if not is_running or camera is None:
        return last_frame
    with webcam_lock:
        for _ in range(3):
            ret, frame = camera.read()
            if ret:
                break
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame
            return frame
        logging.warning("Failed to capture frame")
        return last_frame

with gr.Blocks() as demo:
    gr.Markdown(
        "<h1 style='color: orange; text-align: center; font-size: 4em;'>Jagga Jasoos – Your Personal AI Assistant</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="secondary")
            webcam_output = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                width=320,
                height=240
            )
            webcam_timer = gr.Timer(0.5)

        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                type="messages"
            )
            gr.Markdown("*🎤 Continuous listening mode is active - speak anytime!*")
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")

    start_btn.click(fn=start_webcam, outputs=webcam_output)
    stop_btn.click(fn=stop_webcam, outputs=webcam_output)
    webcam_timer.tick(fn=get_webcam_frame, outputs=webcam_output, show_progress=False)
    clear_btn.click(fn=lambda: [], outputs=chatbot)
    demo.load(fn=process_audio_and_chat, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=False)