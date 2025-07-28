import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=10, phrase_time_limit=None):
    """Record audio from the microphone and save it as an MP3 file."""
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone(sample_rate=16000)
        with mic as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete")
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="64k")
            logging.info(f"Audio saved to {file_path}")
    except sr.WaitTimeoutError:
        logging.warning("No speech detected within timeout")
        return None
    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        return None

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def transcribe_with_groq(audio_filepath):
    """Transcribe audio file using Groq API."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")
    if not os.path.exists(audio_filepath):
        logging.warning("No audio file to transcribe")
        return ""
    stt_model = "whisper-large-v3"
    try:
        with open(audio_filepath, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return ""