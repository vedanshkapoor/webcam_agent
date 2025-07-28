import os
from elevenlabs.client import ElevenLabs
import subprocess
import platform
import threading
import time as time_module
import logging
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))


def play_audio(output_filepath, os_name):
    """Play audio asynchronously in a separate thread."""
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', output_filepath], check=True)
        elif os_name == "Windows":
            audio = AudioSegment.from_mp3(output_filepath)
            wav_file = output_filepath.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_file}").PlaySync();'], check=True)
            os.remove(wav_file)
        elif os_name == "Linux":
            subprocess.run(['aplay', output_filepath], check=True)
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        logging.error(f"Error playing audio: {e}")


last_call_elevenlabs = 0
last_call_gtts = 0


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Convert text to speech using ElevenLabs and play asynchronously."""
    global last_call_elevenlabs
    start_time = time_module.time()

    if not input_text or time_module.time() - last_call_elevenlabs < 1:
        return
    last_call_elevenlabs = time_module.time()

    try:
        audio = client.text_to_speech.convert(
            text=input_text,
            voice_id="ZF6FPAbjXT4488VcRRnw",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32"
        )
        with open(output_filepath, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        os_name = platform.system()
        threading.Thread(target=play_audio, args=(output_filepath, os_name), daemon=True).start()

        logging.info(f"text_to_speech_with_elevenlabs took {time_module.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in ElevenLabs TTS: {e}")


def text_to_speech_with_gtts(input_text, output_filepath):
    """Convert text to speech using gTTS and play asynchronously."""
    global last_call_gtts
    start_time = time_module.time()

    if not input_text or time_module.time() - last_call_gtts < 1:
        return
    last_call_gtts = time_module.time()

    try:
        language = "en"
        audioobj = gTTS(text=input_text, lang=language, slow=False)
        audioobj.save(output_filepath)

        os_name = platform.system()
        threading.Thread(target=play_audio, args=(output_filepath, os_name), daemon=True).start()

        logging.info(f"text_to_speech_with_gtts took {time_module.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in gTTS: {e}")