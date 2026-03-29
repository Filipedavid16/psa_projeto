import threading
import queue
from pathlib import Path

import pygame
from gtts import gTTS

from config import AUDIO_DB_DIR
from utils import sanitize_folder_name


def get_person_audio_path(person_name):
    safe_name = sanitize_folder_name(person_name)
    return AUDIO_DB_DIR / safe_name / "ola.mp3"


def ensure_greeting_audio_file(person_name):
    if not person_name or not str(person_name).strip():
        return None

    audio_path = get_person_audio_path(person_name)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_path.exists():
        return audio_path

    first_name = str(person_name).strip().split()[0]
    text = f"Olá {first_name}"

    try:
        tts = gTTS(text=text, lang="pt")
        tts.save(str(audio_path))
        print(f"Áudio criado para '{person_name}': {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Erro ao criar áudio para '{person_name}': {e}")
        return None


class AudioManager:
    def __init__(self):
        self.messages = queue.Queue()

        pygame.mixer.init()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            audio_path = self.messages.get()
            if audio_path is None:
                break

            try:
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)

                pygame.time.wait(1000)
            except Exception as e:
                print(f"Erro ao reproduzir áudio: {e}")

    def play_person_greeting(self, person_name):
        audio_path = ensure_greeting_audio_file(person_name)
        if audio_path is None:
            return

        self.messages.put(audio_path)

    def stop(self):
        self.messages.put(None)