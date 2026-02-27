import pyaudio
import argparse
import time
import traceback
import os

from cachetools import FIFOCache
from threading import Thread
from dotenv import load_dotenv
from urllib.parse import urlparse

from src.logger import build_logger
from src.translator import llm_translate_text, OBS_TRANS_LLM_HOST, OBS_TRANS_LLM_MODEL
from src.audio.my_whisperlive_client import MyTranscriptionClient

load_dotenv()

OBS_TRANS_AUDIO_DEVICE_NAME = os.getenv("OBS_TRANS_AUDIO_DEVICE_NAME")

OBS_TRANS_STT_HOST = os.getenv("OBS_TRANS_STT_HOST")
OBS_TRANS_STT_SOURCE_LANG = os.getenv("OBS_TRANS_STT_SOURCE_LANG")
OBS_TRANS_STT_OUTPUT_SRT_FILE = os.getenv("OBS_TRANS_STT_OUTPUT_SRT_FILE")
OBS_TRANS_STT_OUTPUT_TRANSCRIPTION = os.getenv("OBS_TRANS_STT_OUTPUT_TRANSCRIPTION")

OBS_TRANS_STT_CACHE_TRANSCRIPTION_LEN = int(
    os.getenv("OBS_TRANS_STT_CACHE_TRANSCRIPTION_LEN")
)
OBS_TRANS_STT_CACHE_TRANSLATION_LEN = int(
    os.getenv("OBS_TRANS_STT_CACHE_TRANSLATION_LEN")
)
OBS_TRANS_STT_MAX_OUTPUT_LEN = int(os.getenv("OBS_TRANS_STT_MAX_OUTPUT_LEN"))

# key: start time in string, value: transcript dict
orig_transcription_cache = FIFOCache(maxsize=OBS_TRANS_STT_CACHE_TRANSCRIPTION_LEN)

# key: start time in string, value: translated dict
translated_cache = FIFOCache(maxsize=OBS_TRANS_STT_CACHE_TRANSLATION_LEN)

logger = build_logger("audio", "audio.log")


def detect_audio_device():
    p = pyaudio.PyAudio()

    for index in range(p.get_device_count()):
        dev = p.get_device_info_by_index(index)

        if dev["maxInputChannels"] > 0:
            print(f"device [{index}] {dev['name']}")

            if OBS_TRANS_AUDIO_DEVICE_NAME in dev["name"]:
                print(f"[detect_audio_device] choose device index {index}")
                return index

    raise Exception(
        f"Failed to find audio device with name '{OBS_TRANS_AUDIO_DEVICE_NAME}'"
    )


def on_transcription(texts: str, transcriptions: list):
    """
    example data of transcriptions
    [
        {
            "start": "2.158",
            "end": "3.072",
            "text": " Coming through",
            "completed": false
        },
        ...
    ]
    """
    global orig_transcription_cache

    for orig in transcriptions:
        logger.info(f"[on_transcription] {orig=}")
        start_time = orig["start"]

        if start_time in orig_transcription_cache:
            old_trans = orig_transcription_cache[start_time]

            if old_trans["completed"]:
                continue

        orig_transcription_cache[start_time] = orig


def transcription_worker(translate_text_func: callable):
    while True:
        try:
            if len(orig_transcription_cache) == 0:
                time.sleep(0.5)
                continue

            is_updated = False

            for start_time, orig in orig_transcription_cache.items():
                text = orig["text"]
                completed = orig["completed"]

                translated_data = translated_cache.get(start_time, None)
                already_translated = translated_data and (
                    "translated_text" in translated_data
                )

                if already_translated:
                    copied_data = translated_data
                else:
                    copied_data = orig.copy()

                if completed and not already_translated:
                    copied_data["translated_text"] = translate_text_func(text)
                    is_updated = True

                if start_time not in translated_cache:
                    is_updated = True
                elif text != translated_cache[start_time]["text"]:
                    is_updated = True

                translated_cache[start_time] = copied_data

            if not is_updated:
                time.sleep(0.1)
                continue

            output_transcription(translated_cache)

        except Exception as e:
            logger.error(f"[transcription_worker] {e}")
            logger.error(
                f"[transcription_worker] exception traceback: {traceback.format_exc()}"
            )
            time.sleep(0.1)


def output_transcription(cache: FIFOCache, filename=OBS_TRANS_STT_OUTPUT_TRANSCRIPTION):
    is_first_line = True

    print(f"----------------------------------------")

    with open(filename, "w", encoding="utf-8") as f:
        if len(cache) > OBS_TRANS_STT_MAX_OUTPUT_LEN:
            last_start_times = list(cache.keys())[-OBS_TRANS_STT_MAX_OUTPUT_LEN:]
        else:
            last_start_times = list(cache.keys())

        for start_time in last_start_times:
            data = cache.get(start_time)

            if not is_first_line:
                f.write("\n\n")

            text = data["text"].strip()
            translated_text = data.get("translated_text", None)

            f.write(text)
            print(text)

            if translated_text:
                f.write("\n" + translated_text)
                print(translated_text)

            is_first_line = False

    print("========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt_host", type=str, default=OBS_TRANS_STT_HOST)
    parser.add_argument("--lang", type=str, default=OBS_TRANS_STT_SOURCE_LANG)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument(
        "--output_transcription_path", type=str, default=OBS_TRANS_STT_OUTPUT_SRT_FILE
    )

    parser.add_argument("--enable_translate", action="store_true")
    parser.add_argument("--llm_host", type=str, default=OBS_TRANS_LLM_HOST)
    parser.add_argument("--llm_model", type=str, default=OBS_TRANS_LLM_MODEL)

    args = parser.parse_args()
    logger.info(f"{args=}")

    def translate_text(text: str):
        if args.enable_translate:
            translate_kwargs = {
                "llm_host": args.llm_host,
                "llm_model": args.llm_model,
            }
            return llm_translate_text(text, **translate_kwargs).strip()
        else:
            return f"Skip translation '{text}'"

    Thread(
        target=transcription_worker,
        kwargs={
            "translate_text_func": translate_text,
        },
        daemon=True,
    ).start()

    parsed_url = urlparse(args.stt_host)
    stt_ip = parsed_url.hostname
    stt_port = parsed_url.port

    common_kwargs = {
        "host": stt_ip,
        "port": stt_port,
        "lang": args.lang,
        "transcription_callback": on_transcription,
        "output_transcription_path": args.output_transcription_path,
    }

    if args.file:
        client = MyTranscriptionClient(**common_kwargs)
        client(args.file)
    else:
        client = MyTranscriptionClient(
            **common_kwargs, audio_input_device=detect_audio_device()
        )
        client()
