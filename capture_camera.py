import argparse
import cv2
import time
import numpy as np
import os
import uuid
import shutil

from datetime import datetime
from pygrabber.dshow_graph import FilterGraph

from logger import build_logger
from image_trans import image_translate, TRANS_DEST_IMAGE_PATH, OCR_SERVICE_URL
from translator import OLLAMA_HOST, OLLAMA_MODEL, ollama_translate_texts

VIDEO_OBS_VIRTUAL_CAMERA_NAME = "OBS Virtual Camera"

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

RETRY_COUNT = 5
RETRY_DELAY_SEC = 3
IMAGE_DIFF_THRESHOLD = 20.0
IMAGE_TIME_INTERVAL = 0.2

TEMP_IMAGE_FOLDER = "./temp"
BLANK_IMAGE = "./pic/blank.png"

SHOW_CURRENT_VIDEO = False

logger = build_logger("capture", "capture.log")


def detect_obs_virtual_camera():
    devices = FilterGraph().get_input_devices()

    for index, device in enumerate(devices):
        print(index, device)
        if VIDEO_OBS_VIRTUAL_CAMERA_NAME in device:
            print(f"[detect_obs_virtual_camera] choose device index {index}")
            return index

    raise Exception(
        f"Failed to find video device with name '{VIDEO_OBS_VIRTUAL_CAMERA_NAME}'"
    )


def init_video_capture():
    cap = cv2.VideoCapture(detect_obs_virtual_camera())

    if not cap.isOpened():
        logger.error("Failed to turn on the camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera resolution set to {actual_width}x{actual_height}")

    return cap


def set_trans_image_blank():
    try:
        shutil.copyfile(BLANK_IMAGE, TRANS_DEST_IMAGE_PATH)
        logger.info(f"Successully copy {BLANK_IMAGE} to {TRANS_DEST_IMAGE_PATH}")
    except Exception as e:
        logger.error(f"Failed when copy {BLANK_IMAGE} to {TRANS_DEST_IMAGE_PATH}: {e}")


def image_diff(image_bytes1: bytes, image_bytes2: bytes) -> float:
    image_array1 = np.frombuffer(image_bytes1, np.uint8)
    image_array2 = np.frombuffer(image_bytes2, np.uint8)

    image1 = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)

    if image1.shape != image2.shape:
        logger.warning("Image shapes differ, resizing for comparison.")
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    mse = np.mean((image1.astype("float") - image2.astype("float")) ** 2)
    return mse


def trigger_image_trans(frame, ocr_url: str, translate_texts_func: callable):
    logger.info("image_trans triggered!")

    set_trans_image_blank()

    image_path = os.path.join(TEMP_IMAGE_FOLDER, gen_png_filename())

    cv2.imwrite(image_path, frame)
    logger.info(f"Saved frame to {image_path}")

    image_translate(
        image_path=image_path,
        ocr_url=ocr_url,
        translate_texts_func=translate_texts_func,
    )
    os.remove(image_path)


def gen_png_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.png"


def keep_capture_and_translate(ocr_url: str, translate_texts_func: callable):
    if not os.path.exists(TEMP_IMAGE_FOLDER):
        os.mkdir(TEMP_IMAGE_FOLDER)

    retry_count = 0
    prev_frame_bytes = None

    cap = init_video_capture()
    set_trans_image_blank()

    while True:
        ret, frame = cap.read()
        if not ret:
            if retry_count >= RETRY_COUNT:
                logger.error("Failed to capture even after retry, exit...")
                break
            else:
                retry_count += 1
                logger.warning(f"Failed to capture, {retry_count=}")
                time.sleep(RETRY_DELAY_SEC)
                continue

        retry_count = 0

        _, frame_bytes = cv2.imencode(".jpg", frame)
        frame_bytes = frame_bytes.tobytes()

        if prev_frame_bytes is None:
            trigger_image_trans(frame, ocr_url=ocr_url, translate_texts_func=translate_texts_func)
        else:
            diff = image_diff(prev_frame_bytes, frame_bytes)
            logger.debug(f"Image diff: {diff:.2f}")

            if diff > IMAGE_DIFF_THRESHOLD:
                trigger_image_trans(frame, ocr_url=ocr_url, translate_texts_func=translate_texts_func)

        prev_frame_bytes = frame_bytes

        if SHOW_CURRENT_VIDEO:
            cv2.imshow("OBS Virtual Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(IMAGE_TIME_INTERVAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_url", type=str, default=OCR_SERVICE_URL)
    parser.add_argument("--ollama_host", type=str, default=OLLAMA_HOST)
    parser.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL)

    args = parser.parse_args()
    logger.info(f"{args=}")

    def translate_texts(texts: list[str]):
        translate_kwargs = {
            "ollama_host": args.ollama_host,
            "ollama_model": args.ollama_model,
        }
        return ollama_translate_texts(texts, **translate_kwargs)

    keep_capture_and_translate(
        ocr_url=args.ocr_url,
        translate_texts_func=translate_texts,
    )
