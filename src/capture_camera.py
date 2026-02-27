import argparse
import cv2
import time
import numpy as np
import os
import uuid
import shutil

from datetime import datetime
from pygrabber.dshow_graph import FilterGraph
from dotenv import load_dotenv

from src.logger import build_logger
from src.image_trans import image_translate, OBS_TRANS_IMAGE_DEST_PATH, OBS_TRANS_OCR_URL
from src.translator import OBS_TRANS_LLM_HOST, OBS_TRANS_LLM_MODEL, llm_translate_texts

load_dotenv()

OBS_TRANS_VIDEO_VIRTUAL_CAMERA_NAME = os.getenv("OBS_TRANS_VIDEO_VIRTUAL_CAMERA_NAME")

OBS_TRANS_VIDEO_DEFAULT_WIDTH = int(os.getenv("OBS_TRANS_VIDEO_DEFAULT_WIDTH"))
OBS_TRANS_VIDEO_DEFAULT_HEIGHT = int(os.getenv("OBS_TRANS_VIDEO_DEFAULT_HEIGHT"))

OBS_TRANS_VIDEO_CAPTURE_RETRY_COUNT = int(
    os.getenv("OBS_TRANS_VIDEO_CAPTURE_RETRY_COUNT")
)
OBS_TRANS_VIDEO_CAPTURE_RETRY_DELAY_SEC = int(
    os.getenv("OBS_TRANS_VIDEO_CAPTURE_RETRY_DELAY_SEC")
)
OBS_TRANS_VIDEO_CAPTURE_INTERVAL_SEC = float(
    os.getenv("OBS_TRANS_VIDEO_CAPTURE_INTERVAL_SEC")
)

OBS_TRANS_IMAGE_DIFF_THRESHOLD = float(os.getenv("OBS_TRANS_IMAGE_DIFF_THRESHOLD"))

OBS_TRANS_IMAGE_TEMP_FOLDER = os.getenv("OBS_TRANS_IMAGE_TEMP_FOLDER")
OBS_TRANS_IMAGE_BLANK = os.getenv("OBS_TRANS_IMAGE_BLANK")

OBS_TRANS_VIDEO_SHOW_CAPTURE = (
    os.getenv("OBS_TRANS_VIDEO_SHOW_CAPTURE").lower() == "true"
)

logger = build_logger("capture", "capture.log")


def detect_obs_virtual_camera():
    devices = FilterGraph().get_input_devices()

    for index, device in enumerate(devices):
        print(index, device)
        if OBS_TRANS_VIDEO_VIRTUAL_CAMERA_NAME in device:
            print(f"[detect_obs_virtual_camera] choose device index {index}")
            return index

    raise Exception(
        f"Failed to find video device with name '{OBS_TRANS_VIDEO_VIRTUAL_CAMERA_NAME}'"
    )


def init_video_capture():
    cap = cv2.VideoCapture(detect_obs_virtual_camera())

    if not cap.isOpened():
        logger.error("Failed to turn on the camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, OBS_TRANS_VIDEO_DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OBS_TRANS_VIDEO_DEFAULT_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera resolution set to {actual_width}x{actual_height}")

    return cap


def set_trans_image_blank():
    try:
        shutil.copyfile(OBS_TRANS_IMAGE_BLANK, OBS_TRANS_IMAGE_DEST_PATH)
        logger.info(
            f"Successully copy {OBS_TRANS_IMAGE_BLANK} to {OBS_TRANS_IMAGE_DEST_PATH}"
        )
    except Exception as e:
        logger.error(
            f"Failed when copy {OBS_TRANS_IMAGE_BLANK} to {OBS_TRANS_IMAGE_DEST_PATH}: {e}"
        )


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

    image_path = os.path.join(OBS_TRANS_IMAGE_TEMP_FOLDER, gen_png_filename())

    cv2.imwrite(image_path, frame)
    logger.info(f"Saved frame to {image_path}")

    image_translate(
        image_path=image_path,
        ocr_url=ocr_url,
        translate_texts_func=translate_texts_func,
    )
    try:
        os.remove(image_path)
    except Exception as e:
        logger.warning(f"Failed to remove image with error {e}")


def gen_png_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())
    return f"{timestamp}_{unique_id}.png"


def keep_capture_and_translate(ocr_url: str, translate_texts_func: callable):
    if not os.path.exists(OBS_TRANS_IMAGE_TEMP_FOLDER):
        os.mkdir(OBS_TRANS_IMAGE_TEMP_FOLDER)

    retry_count = 0
    prev_frame_bytes = None

    cap = init_video_capture()
    set_trans_image_blank()

    while True:
        ret, frame = cap.read()
        if not ret:
            if retry_count >= OBS_TRANS_VIDEO_CAPTURE_RETRY_COUNT:
                logger.error("Failed to capture even after retry, exit...")
                break
            else:
                retry_count += 1
                logger.warning(f"Failed to capture, {retry_count=}")
                time.sleep(OBS_TRANS_VIDEO_CAPTURE_RETRY_DELAY_SEC)
                continue

        retry_count = 0

        _, frame_bytes = cv2.imencode(".jpg", frame)
        frame_bytes = frame_bytes.tobytes()

        if prev_frame_bytes is None:
            trigger_image_trans(
                frame, ocr_url=ocr_url, translate_texts_func=translate_texts_func
            )
        else:
            diff = image_diff(prev_frame_bytes, frame_bytes)
            logger.debug(f"Image diff: {diff:.2f}")

            if diff > OBS_TRANS_IMAGE_DIFF_THRESHOLD:
                trigger_image_trans(
                    frame, ocr_url=ocr_url, translate_texts_func=translate_texts_func
                )

        prev_frame_bytes = frame_bytes

        if OBS_TRANS_VIDEO_SHOW_CAPTURE:
            cv2.imshow("OBS Virtual Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(OBS_TRANS_VIDEO_CAPTURE_INTERVAL_SEC)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_url", type=str, default=OBS_TRANS_OCR_URL)
    parser.add_argument("--llm_host", type=str, default=OBS_TRANS_LLM_HOST)
    parser.add_argument("--llm_model", type=str, default=OBS_TRANS_LLM_MODEL)

    args = parser.parse_args()
    logger.info(f"{args=}")

    def translate_texts(texts: list[str]):
        translate_kwargs = {
            "llm_host": args.llm_host,
            "llm_model": args.llm_model,
        }
        return llm_translate_texts(texts, **translate_kwargs)

    keep_capture_and_translate(
        ocr_url=args.ocr_url,
        translate_texts_func=translate_texts,
    )
