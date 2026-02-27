import argparse
import requests
import re
import os

from typing import List
from dotenv import load_dotenv

from logger import build_logger
from googletrans import Translator

load_dotenv()

OBS_TRANS_LLM_HOST = os.getenv("OBS_TRANS_LLM_HOST")
OBS_TRANS_LLM_MODEL = os.getenv("OBS_TRANS_LLM_MODEL")
OBS_TRANS_TIMEOUT_SEC_LLM = int(os.getenv("OBS_TRANS_TIMEOUT_SEC_LLM"))

OBS_TRANS_MODE = os.getenv("OBS_TRANS_MODE").lower()

OBS_TRANS_LLM_SYSTEM_PROMPT = os.getenv("OBS_TRANS_LLM_SYSTEM_PROMPT")
OBS_TRANS_GOOGLE_SOURCE_LANG = os.getenv("OBS_TRANS_GOOGLE_SOURCE_LANG")
OBS_TRANS_GOOGLE_TARGET_LAND = os.getenv("OBS_TRANS_GOOGLE_TARGET_LAND")


logger = build_logger("ocr_trans", "translation.log")


def file_to_string(filename) -> str:
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()


llm_translate_system_prompt = file_to_string(OBS_TRANS_LLM_SYSTEM_PROMPT)


def google_translate(text: str, **kwargs):
    src_lang = kwargs.get("src_lang", OBS_TRANS_GOOGLE_SOURCE_LANG)
    dest_lang = kwargs.get("dest_lang", OBS_TRANS_GOOGLE_TARGET_LAND)

    translator = Translator()

    try:
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
        translated_text = translation.text
        return translated_text
    except Exception as e:
        logger.error(f"""Failed to translate with text: "{text}", the error is: {e}""")
        return text


def google_translate_texts(texts: List[str], **kwargs) -> List[str]:
    formatted_texts = f"""<!--{"--><!--".join(texts)}-->"""
    logger.info(f"{formatted_texts=}")

    translation_string = google_translate(text=formatted_texts)
    logger.info(f"{translation_string=}")

    # < ！? -+   : left brackets, and allow mark in half-width or full-width following
    # \s*        : allow spaces
    # (.*?)      : get the content we need
    # \s* -+ >   : allow spaces and multiple dashes and ending mark
    pattern = r"<\s*[!！]?\s*-+\s*(.*?)\s*-+\s*>"
    matches = re.findall(pattern, translation_string)
    logger.info(f"{matches=}")

    return matches


def llm_translate_text(text: str, **kwargs) -> str:
    if not text:
        return ""

    llm_host = kwargs.get("llm_host", OBS_TRANS_LLM_HOST)
    llm_model = kwargs.get("llm_model", OBS_TRANS_LLM_MODEL)

    request_body = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": llm_translate_system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "stream": False,
        "think": False,
    }

    logger.info(f"{request_body=}")

    api_url = f"{llm_host}/v1/chat/completions"

    try:
        response = requests.post(
            api_url,
            json=request_body,
            timeout=OBS_TRANS_TIMEOUT_SEC_LLM,
        )
        response.raise_for_status()

        llm_response = response.json()

        translated_text = (
            llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        return translated_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call llm, error={e}")
        return ""


def llm_translate_texts(texts: List[str], **kwargs) -> List[str]:
    formatted_texts = "\n".join([f"『{text}』" for text in texts])
    logger.info(f"{formatted_texts=}")

    translation_string = llm_translate_text(text=formatted_texts, **kwargs)
    logger.info(f"{translation_string=}")

    # split by one new line or multiple new lines
    matches = re.split(r"\n+", translation_string.strip())
    logger.info(f"{matches=}")

    matches = [strip_quotes(text) for text in matches]
    logger.info(f"{matches=}")

    return matches


def strip_quotes(text: str) -> str:
    text = text.strip()

    quotes = [
        ("『", "』"),
        ("‘", "’"),
        ('"', '"'),
        ("'", "'"),
        ("「", "」"),
        ("“", "”"),
        ("《", "》"),
    ]

    for ql, qr in quotes:
        if text.startswith(ql) and text.endswith(qr) and len(text) >= 2:
            return text[len(ql) : -len(qr)]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_host", type=str, default=OBS_TRANS_LLM_HOST)
    parser.add_argument("--llm_model", type=str, default=OBS_TRANS_LLM_MODEL)
    parser.add_argument(
        "--mode",
        type=str,
        default=OBS_TRANS_MODE,
        help="Translation mode('llm' or 'google'), default using 'llm'",
    )

    args = parser.parse_args()
    logger.info(f"{args=}")

    texts = [
        "こんにちは",
        "お名前を伺えますか",
    ]
    # texts = [
    #     "お爺さんとおばあさんはなんだかそれが面白くて体のあちこちをこすっては、前のものと一緒にしてぎゅっと丸め、またこすってあかをだしてはさっき丸めたものに、さらにくっつけて・・とやっていたら、しまいには一かたまりになったので、ちょっといたずら心を起こしたおじいさんは、それを小さな人の形にしてみました。",
    #     "できあがった小さな人形をみているうちに、なんとなくそれが可愛くなってきたおじいさんとおばあさんは、そのまま捨てるにはかわいそうな気がして、あかで作った小さな人形をとりあえず神棚において、特に何を拝むでもなくぱんぱんと手を打ちました。 すると不思議なことに神棚に上がった人形がとつぜんむくむくと体をゆすったと思ったら、大きく伸びをして周りをきょろきょろ見回したのです！",
    #     "ふたりは夢でも見ているのかと思ってぽかんとそれを見ていましたが、すぐに神棚の人形とおじいさんとおばあさんの目が合い、人形はぴょんっと下に飛び降りたかとおもうと、ふたりの前にすっくと立ちました。",
    # ]

    if args.mode == "google":
        logger.info("-----Test for google translate----")
        logger.info(f"{texts=}")
        response = google_translate_texts(texts)
        logger.info(f"{response=}")
    else:
        logger.info("-----Test for LLM translation-----")
        logger.info(f"{texts=}")
        response = llm_translate_texts(
            texts=texts, llm_host=args.llm_host, llm_model=args.llm_model
        )
        logger.info(f"{response=}")
