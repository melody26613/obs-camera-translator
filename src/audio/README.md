## Installation

### Install WhisperLive

0. download model
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade huggingface_hub

hf download Systran/faster-whisper-medium --local-dir Systran/faster-whisper-medium/

hf download Systran/faster-whisper-small --local-dir Systran/faster-whisper-small/

deactivate
```

1. run server by docker
```bash
docker pull ghcr.io/collabora/whisperlive-gpu:latest

docker run -d --name whisperlive --gpus all -p 9090:9090 \
    -v /mnt/d/workspace/models/hf_models:/models \
    ghcr.io/collabora/whisperlive-gpu:latest \
    python run_server.py --faster_whisper_custom_model_path /models/Systran/faster-whisper-small --max_connection_time 28800

# Error on Nvidia MX350
# INFO:root:Custom model option was provided. Switching to single model mode.
# INFO:websockets.server:connection open
# INFO:root:New client connected
# INFO:root:Using custom model /models/Systran/faster-whisper-medium
# /usr/local/lib/python3.10/site-packages/torch/cuda/__init__.py:283: UserWarning:
#     Found GPU0 NVIDIA GeForce MX350 which is of cuda capability 6.1.
#     Minimum and Maximum cuda capability supported by this version of PyTorch is
#     (7.0) - (12.0)

#   warnings.warn(
# /usr/local/lib/python3.10/site-packages/torch/cuda/__init__.py:304: UserWarning:
#     Please install PyTorch with a following CUDA
#     configurations:  12.6 following instructions at
#     https://pytorch.org/get-started/locally/

#   warnings.warn(matched_cuda_warn.format(matched_arches))
# /usr/local/lib/python3.10/site-packages/torch/cuda/__init__.py:326: UserWarning:
# NVIDIA GeForce MX350 with CUDA capability sm_61 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
# If you want to use the NVIDIA GeForce MX350 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

#   warnings.warn(
# INFO:root:Using Device=cuda with precision float32
# INFO:root:Model not in model_sizes
# INFO:root:Loading model: /models/Systran/faster-whisper-medium
# ERROR:root:Failed to load model: CUDA failed with error out of memory
# INFO:root:Running faster_whisper backend.
# INFO:root:Connection closed by client
# INFO:root:Cleaning up.

docker run -it --name whisperlive --gpus all -p 9090:9090 \
    -v /mnt/d/workspace/models/hf_models:/models \
    ghcr.io/collabora/whisperlive-gpu:latest bash

# degrade pytorch
root@a5f516fcfff8:/app# pip install torch==1.13.1
# testing if pytoch support sm_61
root@a5f516fcfff8:/app# python
>>> import torch
>>> print(torch.cuda.get_arch_list())

# testing server
root@a5f516fcfff8:/app# python run_server.py --faster_whisper_custom_model_path /models/Systran/faster-whisper-small --port 9090
```

2. run client
* preparation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* run by python cli
```python
python
>>> from whisper_live.client import TranscriptionClient
>>> client = TranscriptionClient(host="localhost", port=9090, lang="ja")
>>> client("demon_slayer_infinity_castle_trailer.mp3")
最終局面という言葉が何度も頭を絵を切る

>>> from my_whisperlive_client import MyTranscriptionClient
>>> client = MyTranscriptionClient(host="localhost", port=9090, lang="ja")
>>> client("demon_slayer_infinity_castle_trailer.mp3")
```

* run by python script, **without** translate by LLM
```bash
# check your audio device by audio/list_audio_devices.py
# modify the AUDIO_DEVICE_NAME in audio/audio_trans.py

# execute under obs-camera-translator/
# testing audio file
python -m src.audio.audio_trans --stt_host localhost --stt_port 9090 --lang ja --file audio/demon_slayer_infinity_castle_trailer.mp3

# testing microphone
python -m src.audio.audio_trans --stt_host localhost --stt_port 9090 --lang ja
```

* run by python script, **with** translate by LLM
```bash
# check your audio device by audio/list_audio_devices.py
# modify the AUDIO_DEVICE_NAME in audio/audio_trans.py

# execute under obs-camera-translator/
# testing audio file
python -m src.audio.audio_trans --stt_host localhost --stt_port 9090 --lang ja --file audio/demon_slayer_infinity_castle_trailer.mp3 --enable_translate --llm_host http://<ip>:<port> --llm_model <model_name>

# testing microphone
python -m src.audio.audio_trans --stt_host localhost --stt_port 9090 --lang ja --enable_translate --llm_host http://<ip>:<port> --llm_model <model_name>
```

## References

* [Whisper](https://github.com/openai/whisper)
* [WhisperLive](https://github.com/collabora/WhisperLive)
* [WhisperLive docker](https://github.com/collabora/WhisperLive?tab=readme-ov-file#whisper-live-server-in-docker)
* [WhisperLive Youtube Video: Realtime Transcription](https://www.youtube.com/watch?v=0PHWCApIcCI)