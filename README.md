# Camera Translator
Capture video frames from the OBS virtual camera, extract text using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and translate it with the [Ollama](https://ollama.com/download/linux) LLM model.


![demo 1](pic/2025-08-24_10-01-41.gif)

![demo 2](pic/2025-08-24_10-03-11.gif)

* flow

OBS virtual camera → capture image → PaddleOCR REST API → LLM translate → Output translated image

## Preparation
1. install [nvidia driver](https://developer.nvidia.com/cuda-downloads)
2. install [docker](https://docs.docker.com/engine/install/ubuntu/)
3. install [nvidia toolkit for docker container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)

* run the docker container from my repo [paddle-ocr-restapi](https://github.com/melody26613/paddle-ocr-restapi)

4. install [ollama](https://ollama.com/download/linux)

* ollama model
```bash
ollama pull gemma2:2b
```

* ollama config
```bash
sudo systemctl edit ollama
```

example service configuration:
```
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/mnt/d/workspace/ollama_models"
Environment="OLLAMA_DEBUG=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

check config and restart ollama
```bash
cat /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

5. OBS
6. python 3.10 and 3.12 (tested)

## Python Setting
```bash
python -m venv venv
        
source venv/bin/activate # activate virtual environment
        
pip install --upgrade pip
pip install -r requirements.txt

# deactivate
```

## OBS Setting
1. output the game screen via the OBS virtual camera
2. copy .example.env to .env, and edit it
```bash
cp .example.env .env
vim .env
    # do some edition
```
3. execute `python -m src.capture_camera --ocr_url http://<ip>:<port>/ocr/dict --llm_host http://<ip>:<port> --llm_model <model_name>`
4. add the output image `pic/translated_text_overlay.png` as a source in OBS

## TODO
* draw architecture