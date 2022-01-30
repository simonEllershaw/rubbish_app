FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY main.py ./

COPY base_model.pt ./

COPY class_names.json ./

CMD uvicorn --host 0.0.0.0 --port 8080 main:app