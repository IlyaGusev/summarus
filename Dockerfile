FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "run.py"]
