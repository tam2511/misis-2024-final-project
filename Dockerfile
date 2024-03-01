#FROM huggingface/transformers-pytorch-gpu:latest
FROM huggingface/transformers-pytorch-cpu:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src src
COPY api.py api.py

EXPOSE 8889

CMD python api.py
