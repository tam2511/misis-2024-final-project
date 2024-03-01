# Misis ML final project

## Install

```
pip install -r requirements.txt
```

## Using

```
python run.py
```

And use it in browser:
```
127.0.0.1/docs
```

## Docker
```
docker build -t misis_project:latest .
```

```
docker run -p 8889:8889 [--gpus=all] misis_project:latest
```
