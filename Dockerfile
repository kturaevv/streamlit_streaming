FROM python:3.10-slim-buster as base

WORKDIR /usr/src/app

ENV \
	# Turns off writing .pyc files
	PYTHONDONTWRITEBYTECODE=1 \
	# Seems to speed things up
	PYTHONUNBUFFERED=1 \
	# Default VENV usage
	PATH="/venv/bin:${PATH}" \
	VIRTUAL_ENV="/venv" \
	PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Create virtual env to store dependencies
RUN python3 -m venv $VIRTUAL_ENV

# Project dependencies
RUN apt-get update && \
	apt-get -y install libpq5

### ---
FROM base as builder

# Dependencies for psycopg2
RUN apt-get update && \
	apt-get -y install libpq-dev gcc

COPY requirements.lock.txt .
RUN pip3 install -U pip && \
	pip3 install setuptools && \
	pip3 install -r requirements.txt \
    pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu117
	
FROM builder as final

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]