# Based on the official Python 3.10.2 slim image
FROM --platform=linux/arm64 python:3.10.2-slim-bullseye AS execute

# Use Tsinghua mirror for faster Python package installation in China (optional)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONHASHSEED 0

# Set working directory
WORKDIR /codegendata

# Update and install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libkrb5-dev \
    build-essential \
    python3-dev \
    python3-tk \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    tk \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python packages
COPY requirements.txt /codegendata/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY human-eval /codegendata/human-eval
COPY evaluate_functional_correctness.sh /codegendata/

# Install the human-eval package
RUN pip install -e /codegendata/human-eval

# Set the entrypoint
ENTRYPOINT ["/codegendata/evaluate_functional_correctness.sh"]
CMD []