# syntax=docker/dockerfile:1

FROM python:3.11-slim

LABEL description="HBCD ECG task-based QC pipeline"

# Prevent Python from writing .pyc files / buffering stdout (so logs stream live)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# System libraries needed by matplotlib/mne (fonts, freetype, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline script
COPY HBCD_ECG_pipeline_v5.py .
RUN chmod +x HBCD_ECG_pipeline_v5.py

# Expose the script as a normal Linux command "hbcd_ecg_pipeline" on PATH,
# so it can be invoked directly (e.g. by CBRAIN's Boutiques command-line)
# without needing to know it's a Python script or where it lives.
RUN printf '#!/bin/bash\nexec python /app/HBCD_ECG_pipeline_v5.py "$@"\n' \
      > /usr/local/bin/hbcd_ecg_pipeline && \
    chmod +x /usr/local/bin/hbcd_ecg_pipeline

# Default mount points for data (bind-mount your real folders onto these at `docker run`)
RUN mkdir -p /data/input /data/output


ENTRYPOINT []
CMD ["/bin/bash"]