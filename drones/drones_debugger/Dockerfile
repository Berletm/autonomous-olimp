FROM python:3.12-slim

ARG TIMEZONE=Europe/Moscow 

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${TIMEZONE}
ENV QT_DEBUG_PLUGINS=1 
ENV QT_QPA_PLATFORM=xcb

# Установка первых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libgl1 \
    libglx-mesa0 \
    mesa-utils \
    freeglut3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка оставшихся пакетов по группам
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxfixes3 \
    libx11-xcb1 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    tzdata \
    git \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Остальные шаги без изменений
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-deps git+https://github.com/learnsyslab/gym-pybullet-drones.git@8fddcb0288a83d50d11ac907bee0aa1a55f4fa98 \
    && pip install --no-cache-dir -r ./requirements.txt \
    && pip install qasync \
    && pip cache purge

COPY debugger/ ./debugger/
COPY tasks/ ./tasks/

RUN mkdir -p /app/volume

RUN chmod 777 /app/volume

RUN apt-get update && apt-get install -y htop

CMD ["python3", "-u", "./debugger/mainwindow.py"]
