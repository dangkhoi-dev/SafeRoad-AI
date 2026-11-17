# Bản demo chạy thật của SafeRoad AI cho Hugging Face Spaces (SDK: docker, CPU).
# Dựng tại chỗ:  docker build -t saferoad . && docker run -p 7860:7860 saferoad
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 SAFEROAD_MAX_SECONDS=30 SAFEROAD_SAMPLE_PRESET=street

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Hugging Face chạy container bằng user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# PyTorch bản CPU trước, để pip không kéo bản CUDA vài GB.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install -e .

COPY --chown=user configs ./configs
COPY --chown=user scripts/download_assets.py ./scripts/download_assets.py
COPY --chown=user models ./models
COPY --chown=user data/outputs ./data/outputs
COPY --chown=user docs/assets ./docs/assets
# Thư mục data/samples luôn có (ít nhất .gitkeep); video mẫu là tuỳ chọn.
COPY --chown=user data/samples ./data/samples
RUN python scripts/download_assets.py && mkdir -p data/jobs && chown -R user /home/user/app

USER user
EXPOSE 7860
# Video mẫu (nếu có) nằm ở data/samples/demo_sample.mp4.
CMD ["sh", "-c", "python -m saferoad serve --host 0.0.0.0 --port 7860 --demo \
     --results data/outputs/results.json --video docs/assets/demo_overlay.mp4 \
     $( [ -f data/samples/demo_sample.mp4 ] && echo --sample-video data/samples/demo_sample.mp4 )"]
