# Hugging Face Spaces (Docker SDK) image for the Streamlit demo.
# Local test:  docker build -t mic-demo . && docker run -p 7860:7860 \
#              -e MIC_HF_MODEL_REPO=your-hf-username/multilingual-image-captioning mic-demo
FROM python:3.11-slim-bookworm

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MIC_CHECKPOINT_DIR=/home/user/app/checkpoints
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the 84 MB ImageNet weights into the image so a cold start does not download them.
RUN python -c "import tensorflow as tf; tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')"

COPY --chown=user . .

EXPOSE 7860
# XSRF protection is off because Streamlit's upload check fails inside the Spaces iframe
# (HTTP 403 on upload). The app keeps no accounts or private state for XSRF to protect.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
