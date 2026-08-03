# Scoring service image. Build with the artifact folder copied in, or mount it:
#   docker build -t automl-service --build-arg EXTRA_MODELS="xgboost lightgbm" .
#   docker run -p 8000:8000 -v /path/to/artifact:/app/artifact automl-service
#
# EXTRA_MODELS installs the champion's model libraries (see the artifact's own
# requirements.txt) — without them a GBM-champion artifact cannot unpickle.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
ARG EXTRA_MODELS=""
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ -n "$EXTRA_MODELS" ]; then pip install --no-cache-dir $EXTRA_MODELS; fi
COPY app.py .
RUN useradd --create-home --uid 1000 scorer
USER scorer
ENV MODEL_DIR=/app/artifact
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
