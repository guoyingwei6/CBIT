FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    GBC_ASSET_DIR=/app/assets

WORKDIR /app
COPY services/gbc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/gbc/app ./app
COPY app.bak/public/gbc/gbc-reference.json /tmp/gbc-assets/
COPY app.bak/public/gbc/gbc-reference-*.bin.gz /tmp/gbc-assets/
COPY services/gbc/scripts/prepare_reference.py /tmp/prepare_reference.py
RUN python /tmp/prepare_reference.py /tmp/gbc-assets /app/assets \
    && rm -rf /tmp/gbc-assets /tmp/prepare_reference.py

USER 65532:65532
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
