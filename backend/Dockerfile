# Deployment image for the API (also used by docker-compose).
# Defaults to port 7860 (Hugging Face Spaces); set PORT to change it.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
USER app
WORKDIR /home/app
ENV PATH="/home/app/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=7860

COPY --chown=app requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

# Bake the two local models into the image so the first request does not wait for a download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=app app ./app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
