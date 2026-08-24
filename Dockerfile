# Build: docker build -t vitalsense .
# Run:   docker run -p 8501:8501 \
#          -e GROQ_API_KEY=... \
#          -e SUPABASE_URL=... \
#          -e SUPABASE_KEY=... \
#          vitalsense

FROM python:3.11-slim

# System deps:
# - libgomp1: required by faiss-cpu (OpenMP runtime for FAISS BLAS operations)
# - libglib2.0-0, libgl1: required by sentence-transformers / torch's image utils
# - build-essential, git: needed by some pip source builds (e.g. tokenizers wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (secrets.toml is excluded via .dockerignore)
COPY src/ ./src/
COPY public/ ./public/
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501

# Secrets are injected at runtime via -e env vars, NOT baked into the image.
# The app uses get_secret() in src/utils/config.py, which checks os.environ first 
# and falls back to st.secrets, enabling secrets to be injected via -e/--env-file 
# at container runtime without being baked into the image.
CMD ["streamlit", "run", "src/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
