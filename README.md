<div align="center">
  <h1>❖ VitalSense</h1>
  <p>Hybrid RAG pipeline (BM25 + FAISS + RRF, cross-encoder reranking) for health report Q&A — RAGAS-evaluated, Dockerized | LangChain · Groq · Streamlit</p>

  <a href="https://github.com/GitSamarth/VitalSense/issues"><img src="https://img.shields.io/github/issues/GitSamarth/VitalSense"></a>
  <a href="https://github.com/GitSamarth/VitalSense/stargazers"><img src="https://img.shields.io/github/stars/GitSamarth/VitalSense"></a>
  <a href="https://github.com/GitSamarth/VitalSense/network/members"><img src="https://img.shields.io/github/forks/GitSamarth/VitalSense"></a>
  <a href="https://github.com/GitSamarth/VitalSense/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</div>

## 🌟 What is VitalSense?

**VitalSense** is a modern, AI-powered health assistant application designed to securely manage, analyze, and gain insights from your medical reports. It leverages advanced Retrieval-Augmented Generation (RAG) capabilities to allow users to intuitively chat with and extract insights from their health documents.

## 🚀 Current Features

*   **Secure Authentication:** User login and signup powered by Supabase.
*   **Medical Report Analysis:** Upload your medical reports (PDFs) and extract structured insights instantly.
*   **Hybrid RAG Retrieval:** Combines BM25 (sparse) and FAISS (dense) retrieval via Reciprocal Rank Fusion, followed by cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for grounded, accurate answers.
*   **Evaluated with RAGAS:** Retrieval quality is measured with a RAGAS-based evaluation harness (faithfulness, answer relevancy) — hybrid retrieval showed a 60%+ faithfulness improvement and 91% relevancy improvement over dense-only retrieval in testing. See `evaluation/`.
*   **Guardrails:** PII redaction (patient name, age, ID fields masked before reaching the LLM) and prompt injection detection — user queries containing injection attempts are rejected outright, while injection patterns embedded in uploaded documents are silently stripped before analysis, with all actions logged for auditability (no sensitive content logged).
*   **Resilient Multi-Model Fallback:** A tiered fallback cascade across multiple Groq models ensures continued availability if a specific model becomes rate-limited or is deprecated by the provider.
*   **Interactive RAG Chat:** Chat directly with your reports to ask follow-up questions, understand complex medical jargon, and track health trends.
*   **Modern UI/UX:** A clean, visually consistent interface with a dark Navy/Teal aesthetic.
*   **Database Management:** Structured storage for user sessions, chats, and uploaded reports.
*   **Containerised Deployment:** Docker support with runtime secret injection — no credentials baked into the image.

## 📊 Retrieval Evaluation (RAGAS)

Retrieval quality was measured with a RAGAS-based evaluation harness (faithfulness, answer relevancy) across three retrieval configurations, using 8 test questions against a sample blood report.

| Metric | Dense-Only | Hybrid (BM25 + RRF) | Hybrid + Reranked |
|---|---|---|---|
| Faithfulness | 0.54 | 0.86 | 0.85 |
| Answer Relevancy | 0.42 | 0.82 | 0.83 |

Hybrid retrieval (BM25 + FAISS dense search fused via Reciprocal Rank Fusion) improved faithfulness by ~60% and answer relevancy by ~91% over dense-only retrieval — dense-only frequently failed to surface the correct chunk even when it was present in retrieval, answering "I don't know" for several lab values despite relevant context being available.

Adding a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on top of hybrid retrieval showed no regression and correctly resolved ordering ambiguity in targeted tests (e.g. promoting the correct chunk to top rank for a query where RRF alone ranked an irrelevant chunk first). On this small, well-structured test corpus, RRF fusion already surfaces the right chunk most of the time; the reranker's benefit is expected to be more pronounced on larger, noisier real-world documents. Full results in `evaluation/`.

## 🛠️ Technologies Used

*   **Frontend & Application Logic:** [Streamlit](https://streamlit.io/)
*   **Authentication & Database:** [Supabase](https://supabase.com/)
*   **LLM Engine:** Groq (`openai/gpt-oss-120b` primary, multi-tier fallback cascade across Production-tier models)
*   **Retrieval:** LangChain, FAISS, BM25 (`rank_bm25`), HuggingFace embeddings, cross-encoder reranking
*   **Evaluation:** RAGAS (faithfulness, answer relevancy)
*   **Containerisation:** Docker (runtime secret injection via `--env-file`)
*   **Styling:** Custom Streamlit CSS & Theming

## 🏗️ Project Structure

```text
VitalSense/
├── .streamlit/             # Streamlit configuration and themes
├── .dockerignore           # Ensures secrets and caches are excluded from the image
├── Dockerfile              # Container definition with runtime secret injection
├── evaluation/             # RAG evaluation scripts and results (RAGAS, hybrid vs baseline)
├── public/                 # Static assets
├── src/                    # Source code
│   ├── auth/               # Authentication logic and session management
│   ├── agents/             # Chat and analysis agents (RAG pipeline, reranking)
│   ├── components/         # Reusable UI components (Sidebar, Footer, Analysis Forms)
│   ├── config/             # App configurations and constants
│   ├── utils/              # Helper functions and validators
│   └── main.py             # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/GitSamarth/VitalSense.git
cd VitalSense
```

### 2. Install dependencies
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.streamlit/secrets.toml` file with your credentials:
```toml
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"
GROQ_API_KEY = "your-groq-api-key"
```

### 4. Run the Application
```bash
streamlit run src/main.py
```

### 5. Run with Docker

Secrets are injected at runtime via an `.env` file — never committed to the repo or baked into the image.

Create a `.env` file in the project root (it is already in `.gitignore` and `.dockerignore`):
```
GROQ_API_KEY=your-groq-api-key
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

> ⚠️ No spaces around `=`. Docker's env-file parser does not strip whitespace.

```bash
# Build
docker build -t vitalsense .

# Run (recommended — uses .env file)
docker run --rm --env-file .env -p 8501:8501 vitalsense

# Run (one-liner alternative — inline env vars)
docker run --rm -p 8501:8501 \
  -e GROQ_API_KEY=your-groq-api-key \
  -e SUPABASE_URL=your-supabase-url \
  -e SUPABASE_KEY=your-supabase-key \
  vitalsense
```

Then open [http://localhost:8501](http://localhost:8501).


## 📈 Project Evolution

### Current Version (VitalSense)
Built on top of the original open-source **[Hia](https://github.com/harshhh28/hia)** project by [Harsh Gajjar](https://github.com/harshhh28). This fork retains the original authentication, PDF analysis, and UI foundation, and extends it with:
- Hybrid retrieval (BM25 + FAISS dense search via Reciprocal Rank Fusion)
- Cross-encoder reranking for improved context relevance
- A RAGAS-based evaluation harness with measured before/after retrieval quality
- PII redaction and prompt injection guardrails, verified against both user-query and document-embedded injection attempts
- A resilient multi-model fallback cascade (recovered from a real incident where the provider deprecated all previously configured models)
- Dockerized deployment with secure runtime secret injection
- Rebranded UI (Navy/Teal theme) and streamlined repository

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.