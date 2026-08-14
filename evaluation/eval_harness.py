"""
Eval harness config — KEEP CONSISTENT across baseline (dense-only) and hybrid runs.
Changing either model or the scoring config invalidates before/after comparisons.

Generation model (agent.model): llama-3.1-8b-instant
Judge model: llama-3.1-8b-instant
(Both switched from llama-3.3-70b-versatile due to 70b daily rate limit
exhaustion during testing — 70b was producing literal rate-limit error 
strings mid-run, which got silently scored as ungrounded answers.)

faithfulness: default strictness (n=1, via agenerate/generate monkey-patch 
              that strips 'n' before it reaches Groq's API)
answer_relevancy: strictness=1 (same n=1 constraint)

Both metrics must complete successfully in the SAME run with real (non-error) 
generated answers to be a valid baseline. Do not mix scores across runs, 
and don't trust a score if the underlying answer text looks like an error message.
"""

import sys
import logging
import json
import streamlit as st
import unittest.mock as mock

# Stub st.secrets
st.secrets = mock.MagicMock()
st.secrets.__getitem__ = mock.Mock(return_value="dummy-key")  # Will be replaced below

# Mock missing vertexai to prevent ragas 0.3.9 crash without installing heavy Google dependencies
sys.modules['langchain_community.chat_models.vertexai'] = mock.MagicMock()

# RAGAS / Langchain imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq

sys.path.insert(0, "src")
from agents.chat_agent import ChatAgent

# --- GROQ/RAGAS CONFIGURATION WARNING ---
# Ragas needs an LLM to evaluate the LLM (LLM-as-a-judge). 
# We wrap Groq in Langchain's ChatGroq for Ragas compatibility.
# You must have GROQ_API_KEY in your environment variables for this script to run.
import os
import tomllib

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    # Try reading from Streamlit secrets file
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomllib.load(f)
            groq_key = secrets.get("GROQ_API_KEY")
    except Exception:
        pass

if not groq_key:
    print("WARNING: GROQ_API_KEY is completely missing. Script will fail.")
    groq_key = "missing-key"

st.secrets.__getitem__ = mock.Mock(return_value=groq_key)

# The sample text used in our earlier tests
SAMPLE = """
BLOOD TEST REPORT
Date: 13/08/2026
Patient: Test Patient | Age: 42 | Sex: Male
Laboratory: VitalSense Diagnostics

=== COMPLETE BLOOD COUNT (CBC) ===
Hemoglobin: 11.2 g/dL  [LOW]  (Reference: 13.5-17.5)
Hematocrit: 34%         [LOW]  (Reference: 41-53%)
Red Blood Cells (RBC): 3.8 M/uL [LOW] (Reference: 4.5-5.9)
White Blood Cells (WBC): 11,200 /uL [HIGH] (Reference: 4,500-11,000)
Neutrophils: 78%        [HIGH] (Reference: 40-70%)
Lymphocytes: 16%        [LOW]  (Reference: 20-40%)
Monocytes: 6%           (Reference: 2-8%)
Platelets: 420,000 /uL  [HIGH] (Reference: 150,000-400,000)
MCV: 72 fL              [LOW]  (Reference: 80-100)
MCH: 22 pg              [LOW]  (Reference: 27-33)
MCHC: 30 g/dL           [LOW]  (Reference: 32-36)
RDW: 17.5%              [HIGH] (Reference: 11.5-14.5)

=== LIPID PANEL ===
Total Cholesterol: 238 mg/dL    [HIGH]  (Reference: <200)
HDL Cholesterol: 38 mg/dL       [LOW]   (Reference: >40 male)
LDL Cholesterol: 162 mg/dL      [HIGH]  (Reference: <130)
Triglycerides: 195 mg/dL        [HIGH]  (Reference: <150)
VLDL Cholesterol: 39 mg/dL      [HIGH]  (Reference: 5-40)
Total Cholesterol/HDL Ratio: 6.3 [HIGH] (Reference: <5.0)

=== LIVER FUNCTION TESTS (LFT) ===
ALT (SGPT): 68 U/L              [HIGH]  (Reference: 7-56)
AST (SGOT): 54 U/L              [HIGH]  (Reference: 10-40)
Alkaline Phosphatase (ALP): 112 U/L     (Reference: 44-147)
Total Bilirubin: 1.4 mg/dL      [HIGH]  (Reference: 0.3-1.2)
Direct Bilirubin: 0.5 mg/dL             (Reference: 0.0-0.3)
Indirect Bilirubin: 0.9 mg/dL           (Reference: 0.1-1.0)
Total Protein: 7.2 g/dL                 (Reference: 6.3-8.2)
Albumin: 4.1 g/dL                       (Reference: 3.5-5.0)
Globulin: 3.1 g/dL                      (Reference: 2.3-3.5)

=== THYROID FUNCTION TESTS (TFT) ===
TSH (Thyroid Stimulating Hormone): 6.8 uIU/mL [HIGH] (Reference: 0.4-4.0)
Free T4 (FT4): 0.7 ng/dL        [LOW]  (Reference: 0.8-1.8)
Free T3 (FT3): 2.4 pg/mL        [LOW]  (Reference: 2.3-4.2)
Anti-TPO Antibodies: 145 IU/mL  [HIGH] (Reference: <35)
"""

QA_PAIRS = [
    ("What is the hemoglobin level?", "11.2 g/dL, low"),
    ("What is the total cholesterol?", "238 mg/dL, high"),
    ("What is the LDL cholesterol?", "162 mg/dL, high"),
    ("What is the ALT level?", "68 U/L, high"),
    ("What is the TSH level?", "6.8 uIU/mL, high"),
    ("What is the Free T4 level?", "0.7 ng/dL, low"),
    ("What is the platelet count?", "420,000 /uL, high"),
    ("What is the AST level?", "54 U/L, high"),
]

def main():
    agent = ChatAgent()
    # OVERRIDE: Also switch the agent's internal model to bypass the 70B rate limit
    agent.model_name = "llama-3.1-8b-instant"
    vectorstore = agent.initialize_vector_store(SAMPLE)
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("Generating answers for evaluation...")
    for q, gt in QA_PAIRS:
        # 1. Manually pull the context our hybrid search returns so Ragas can judge it
        # (We replicate the retrieval block since get_response hides the exact context string from its return type)
        dense_docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(q)
        dense_chunks = [d.page_content for d in dense_docs]

        if hasattr(agent, "retrieve_bm25") and hasattr(agent, "fuse_rrf"):
            # Hybrid path: BM25 + RRF fusion
            bm25_chunks = agent.retrieve_bm25(q, k=5)
            fused = agent.fuse_rrf(dense_chunks, bm25_chunks, k=60)
        else:
            # Dense-only path: just use FAISS results directly
            fused = dense_chunks
        final_contexts = fused[:3]
        
        # 2. Get the actual LLM response
        ans = agent.get_response(q, vectorstore, [])
        
        data["question"].append(q)
        data["answer"].append(ans)
        data["contexts"].append(final_contexts)
        data["ground_truth"].append(gt)
        print(f"Q: {q}\nA: {ans}\n")

    # Build Ragas dataset
    dataset = Dataset.from_dict(data)
    
    # Configure Ragas judge
    # Uses our HF embeddings (all-MiniLM-L6-v2) and Groq Llama 3 8B (to save tokens)
    judge_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)
    
    # MONKEY-PATCH: Groq's API strictly rejects n > 1. Ragas requests n>1 
    # for self-consistency in some metrics (like answer_relevancy). 
    # We strip 'n' out before it hits the Langchain ChatGroq client.
    # We patch the class itself because Pydantic blocks instance-level method replacement.
    original_agenerate = ChatGroq.agenerate
    async def agenerate_wrapper(self, *args, **kwargs):
        kwargs.pop("n", None)
        return await original_agenerate(self, *args, **kwargs)
    ChatGroq.agenerate = agenerate_wrapper

    original_generate = ChatGroq.generate
    def generate_wrapper(self, *args, **kwargs):
        kwargs.pop("n", None)
        return original_generate(self, *args, **kwargs)
    ChatGroq.generate = generate_wrapper
    
    judge_embeddings = agent.embeddings
    
    # Ragas answer_relevancy defaults to generating multiple questions (n>1).
    # Since Groq rejects n>1, set strictness to 1 to force n=1 in Ragas math.
    answer_relevancy.strictness = 1
    
    print("Evaluating with RAGAS (this may take a minute)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    
    # Print results
    print("\n========== EVALUATION RESULTS ==========")
    df = result.to_pandas()
    try:
        print(df[["question", "answer", "faithfulness", "answer_relevancy"]].to_string())
    except KeyError:
        print("Warning: Expected columns not found, dumping raw dataframe:")
        print(df.to_string())
    
    print("\n========== AVERAGE SCORES ==========")
    print(result)
    
    # Save to JSON
    out_file = "scratch/eval_results_baseline_dense_only.json"
    with open(out_file, "w") as f:
        # Convert EvaluationResult -> Pandas DataFrame -> List of Dicts for JSON serialization
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
