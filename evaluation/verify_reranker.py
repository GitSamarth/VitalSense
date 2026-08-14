import sys, logging, json
sys.path.insert(0, "src")

# We'll use a clean basicConfig just for this script
logging.basicConfig(level=logging.WARNING, format="%(name)s - %(message)s")
# Suppress everything except chat_agent
logging.getLogger("chat_agent").setLevel(logging.WARNING)

import unittest.mock as mock
import streamlit as st
st.secrets = mock.MagicMock()
st.secrets.__getitem__ = mock.Mock(return_value="dummy-key")

from agents.chat_agent import ChatAgent

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

Clinical Notes: Pattern consistent with iron-deficiency anemia and subclinical
hypothyroidism. Elevated liver enzymes and lipid profile require follow-up.
Recommend dietary review and repeat CBC in 6 weeks.
"""

agent = ChatAgent()
vectorstore = agent.initialize_vector_store(SAMPLE)

queries = [
    "What is my hemoglobin level?",
    "What is my cholesterol and LDL?",
    "What is my TSH and thyroid levels?",
    "What is my ALT level?"
]

results = []

for query in queries:
    print(f"\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)

    # 1. Retrieve Dense & Sparse
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    dense_docs = retriever.invoke(query)
    dense_chunks = [doc.page_content for doc in dense_docs]
    bm25_chunks = agent.retrieve_bm25(query, k=5)
    
    # 2. Fuse
    fused_pool = agent.fuse_rrf(dense_chunks, bm25_chunks, k=60, top_n=8)
    
    print("\n--- PRE-RERANK (RRF Fused top-8) ---")
    for i, chunk in enumerate(fused_pool, 1):
        print(f"{i}. {chunk[:100].strip().replace(chr(10), ' ')}...")
        
    # 3. Rerank
    reranked_results = agent.rerank(query, fused_pool, top_k=3)
    
    print("\n--- POST-RERANK (Cross-Encoder top-3) ---")
    for i, (chunk, score) in enumerate(reranked_results, 1):
        print(f"{i}. [score={score:.4f}] {chunk[:100].strip().replace(chr(10), ' ')}...")
        
    # Save to results list
    reranked_chunks = [chunk for chunk, _ in reranked_results]
    ce_scores = [score for _, score in reranked_results]
    
    results.append({
        "query": query,
        "pre_rerank_top3": fused_pool[:3],
        "post_rerank_top3": reranked_chunks,
        "ce_scores": ce_scores
    })

# Write to JSON
output_path = "evaluation/verify_reranker_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved structured results to {output_path}")
