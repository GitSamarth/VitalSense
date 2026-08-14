import sys, logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

# Stub st.secrets so it doesn't blow up outside Streamlit
import unittest.mock as mock
import streamlit as st
st.secrets = mock.MagicMock()
st.secrets.__getitem__ = mock.Mock(return_value="dummy-key")

from agents.chat_agent import ChatAgent

# ~1800 chars across 4 clinical sections — forces 3-4 chunks at chunk_size=512/overlap=50
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

print(f"\nSample length: {len(SAMPLE)} chars\n")

agent = ChatAgent()
agent.initialize_vector_store(SAMPLE)

print(f"\nTotal BM25 chunks created: {len(agent.bm25_chunks)}")
print("\n" + "="*60)
print("Chunk boundaries (first 80 chars each):")
for i, chunk in enumerate(agent.bm25_chunks):
    print(f"  [{i}] {chunk[:80].strip()!r}")
print("="*60)

# --- Test 1: hemoglobin query → should rank CBC chunk highest ---
print("\n\n>>> TEST 1: 'what is the hemoglobin level'")
results = agent.retrieve_bm25("what is the hemoglobin level", k=4)
for i, chunk in enumerate(results, 1):
    print(f"\n--- BM25 result {i} ---\n{chunk[:200]}")

# --- Test 2: cholesterol query → should rank lipid chunk highest ---
print("\n\n>>> TEST 2: 'total cholesterol and LDL values'")
results = agent.retrieve_bm25("total cholesterol and LDL values", k=4)
for i, chunk in enumerate(results, 1):
    print(f"\n--- BM25 result {i} ---\n{chunk[:200]}")

# --- Test 3: thyroid query → should rank TFT chunk highest ---
print("\n\n>>> TEST 3: 'TSH and thyroid hormone levels'")
results = agent.retrieve_bm25("TSH and thyroid hormone levels", k=4)
for i, chunk in enumerate(results, 1):
    print(f"\n--- BM25 result {i} ---\n{chunk[:200]}")

print("\n========== TEST COMPLETE ==========")
