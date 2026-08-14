import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging
import re
import os
from utils.config import get_secret

logger = logging.getLogger("chat_agent")


class ChatAgent:
    def __init__(self):
        # tokenizer_kwargs: use_fast=True prevents the slow-tokenizer fallback path
        # that also triggers huggingface_hub's chat-template metadata fetch (404 on
        # embedding-only models like all-MiniLM-L6-v2 under huggingface_hub>=0.35).
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"tokenizer_kwargs": {"use_fast": True}},
        )
        # smaller chunks improve retrieval precision for narrow factual lookups 
        # (e.g. specific lab values) at k=3, since large 1000-char chunks risk 
        # diluting relevance with unrelated report sections in the same chunk.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        )
        self.client = Groq(api_key=get_secret("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"
        # BM25 index and raw chunks — populated in initialize_vector_store.
        # Parallel retrieval path; not yet fused into LLM context (RRF next step).
        self.bm25_index = None
        self.bm25_chunks = []
        # Cross-encoder reranker — loaded once at construction, used in rerank().
        # Isolated from get_response until rerank() is verified standalone.
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.debug("CrossEncoder model loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")

    def initialize_vector_store(self, text_content):
        """Create FAISS vector store and BM25 index from text content."""
        if not text_content or text_content.strip() == "":
            # Create a minimal vector store with a placeholder
            text_content = "No report context available."

        texts = self.text_splitter.split_text(text_content)
        if not texts:
            # If splitting results in empty list, add at least one text
            texts = [text_content]

        # Dense retrieval: FAISS vector store (unchanged)
        vectorstore = FAISS.from_texts(texts, self.embeddings)

        # Sparse retrieval: BM25 over the same chunks.
        # Tokenise by stripping non-alphanumeric chars then lowercasing+splitting.
        # Plain whitespace split kept 'hemoglobin:' as a single token, causing
        # zero BM25 scores when the query contained bare 'hemoglobin'.
        tokenised_chunks = [self._tokenize(chunk) for chunk in texts]
        self.bm25_index = BM25Okapi(tokenised_chunks)
        self.bm25_chunks = texts
        logger.debug("BM25 index built over %d chunks.", len(texts))

        return vectorstore

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase + strip punctuation before whitespace split.

        Bare whitespace split keeps trailing punctuation attached to tokens
        (e.g. 'hemoglobin:'), causing BM25 misses when queries contain
        the bare form ('hemoglobin'). Stripping non-alphanumeric chars first
        normalises both sides of the match.
        """
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def retrieve_bm25(self, query: str, k: int = 5) -> list[str]:
        """Return the top-k chunks from BM25 for a given query.

        Used both in isolation (testing) and internally by get_response as one
        of the two ranked lists passed to fuse_rrf.

        Args:
            query: The raw query string.
            k:     Number of top BM25 results to return (default 5).

        Returns:
            Ordered list of chunk strings (highest BM25 score first).
        """
        if self.bm25_index is None or not self.bm25_chunks:
            logger.warning("retrieve_bm25 called before initialize_vector_store.")
            return []

        tokenised_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenised_query)

        # Pair each chunk with its score, sort descending, return top-k text
        ranked = sorted(
            zip(scores, self.bm25_chunks), key=lambda x: x[0], reverse=True
        )
        top_k = [chunk for _, chunk in ranked[:k]]
        logger.debug(
            "BM25 top-%d for query %r:\n%s",
            k, query,
            "\n---\n".join(f"[score={s:.4f}] {c[:120]}" for s, c in ranked[:k]),
        )
        return top_k

    @staticmethod
    def fuse_rrf(
        dense_results: list[str],
        bm25_results: list[str],
        k: int = 60,
        top_n: int | None = 8,
    ) -> list[str]:
        """Combine FAISS dense and BM25 sparse ranked lists via Reciprocal Rank Fusion.

        RRF score per chunk:  score(d) = sum_over_lists( 1 / (k + rank_in_list) )
        where rank is 1-indexed.  Chunks present in only one list still accumulate
        a score from that list.  k=60 is the standard RRF smoothing constant.

        NOTE on zero-score BM25 chunks: BM25Okapi ranks ALL corpus chunks even when
        their score is 0.0 (no query token match). Those chunks are still present in
        bm25_results at their natural rank position and contribute a small RRF score
        ( 1/(60+rank) ).  We do NOT treat 0-score BM25 chunks as absent.

        Args:
            dense_results: Chunks ordered by FAISS similarity (index 0 = most similar).
            bm25_results:  Chunks ordered by BM25 score (index 0 = highest score).
            k:             RRF smoothing constant (default 60).
            top_n:         Cap on returned chunks.  Default 8 gives the cross-encoder
                           reranker a meaningful pool without blowing out context.  Pass
                           None to return all unique fused chunks.

        Returns:
            Up to top_n unique chunks sorted by combined RRF score, descending.
        """
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (k + rank)

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Debug: show per-chunk rank sources and final RRF score
        dense_rank_map = {c: r for r, c in enumerate(dense_results, start=1)}
        bm25_rank_map  = {c: r for r, c in enumerate(bm25_results, start=1)}
        log_lines = []
        for chunk, score in fused:
            dr = dense_rank_map.get(chunk, "-")
            br = bm25_rank_map.get(chunk, "-")
            log_lines.append(
                f"[rrf={score:.5f} | dense_rank={dr} | bm25_rank={br}] "
                f"{chunk[:100].strip()!r}"
            )
        logger.debug("RRF fused ranking:\n%s", "\n".join(log_lines))

        all_chunks = [chunk for chunk, _ in fused]
        return all_chunks[:top_n] if top_n is not None else all_chunks

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Rerank chunks using the cross-encoder and return the top_k best.

        Scores every (query, chunk) pair with the cross-encoder in a single
        batched call, then sorts descending by score.  Mirrors the standalone
        pattern used for retrieve_bm25 — not yet wired into get_response.

        Args:
            query:  The user query (or contextualized form of it).
            chunks: Candidate chunks to rerank, e.g. from fuse_rrf().
            top_k:  Number of top-scoring chunks to return (default 3).

        Returns:
            List of up to top_k (chunk, score) tuples sorted by cross-encoder score, descending.
        """
        if not chunks:
            logger.warning("rerank() called with empty chunk list — returning []")
            return []

        pairs = [(query, chunk) for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)   # ndarray, one score per pair

        scored = sorted(
            zip(scores, chunks), key=lambda x: x[0], reverse=True
        )

        logger.debug(
            "Cross-encoder scores for query %r:\n%s",
            query,
            "\n".join(
                f"  [ce_score={s:.4f}] {c[:120].strip()!r}"
                for s, c in scored
            ),
        )

        return [(chunk, float(score)) for score, chunk in scored[:top_k]]

    def _format_chat_history(self, chat_history):
        """Format chat history for Groq API."""
        messages = []
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    def _contextualize_query(self, query, chat_history):
        """Reformulate query considering chat history."""
        if not chat_history:
            return query

        # Build context from recent chat history
        recent_history = chat_history[-4:]  # Last 2 exchanges
        history_text = "\n".join(
            [
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ]
        )

        contextualize_prompt = f"""Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{history_text}

Latest User Question: {query}

Standalone Question:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You reformulate questions to be standalone.",
                    },
                    {"role": "user", "content": contextualize_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query  # Fallback to original query

    def get_response(self, query, vectorstore, chat_history=None):
        """Get response using RAG."""
        if chat_history is None:
            chat_history = []

        # 1. Contextualize query based on chat history
        contextualized_query = self._contextualize_query(query, chat_history)

        # 2. Hybrid retrieval: FAISS dense + BM25 sparse, fused via RRF
        try:
            # 2a. Dense: FAISS top-5 (wider than final k=3 to give RRF more to work with)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            dense_docs = retriever.invoke(contextualized_query)
            dense_chunks = [doc.page_content for doc in dense_docs]

            # 2b. Sparse: BM25 top-5 over the same corpus
            bm25_chunks = self.retrieve_bm25(contextualized_query, k=5)

            # 2c. Fuse both ranked lists with RRF (k=60 smoothing constant, widened to top 8)
            fused_chunks = self.fuse_rrf(dense_chunks, bm25_chunks, k=60, top_n=8)

            # 2d. Rerank the fused pool using the cross-encoder
            reranked_results = self.rerank(contextualized_query, fused_chunks, top_k=3)
            reranked_chunks = [chunk for chunk, _ in reranked_results]

            # Debug log the final selected context chunks and their scores
            if reranked_results:
                logger.debug("Final LLM Context (Cross-Encoder top-3):")
                for i, (chunk, score) in enumerate(reranked_results, 1):
                    logger.debug(
                        "--- FINAL CHUNK %d (ce_score=%.4f) ---\n%s\n----------------------------------------",
                        i, score, chunk
                    )

            # 2e. Take top-3 reranked chunks as the LLM context window
            context = "\n\n".join(reranked_chunks)

            # Guard against placeholder-only context
            if context.strip() == "No report context available.":
                context = ""
        except Exception as e:
            # Fallback: no context rather than crash
            logger.error(f"Retrieval failed: {e}")
            context = ""

        # 3. Build prompt with context and chat history
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
        )

        # Format messages for Groq API
        messages = [{"role": "system", "content": qa_system_prompt}]

        # Add chat history
        if chat_history:
            formatted_history = self._format_chat_history(
                chat_history[-6:]
            )  # Last 3 exchanges
            messages.extend(formatted_history)

        # Add context and current query
        if (
            context
            and context.strip()
            and context.strip() != "No report context available."
        ):
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            # No report context available, rely on chat history only
            user_message = f"Question: {query}\n\nNote: No report context is available. Please answer based on the chat history."
        messages.append({"role": "user", "content": user_message})

        # 4. Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
