import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import logging
import re
import os


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
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.model_name = "llama-3.3-70b-versatile"
        # BM25 index and raw chunks — populated in initialize_vector_store.
        # Parallel retrieval path; not yet fused into LLM context (RRF next step).
        self.bm25_index = None
        self.bm25_chunks = []

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
        logging.debug("BM25 index built over %d chunks.", len(texts))

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

        This is a standalone retrieval path — not used in get_response yet.
        Call this directly to verify BM25 is surfacing sensible results before
        wiring it into the hybrid/RRF fusion step.

        Args:
            query: The raw query string.
            k:     Number of top BM25 results to return (default 5).

        Returns:
            Ordered list of chunk strings (highest BM25 score first).
        """
        if self.bm25_index is None or not self.bm25_chunks:
            logging.warning("retrieve_bm25 called before initialize_vector_store.")
            return []

        tokenised_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenised_query)

        # Pair each chunk with its score, sort descending, return top-k text
        ranked = sorted(
            zip(scores, self.bm25_chunks), key=lambda x: x[0], reverse=True
        )
        top_k = [chunk for _, chunk in ranked[:k]]
        logging.debug(
            "BM25 top-%d for query %r:\n%s",
            k, query,
            "\n---\n".join(f"[score={s:.4f}] {c[:120]}" for s, c in ranked[:k]),
        )
        return top_k

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

        # 2. Retrieve relevant documents
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(contextualized_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # If context is just placeholder text, set to empty
            if context.strip() == "No report context available.":
                context = ""
        except Exception:
            # If retrieval fails, proceed without context
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
