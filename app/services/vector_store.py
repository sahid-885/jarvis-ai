"""
VECTOR STORE SERVICE MODULE
===========================

This service builds and queries the FAISS vector index used for context retrieval.
Learning data (database/learning_data/*.txt) and past chats (database/chats_data/*.json)
are loaded at startup, split into chunks, embedded with HuggingFace, and stored in FAISS.
When the user asks a question we embed it and retrieve the k most similar chunks; only
those chunks are sent to the LLM, so token usage is bounded.

LIFECYCLE:
  - create_vector_store(): Load all .txt and .json, chunk, embed, build FAISS, save to disk.
    Called once at startup. Restart the server after adding new .txt files so they are included.
  - get_retriever(k): Return a retriever that fetches k nearest chunks for a query string.
  - save_vector_store(): Write the current FAISS index to database/vector_store/ (called after create).

Embeddings run locally (sentence-transformers); no extra API key. Groq and Realtime services
call get_retriever() for every request to get context.

─────────────────────────────────────────────────────────────────────────────
WHAT IS A VECTOR STORE AND WHY DO WE NEED ONE?
─────────────────────────────────────────────────────────────────────────────
Large Language Models (LLMs) like Groq-hosted Llama have a limited context window
(the maximum number of tokens they can read at once). We can't paste our entire
knowledge base into every prompt — it would exceed the limit and waste money/time.

A **vector store** solves this with *semantic search*:
  1. Offline: convert every chunk of text into a numeric vector (embedding) that
     captures its *meaning*, then store all vectors in an index.
  2. At query time: convert the user's question into a vector, then find the
     chunks whose vectors are closest (most similar in meaning) to the question.
  3. Send only those top-k chunks to the LLM as context. This is called
     **Retrieval-Augmented Generation (RAG)**.

Result: the LLM gets relevant context without seeing the entire database.

─────────────────────────────────────────────────────────────────────────────
HOW FAISS WORKS (HIGH LEVEL)
─────────────────────────────────────────────────────────────────────────────
FAISS (Facebook AI Similarity Search) is a library optimized for finding the
nearest neighbors of a vector in a large collection:
  - It stores all chunk embeddings in an efficient in-memory index.
  - When given a query vector, it computes distances (e.g. L2 or cosine) to
    every stored vector and returns the top-k closest matches.
  - For small-to-medium datasets (<100k vectors) a flat (brute-force) index is
    used; for larger datasets FAISS supports approximate methods (IVF, HNSW).
  - The index can be saved to disk and loaded back, so we only rebuild when data changes.

─────────────────────────────────────────────────────────────────────────────
HOW HUGGINGFACE EMBEDDINGS WORK
─────────────────────────────────────────────────────────────────────────────
HuggingFaceEmbeddings uses a pre-trained transformer model (e.g. all-MiniLM-L6-v2)
to convert a piece of text into a fixed-size numeric vector (e.g. 384 dimensions).
  - Texts with similar *meaning* get vectors that are close together in the
    384-dimensional space, even if the exact words differ.
  - Example: "How is the weather?" and "What's the temperature outside?" would
    have very similar vectors because they mean nearly the same thing.
  - The model runs **locally on CPU** — no API key or internet needed after the
    first download of the model weights.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

# ─── LangChain components ───────────────────────────────────────────────────
# RecursiveCharacterTextSplitter: breaks long documents into smaller overlapping chunks.
# HuggingFaceEmbeddings:          wraps a sentence-transformers model to produce vectors.
# FAISS:                          LangChain wrapper around Facebook's FAISS library.
# Document:                       simple container holding page_content (str) + metadata (dict).
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,   # Path to database/learning_data/ (contains .txt knowledge files)
    CHATS_DATA_DIR,      # Path to database/chats_data/ (contains .json past-conversation files)
    VECTOR_STORE_DIR,    # Path to database/vector_store/ (where the FAISS index is saved on disk)
    EMBEDDING_MODEL,     # Name of the HuggingFace model, e.g. "all-MiniLM-L6-v2"
    CHUNK_SIZE,          # Maximum characters per text chunk (e.g. 1000)
    CHUNK_OVERLAP,       # Characters of overlap between consecutive chunks (e.g. 200)
)


logger = logging.getLogger("J.A.R.V.I.S")


# =============================================================================
# VECTOR STORE SERVICE CLASS
# =============================================================================

class VectorStoreService:
    """
    Builds a FAISS index from learning_data .txt files and chats_data .json files,
    and provides a retriever to fetch the k most relevant chunks for a query.

    TYPICAL USAGE (inside app startup):
        vs = VectorStoreService()
        vs.create_vector_store()          # load data → chunk → embed → index → save
        retriever = vs.get_retriever(k=5) # later, per-request, fetch top-5 chunks

    WHY A CLASS?
    Encapsulating the embeddings model, text splitter, FAISS index, and retriever
    cache in one object makes it easy to pass around and to mock in tests.
    """

    def __init__(self):
        """
        Create the embedding model (local) and text splitter; vector_store is set
        later in create_vector_store().

        Nothing is loaded from disk yet — this just prepares the tools we'll need.
        """
        # ── Embedding model ─────────────────────────────────────────────────
        # HuggingFaceEmbeddings downloads the model on first run, then caches it.
        # model_kwargs={"device": "cpu"} forces CPU inference.  If you have a GPU
        # and want faster embedding, change to "cuda".
        # The embedding model converts text → vector (e.g. 384-dimensional float array).
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # ── Text splitter ───────────────────────────────────────────────────
        # Documents can be thousands of characters long, but embeddings work best
        # on short passages.  The splitter breaks long text into chunks of at most
        # `chunk_size` characters, with `chunk_overlap` characters shared between
        # consecutive chunks.
        #
        # WHY OVERLAP?
        # If a relevant sentence falls right at the boundary between two chunks,
        # overlap ensures it appears in *both* chunks, so the retriever can still
        # find it regardless of which chunk it searches.
        #
        # "Recursive" means it tries to split on paragraph breaks first, then
        # sentences, then words — preserving natural boundaries when possible.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # The FAISS index itself — None until create_vector_store() is called.
        self.vector_store: Optional[FAISS] = None

        # Retriever cache: maps k (number of results) → retriever object.
        # We cache so we don't recreate a retriever for the same k on every request.
        self._retriever_cache: dict = {}

    # -------------------------------------------------------------------------
    # LOAD DOCUMENTS FROM DISK
    # -------------------------------------------------------------------------

    def load_learning_data(self) -> List[Document]:
        """
        Read all .txt files in database/learning_data/ and return one Document per file.

        HOW IT WORKS:
          1. Glob for *.txt files in sorted order (deterministic ordering).
          2. Read each file's text content.
          3. Wrap it in a LangChain Document(page_content=..., metadata={"source": filename}).
             - page_content: the raw text that will later be chunked and embedded.
             - metadata["source"]: the filename, preserved so we can trace which file
               a retrieved chunk came from (useful for debugging).
          4. Skip empty files silently; log warnings for read errors.

        WHY .txt FILES?
        Plain text is the simplest format for knowledge.  You can drop any
        information you want JARVIS to know about into a .txt file, restart the
        server, and it will be indexed and retrievable.

        Returns:
            List of Document objects — one per successfully loaded .txt file.
        """
        documents = []
        # sorted() ensures consistent ordering across OS file-system implementations.
        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # skip empty files
                        # Create a Document: LangChain's standard container for text + metadata.
                        documents.append(Document(page_content=content, metadata={"source": str(file_path.name)}))
                        logger.info("[VECTOR] Loaded learning data: %s (%d chars)", file_path.name, len(content))
            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)
        logger.info("[VECTOR] Total learning data files loaded: %d", len(documents))
        return documents

    def load_chat_history(self) -> List[Document]:
        """
        Load all .json files in database/chats_data/ and convert each conversation
        into a single Document.

        HOW IT WORKS:
          1. Each .json file is a saved chat session: {"messages": [{"role": ..., "content": ...}, ...]}.
          2. We iterate over the messages and format them as readable lines:
               "User: <message>"    for role == "user"
               "Assistant: <message>" for role == "assistant"
          3. All lines are joined with newlines into one string — this becomes the
             Document's page_content.
          4. metadata["source"] is set to "chat_<filename>" so we can tell chat-derived
             chunks apart from learning-data chunks.

        WHY INDEX PAST CHATS?
        If a user asked "What is Kubernetes?" last week and got a great answer,
        indexing that exchange means JARVIS can reuse the context in future
        conversations without calling the LLM again for the same topic.

        Returns:
            List of Document objects — one per successfully loaded .json chat file.
        """
        documents = []
        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)  # parse JSON into a Python dict

                messages = chat_data.get("messages", [])

                # Format each message as "User: ..." or "Assistant: ..." and join them.
                chat_content = "\n".join([
                    f"User: {msg.get('content', '')}" if msg.get('role') == 'user'
                    else f"Assistant: {msg.get('content', '')}"
                    for msg in messages
                ])

                if chat_content.strip():  # skip if no actual content
                    documents.append(Document(page_content=chat_content, metadata={"source": f"chat_{file_path.stem}"}))
                    logger.info("[VECTOR] Loaded chat history: %s (%d messages)", file_path.name, len(messages))
            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)
        logger.info("[VECTOR] Total chat history files loaded: %d", len(documents))
        return documents

    # -------------------------------------------------------------------------
    # BUILD AND SAVE FAISS INDEX
    # -------------------------------------------------------------------------

    def create_vector_store(self) -> FAISS:
        """
        Full pipeline: Load → Chunk → Embed → Index → Save.

        STEP-BY-STEP:
          1. LOAD: Read all .txt learning files and .json chat files from disk.
          2. CHUNK: Split each Document into smaller overlapping pieces using the
             text splitter. This converts N large documents into many small chunks
             (e.g. 10 documents → 200 chunks).
          3. EMBED: Each chunk's text is passed through the HuggingFace embedding
             model, producing a numeric vector per chunk.
          4. INDEX: FAISS.from_documents() stores all vectors in a FAISS index,
             which supports fast nearest-neighbor search.
          5. SAVE: The index is written to disk so it *could* be reloaded later
             without re-embedding (though currently we rebuild every startup).

        If there are NO documents at all (empty database/), we create a tiny
        placeholder index with a single dummy entry. This prevents crashes in
        get_retriever() — the retriever will simply return "No data available yet."

        Called once during server startup (in app/main.py lifespan).

        Returns:
            The built FAISS vector store instance.
        """
        # Step 1: Load raw documents from both sources.
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()
        all_documents = learning_docs + chat_docs
        logger.info("[VECTOR] Total documents to index: %d (learning: %d, chat: %d)",
                     len(all_documents), len(learning_docs), len(chat_docs))

        if not all_documents:
            # No data yet — create a minimal index so the rest of the app doesn't crash.
            self.vector_store = FAISS.from_texts(["No data available yet."], self.embeddings)
            logger.info("[VECTOR] No documents found, created placeholder index")
        else:
            # Step 2: Split documents into smaller chunks.
            # Each chunk is still a Document object, but with shorter page_content.
            chunks = self.text_splitter.split_documents(all_documents)
            logger.info("[VECTOR] Split into %d chunks (chunk_size=%d, overlap=%d)",
                         len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

            # Steps 3 & 4: Embed all chunks and build the FAISS index.
            # from_documents() calls self.embeddings.embed_documents() internally
            # for every chunk, then inserts the resulting vectors into the index.
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("[VECTOR] FAISS index built successfully with %d vectors", len(chunks))

        # Clear the retriever cache because the index just changed.
        self._retriever_cache.clear()
        # Step 5: Persist the index to disk.
        self.save_vector_store()
        return self.vector_store

    def save_vector_store(self):
        """
        Write the current FAISS index to database/vector_store/.

        FAISS save_local() writes two files:
          - index.faiss: the actual vector index (binary).
          - index.pkl: the metadata (Document objects) associated with each vector.

        On error we only log — a save failure shouldn't crash the server, and the
        in-memory index still works fine for the current session.
        """
        if self.vector_store:
            try:
                self.vector_store.save_local(str(VECTOR_STORE_DIR))
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    # -------------------------------------------------------------------------
    # RETRIEVER FOR CONTEXT
    # -------------------------------------------------------------------------

    def get_retriever(self, k: int = 10):
        """
        Return a retriever that returns the k most similar chunks for a query string.

        WHAT IS A RETRIEVER?
        A retriever is an object with an .invoke(query) method. When called:
          1. It embeds the query string into a vector (using the same embedding model).
          2. It searches the FAISS index for the k nearest vectors.
          3. It returns the corresponding Document objects (text + metadata).

        The chat services call retriever.invoke(user_message) and then inject the
        returned chunks into the LLM prompt as context.

        CACHING:
        We cache retriever objects by k-value in self._retriever_cache. If multiple
        requests ask for k=10, they share the same retriever object instead of
        creating a new one each time.  The cache is cleared whenever the vector
        store is rebuilt (in create_vector_store) so stale retrievers are never used.

        Args:
            k: Number of nearest chunks to retrieve (default 10).

        Returns:
            A LangChain VectorStoreRetriever instance.

        Raises:
            RuntimeError: If the vector store hasn't been initialized yet.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. This should not happen.")
        if k not in self._retriever_cache:
            # as_retriever() wraps the FAISS index in a Retriever interface.
            # search_kwargs={"k": k} tells it how many results to return.
            self._retriever_cache[k] = self.vector_store.as_retriever(search_kwargs={"k": k})
        return self._retriever_cache[k]
