import re
import uuid
from typing import List, Tuple

import chromadb
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Configuration
LLM_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 800          # target characters per chunk
CHUNK_OVERLAP = 100       # characters of overlap carried into the next chunk
N_RESULTS = 3             # number of chunks to retrieve
EMBED_BATCH_SIZE = 50     # texts per embedding API call (keeps request count low on free tier)


def split_into_sentences(text: str) -> List[str]:
    """Naive but effective sentence splitter using punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def smart_text_splitter(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks along sentence boundaries so chunks never get
    cut off mid-sentence. Each chunk grows until it would exceed chunk_size,
    then a new chunk starts, carrying the last few sentences forward as
    overlap so context isn't lost at chunk edges.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence) + 1

        if current_length + sentence_length > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Carry trailing sentences forward as overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_sentences):
                if overlap_length + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s) + 1
            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def embed_documents(chunks: List[str], api_key: str) -> List[List[float]]:
    """Embed document chunks in batches using Gemini's retrieval-document embedding."""
    try:
        client = genai.Client(api_key=api_key)
        embeddings: List[List[float]] = []

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.extend([e.values for e in result.embeddings])

        return embeddings

    except APIError as e:
        raise Exception(f"Gemini Embedding Error: {str(e)}")


def embed_query(query: str, api_key: str) -> List[float]:
    """Embed a user query using Gemini's retrieval-query embedding."""
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    except APIError as e:
        raise Exception(f"Gemini Embedding Error: {str(e)}")


def setup_rag_pipeline(document_text: str, api_key: str) -> Tuple[chromadb.Collection, int]:
    """
    Set up the RAG pipeline: chunk the text, embed chunks with Gemini,
    and store them in a ChromaDB collection.
    """
    chunks = smart_text_splitter(document_text)

    if not chunks:
        raise ValueError("No chunks created from document. Document may be too short.")

    chunk_embeddings = embed_documents(chunks, api_key)

    client = chromadb.Client()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.get_or_create_collection(name=collection_name)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return collection, len(chunks)


def retrieve_context(collection: chromadb.Collection, query: str, api_key: str, n_results: int = N_RESULTS) -> List[str]:
    """Retrieve relevant context chunks for a given query using Gemini query embeddings."""
    query_embedding = embed_query(query, api_key)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_chunks = results['documents'][0] if results['documents'] else []
    return retrieved_chunks


def construct_rag_prompt(query: str, context: List[str]) -> str:
    """Construct the final prompt with context and query for the LLM."""
    context_str = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context)])

    prompt = f"""You are a helpful Q&A assistant. Your task is to answer questions based ONLY on the provided context from a document.

IMPORTANT INSTRUCTIONS:
- Use ONLY the context provided below to answer the question
- If the answer is not contained in the context, clearly state: "I cannot find this information in the provided document."
- Do not use any external knowledge or make assumptions
- Be concise and direct in your answer
- If the context provides partial information, mention what is available

CONTEXT:
{context_str}

USER QUESTION: {query}

ANSWER:"""

    return prompt


def generate_answer(prompt: str, api_key: str) -> str:
    """Generate answer using Google Gemini API."""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        return response.text.strip()

    except APIError as e:
        raise Exception(f"Gemini API Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")


def query_document(collection: chromadb.Collection, query: str, api_key: str) -> str:
    """Main function to query the document using the RAG pipeline."""
    context_chunks = retrieve_context(collection, query, api_key)

    if not context_chunks:
        return "I couldn't find any relevant information in the document to answer your question."

    final_prompt = construct_rag_prompt(query, context_chunks)
    answer = generate_answer(final_prompt, api_key)

    return answer