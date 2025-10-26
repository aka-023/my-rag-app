import chromadb
import uuid
from typing import List, Tuple
from google import genai
from google.genai.errors import APIError

# Configuration
LLM_MODEL = "gemini-2.0-flash-exp"  # Updated to latest model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
N_RESULTS = 3  # Number of chunks to retrieve


def simple_text_splitter(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better context retention.
    
    Args:
        text: Input text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Get chunk with overlap
        end_pos = current_pos + chunk_size
        chunk = text[current_pos:end_pos].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move position forward (accounting for overlap)
        current_pos += chunk_size - overlap
        
        # Avoid infinite loop on last small chunk
        if current_pos >= len(text):
            break
    
    return chunks


def setup_rag_pipeline(document_text: str) -> Tuple[chromadb.Collection, int]:
    """
    Set up the RAG pipeline by chunking text and storing in vector database.
    
    Args:
        document_text: The full text of the document
    
    Returns:
        Tuple of (ChromaDB collection, number of chunks)
    """
    # 1. Chunking
    chunks = simple_text_splitter(document_text)
    
    if not chunks:
        raise ValueError("No chunks created from document. Document may be too short.")
    
    # 2. Initialize Chroma client (ephemeral/in-memory)
    client = chromadb.Client()
    
    # Create unique collection name
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.get_or_create_collection(name=collection_name)
    
    # 3. Prepare data for ChromaDB
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    
    # 4. Add documents to collection (ChromaDB handles embeddings automatically)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection, len(chunks)


def retrieve_context(collection: chromadb.Collection, query: str, n_results: int = N_RESULTS) -> List[str]:
    """
    Retrieve relevant context chunks for a given query.
    
    Args:
        collection: ChromaDB collection
        query: User's question
        n_results: Number of chunks to retrieve
    
    Returns:
        List of relevant text chunks
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Extract the retrieved text chunks
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    
    return retrieved_chunks


def construct_rag_prompt(query: str, context: List[str]) -> str:
    """
    Construct the final prompt with context and query for the LLM.
    
    Args:
        query: User's question
        context: Retrieved context chunks
    
    Returns:
        Formatted prompt string
    """
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
    """
    Generate answer using Google Gemini API.
    
    Args:
        prompt: The complete prompt with context and query
        api_key: Google Gemini API key
    
    Returns:
        Generated answer text
    """
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=api_key)
        
        # Generate response
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
    """
    Main function to query the document using RAG pipeline.
    
    Args:
        collection: ChromaDB collection with document chunks
        query: User's question
        api_key: Google Gemini API key
    
    Returns:
        Generated answer
    """
    # 1. Retrieve relevant context
    context_chunks = retrieve_context(collection, query)
    
    if not context_chunks:
        return "I couldn't find any relevant information in the document to answer your question."
    
    # 2. Construct prompt
    final_prompt = construct_rag_prompt(query, context_chunks)
    
    # 3. Generate answer
    answer = generate_answer(final_prompt, api_key)
    
    return answer