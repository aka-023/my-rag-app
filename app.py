import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from document_processor import extract_text_from_file
from rag_engine import setup_rag_pipeline, query_document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0

def reset_session():
    """Reset session state for new document upload"""
    st.session_state.collection = None
    st.session_state.qa_history = []
    st.session_state.document_name = None
    st.session_state.chunk_count = 0

# App Header
st.title("📄 RAG Document Q&A System")
st.markdown("Upload a document and ask questions about its content!")

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ Please enter your API key")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This app uses RAG (Retrieval Augmented Generation) to answer questions based on your uploaded document.")
    
    if st.session_state.document_name:
        st.markdown("---")
        st.markdown("### Current Document")
        st.write(f"📄 **{st.session_state.document_name}**")
        st.write(f"📊 Chunks: {st.session_state.chunk_count}")

# Main content area
st.markdown("---")

# File Upload Section
st.header("1️⃣ Upload Document")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['pdf', 'txt', 'docx'],
    help="Supported formats: PDF, TXT, DOCX (Max 10MB)"
)

if uploaded_file is not None:
    # Check file size (10MB limit)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File too large! Maximum size is 10MB.")
    else:
        # Check if it's a new file
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("🔄 Processing document..."):
                try:
                    # Extract text from file
                    document_text = extract_text_from_file(uploaded_file)
                    
                    if not document_text or len(document_text.strip()) < 50:
                        st.error("❌ Document appears to be empty or too short. Please upload a valid document.")
                    else:
                        # Reset session for new document
                        reset_session()
                        
                        # Setup RAG pipeline
                        collection, chunk_count = setup_rag_pipeline(document_text)
                        
                        # Update session state
                        st.session_state.collection = collection
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.chunk_count = chunk_count
                        
                        st.success(f"✅ Document processed successfully! Split into {chunk_count} chunks.")
                        st.balloons()
                
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

# Display document status
if st.session_state.collection is not None:
    st.info(f"📄 **Current Document:** {st.session_state.document_name} | **Chunks:** {st.session_state.chunk_count}")
    
    # Button to upload new document
    if st.button("🔄 Clear & Upload New Document"):
        reset_session()
        st.rerun()

st.markdown("---")

# Query Section
st.header("2️⃣ Ask Questions")

if st.session_state.collection is None:
    st.warning("⚠️ Please upload a document first before asking questions.")
    st.stop()

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ Please set your Google Gemini API key in the sidebar.")
    st.stop()

# Query input
with st.form(key="query_form"):
    user_query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What is the main topic of this document?"
    )
    submit_button = st.form_submit_button("🔍 Get Answer")

if submit_button and user_query.strip():
    with st.spinner("🤔 Thinking..."):
        try:
            answer = query_document(
                st.session_state.collection,
                user_query,
                os.getenv("GOOGLE_API_KEY")
            )
            
            # Add to history
            st.session_state.qa_history.append({
                "question": user_query,
                "answer": answer
            })
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

# Display Q&A History
if st.session_state.qa_history:
    st.markdown("---")
    st.header("💬 Q&A History")
    
    # Display in reverse order (newest first)
    for i, qa in enumerate(reversed(st.session_state.qa_history)):
        with st.container():
            st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
            st.markdown(f"**A{len(st.session_state.qa_history) - i}:** {qa['answer']}")
            st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.qa_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit ❤️ | Powered by Google Gemini ⭐"
    "</div>",
    unsafe_allow_html=True
)
