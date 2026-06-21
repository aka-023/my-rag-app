import streamlit as st
from document_processor import extract_text_from_file
from rag_engine import setup_rag_pipeline, query_document
import os
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI - Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Professional Minimalist UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Smooth sidebar styling */
    .stSidebar {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Custom section labels for sidebar */
    .sidebar-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6c757d;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Clean chat message tweaks */
    .stChatMessage {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Hide default header/footer menu fluff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0

def reset_session():
    st.session_state.collection = None
    st.session_state.messages = []
    st.session_state.document_name = None
    st.session_state.chunk_count = 0

# --- SIDEBAR PANEL (Controls & Upload) ---
with st.sidebar:
    st.markdown("<h1>📄 <span style='color:black;'>DocuMind AI</span></h1>", unsafe_allow_html=True)  
    st.markdown("<p style='color:black; font-size: 0.95em; opacity: 0.8;'>Intelligent RAG-powered document exploration.</p>", unsafe_allow_html=True)  
    
    # 1. API Key Configuration
    st.markdown('<div class="sidebar-label">API Configuration</div>', unsafe_allow_html=True)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("Gemini API Key Connected", icon="⚡")
    else:
        api_key_input = st.text_input("Enter Google API Key", type="password")
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            st.rerun()
        else:
            st.error("API key missing. Provide it above or via environment variables.")
            st.stop()

    st.markdown("---")

    # 2. Document Upload Section
    st.markdown('<div class="sidebar-label">Document Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload workspace source",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX (Max 10MB)",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit.")
        else:
            # Check if this is a newly uploaded file
            if st.session_state.document_name != uploaded_file.name:
                with st.spinner("Analyzing and embedding document..."):
                    try:
                        document_text = extract_text_from_file(uploaded_file)
                        
                        if not document_text or len(document_text.strip()) < 50:
                            st.error("Document content insufficient or parsing failed.")
                        else:
                            # Fresh file means fresh session context
                            reset_session()
                            collection, chunk_count = setup_rag_pipeline(document_text, os.getenv("GOOGLE_API_KEY"))
                            st.session_state.collection = collection
                            st.session_state.document_name = uploaded_file.name
                            st.session_state.chunk_count = chunk_count
                            st.rerun()
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")

    # 3. Active Document Status & Control
    if st.session_state.document_name:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">Active Knowledge Source</div>', unsafe_allow_html=True)
        st.info(f"**{st.session_state.document_name}**\n\n🧩 {st.session_state.chunk_count} semantic chunks indexed.")
        
        if st.button("Unload Document", type="secondary", use_container_width=True):
            reset_session()
            st.rerun()
            
    # Clear conversation thread separately if wanted
    if len(st.session_state.messages) > 0:
        if st.button("Clear Chat History", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# --- MAIN CHAT INTERFACE ---
st.header("Document Discussion Workspace", divider="gray")

# Direct user to upload file if empty state
if st.session_state.collection is None:
    st.info("Welcome! Please upload a PDF, TXT, or DOCX document in the sidebar panel to start analyzing.", icon="ℹ️")
else:
    # Display the current file being queried contextually at the top
    st.caption(f"Conversing with: `{st.session_state.document_name}`")

    # Display conversational thread history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle real-time user chat inputs
    if user_query := st.chat_input("Ask anything about the active document..."):
        
        # 1. Immediately display user prompt in chat window
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # 2. Generate and stream/display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document context & composing answer..."):
                try:
                    answer = query_document(
                        st.session_state.collection,
                        user_query,
                        os.getenv("GOOGLE_API_KEY")
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error yielding answer: {str(e)}")