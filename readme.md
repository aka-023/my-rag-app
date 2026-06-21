# 📄 RAG Document Q&A System

A Streamlit-based web application that allows users to upload documents (PDF, DOCX, TXT) and ask questions about their content using Retrieval Augmented Generation (RAG) powered by Google Gemini.

## 🌟 Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Intelligent Q&A**: Ask questions and get accurate answers based on document content
- **RAG Pipeline**: Uses vector embeddings and semantic search for relevant context retrieval
- **Session-Based**: No permanent storage - each session is independent
- **Chat History**: Track all your questions and answers in the current session
- **User-Friendly Interface**: Clean and intuitive Streamlit UI

## 🏗️ Architecture

```
User Upload Document
    ↓
Text Extraction (PDF/DOCX/TXT)
    ↓
Text Chunking (with overlap)
    ↓
Vector Embeddings (ChromaDB)
    ↓
User Asks Question
    ↓
Semantic Search (Retrieve relevant chunks)
    ↓
Prompt Construction (Context + Query)
    ↓
LLM Generation (Google Gemini)
    ↓
Answer Displayed
```

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get it here](https://aistudio.google.com/app/apikey))

## 🚀 Installation

1. **Clone the repository** (or create project folder):
```bash
mkdir rag-document-qa
cd rag-document-qa
```

2. **Create virtual environment**:
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# make .env file in your project root and add your Google Gemini API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

## 🎮 Usage

1. **Run the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** (usually auto-opens at `http://localhost:8501`)

3. **Upload a document**:
   - Click "Browse files" or drag & drop
   - Supported formats: PDF, DOCX, TXT
   - Max size: 10MB

5. **Ask questions**:
   - Type your question in the text area
   - Click "Get Answer"
   - View answer and chat history

6. **Upload new document**:
   - Click "Clear & Upload New Document"
   - Previous session data is cleared

## 📁 Project Structure

```
rag-document-qa/
├── app.py                  # Streamlit web interface
├── rag_engine.py          # Core RAG logic (chunking, retrieval, generation)
├── document_processor.py  # File parsing for PDF, DOCX, TXT
├── requirements.txt       # Python dependencies
├── .env                  # Your actual API key (don't commit!)
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables (.env)

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

### RAG Parameters (in `rag_engine.py`)

- `CHUNK_SIZE`: Size of text chunks (default: 500 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50 characters)
- `N_RESULTS`: Number of chunks to retrieve (default: 3)
- `LLM_MODEL`: Gemini model to use (default: "gemini-2.0-flash-exp")

## 🔒 Security & Privacy

- **No Permanent Storage**: Vector database is in-memory (ephemeral)
- **Session-Based**: Each user session is independent
- **API Key Security**: Store in .env file, never commit to version control
- **Local Processing**: Document processing happens on your machine

## 📝 Example Questions

After uploading a document, try asking:

- "What is the main topic of this document?"
- "Summarize the key points"
- "What does the document say about [specific topic]?"
- "Are there any specific dates or numbers mentioned?"
- "What are the conclusions or recommendations?"

## 🛠️ Troubleshooting

### API Key Issues
- Ensure your API key is correctly set in `.env` or sidebar
- Verify the key is active at [Google AI Studio](https://aistudio.google.com)

### File Upload Errors
- Check file size (max 10MB)
- Ensure file is not corrupted
- Try re-saving the document

### "Document too short" Error
- Document must have at least 50 characters
- Check if PDF text extraction worked (some PDFs are image-based)

### ChromaDB Errors
- Restart the app
- Clear browser cache
- Check Python version (3.8+ required)

## 🚀 Advanced Usage

### Using Different LLM Models

Edit `rag_engine.py`:
```python
LLM_MODEL = "gemini-1.5-pro"  # or other available models
```

### Adjusting Chunk Size

Edit `rag_engine.py`:
```python
CHUNK_SIZE = 1000  # Larger chunks for more context
CHUNK_OVERLAP = 100  # More overlap for better continuity
```

### Increasing Retrieved Chunks

Edit `rag_engine.py`:
```python
N_RESULTS = 5  # Retrieve more chunks for complex questions
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **ChromaDB**: For the vector database
- **Google Gemini**: For the LLM capabilities
- **PyPDF & python-docx**: For document processing

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review [Streamlit documentation](https://docs.streamlit.io)
3. Check [Google Gemini API docs](https://ai.google.dev/docs)

---

Built with ❤️ using Python, Streamlit, and Google Gemini
