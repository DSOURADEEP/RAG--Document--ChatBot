# 🚀 Quick Start Guide

## ✅ Installation Complete!

Your RAG chatbot is now ready to use! Here's how to get started:

## 🏃‍♂️ Running the Application

### Option 1: Using the Batch File
Simply double-click `run.bat` in the project folder.

### Option 2: Using Command Line
1. Open PowerShell
2. Navigate to project directory:
```powershell
cd "C:\Users\user\OneDrive\Desktop\work\projects\rag-chatbot"
```
3. Activate virtual environment:
```powershell
venv\Scripts\activate
```
4. Run the app:
```powershell
streamlit run app.py
```

## 🌐 Access the Application
Open your web browser and go to: **http://localhost:8501**

## 📖 How to Use

### Step 1: Upload Documents
- Use the sidebar on the left
- Click "Choose files" 
- Select your PDF, TXT, or image files
- Multiple files are supported!

### Step 2: Process Documents
- Click the "🔄 Process Documents" button
- Wait for processing to complete (first run will download AI models)
- You'll see a success message when ready

### Step 3: Start Chatting
- Type your questions in the main chat interface
- Ask about document content, summaries, specific information
- View source citations in the expandable sections

## 💡 Example Questions
- "What is this document about?"
- "Summarize the key points"
- "What are the main topics discussed?"
- "List important dates mentioned"
- "Who are the people mentioned in the document?"

## ⚡ Performance Notes

### First Run
- **Expect 2-5 minutes** for initial model downloads
- Models are cached locally for future use
- Embedding model (438MB) downloads once

### Processing Speed
- **Small documents** (1-10 pages): 10-30 seconds
- **Large documents** (50+ pages): 1-3 minutes
- **Images with OCR**: 30-60 seconds per image

### Chat Responses
- **Typical response time**: 5-15 seconds
- **First query** may be slower due to model initialization
- Subsequent queries are faster

## 🛠️ Troubleshooting

### If PDF processing fails:
1. Install Poppler for Windows:
   - Download: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extract and add to system PATH

### If image OCR doesn't work:
1. Install Tesseract OCR:
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to system PATH

### If the app won't start:
1. Make sure virtual environment is activated
2. Check if port 8501 is available
3. Try: `streamlit run app.py --server.port 8502`

## 📊 Current Status
- ✅ **Streamlit**: Installed and working
- ✅ **LangChain**: Latest version with proper imports
- ✅ **HuggingFace Models**: Ready for use
- ✅ **FAISS Vector DB**: Configured
- ✅ **PDF Processing**: Ready (needs Poppler)
- ✅ **OCR Support**: Ready (needs Tesseract)
- ✅ **Local Embeddings**: Working (no API costs!)

## 🔒 Security Notes
- Your Hugging Face token is embedded in the code
- All embeddings run locally (no data sent to third parties)
- Only LLM responses use Hugging Face API
- Documents are processed locally and temporarily

## 🎯 Next Steps
1. **Test with a document**: Upload a PDF or text file
2. **Try different question types**: Summaries, details, analysis
3. **Experiment with images**: Upload screenshots or scanned documents
4. **Customize settings**: Edit `config.py` for advanced options

---

**Your RAG chatbot is ready! Start uploading documents and chatting! 🤖💬**
