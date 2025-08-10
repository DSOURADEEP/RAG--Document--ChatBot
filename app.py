import streamlit as st
import os
import tempfile
import pickle
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

# For image processing (OCR)
try:
    import cv2
    import pytesseract
    from PIL import Image
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdf(self, file_path: str) -> List:
        """Load PDF document"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def load_txt(self, file_path: str) -> List:
        """Load TXT document"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    
    def process_image_ocr(self, file_path: str) -> List:
        """Process image using OCR"""
        if not OCR_AVAILABLE:
            st.error("OCR libraries not installed. Please install opencv-python and pytesseract.")
            return []
        
        try:
            # Read image
            image = cv2.imread(file_path)
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(rgb_image)
            
            # Create a document-like structure
            from langchain_core.documents import Document
            return [Document(page_content=text, metadata={"source": file_path})]
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []
    
    def process_documents(self, files) -> List:
        """Process uploaded files and return documents"""
        all_documents = []
        
        for file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            try:
                file_extension = file.name.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    documents = self.load_pdf(tmp_file_path)
                elif file_extension == 'txt':
                    documents = self.load_txt(tmp_file_path)
                elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    documents = self.process_image_ocr(tmp_file_path)
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
                    continue
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(documents)
                all_documents.extend(split_docs)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return all_documents

class SimpleQAChain:
    """Simple QA chain using retrieval and basic text processing"""
    
    def __init__(self, vectorstore, hf_token):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.hf_token = hf_token
        self.chat_history = []
        
        # Try to initialize HF model
        try:
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.1, "max_length": 512}
            )
            self.llm_available = True
        except Exception as e:
            st.warning(f"Could not initialize HuggingFace model: {str(e)}")
            st.info("Using basic text extraction instead of AI generation.")
            self.llm_available = False
    
    def __call__(self, inputs):
        """Process the question and return an answer"""
        question = inputs.get("question", "")
        
        # Retrieve relevant documents
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception as e:
            return {
                "answer": f"Error retrieving documents: {str(e)}",
                "source_documents": []
            }
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "source_documents": []
            }
        
        # If HF model is available, use it
        if self.llm_available:
            try:
                # Create context from retrieved documents
                context = "\n".join([doc.page_content for doc in docs[:3]])
                
                # Create a prompt
                prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
                
                # Get response from LLM
                answer = self.llm(prompt)
                
                return {
                    "answer": answer,
                    "source_documents": docs
                }
                
            except Exception as e:
                st.error(f"Error with HuggingFace model: {str(e)}")
                # Fall back to basic extraction
                pass
        
        # Basic fallback: return most relevant chunks
        answer = f"Based on the documents, here are the most relevant sections:\n\n"
        for i, doc in enumerate(docs[:2], 1):
            answer += f"**Section {i}:**\n{doc.page_content[:300]}...\n\n"
        
        return {
            "answer": answer,
            "source_documents": docs
        }

class RAGChatbot:
    """Main RAG chatbot class"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.embeddings = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize the embedding model"""
        try:
            # Initialize embeddings model (local)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
            return True
            
        except Exception as e:
            st.error(f"Error setting up embeddings: {str(e)}")
            return False
    
    def create_vectorstore(self, documents: List) -> FAISS:
        """Create FAISS vectorstore from documents"""
        try:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return None
    
    def setup_conversation_chain(self, vectorstore: FAISS):
        """Setup the conversational retrieval chain"""
        try:
            # Use our simple QA chain
            conversation_chain = SimpleQAChain(vectorstore, self.hf_token)
            return conversation_chain
            
        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return None

def main():
    """Main application function"""
    
    st.title("🤖 RAG Chatbot - Chat with Your Documents")
    st.markdown("Upload your documents (PDF, TXT, Images) and chat with them using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # Hugging Face token from environment variable
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN', 'your-hugging-face-token-here')
        
        if hf_token == 'your-hugging-face-token-here':
            st.error("🔑 Please set your HUGGINGFACE_API_TOKEN environment variable")
            st.info("You can get a free token from: https://huggingface.co/settings/tokens")
            st.stop()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload PDF, TXT files, or images for OCR processing"
        )
        
        # Process documents button
        if st.button("🔄 Process Documents", disabled=not uploaded_files):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    # Initialize document processor
                    doc_processor = DocumentProcessor()
                    
                    # Process documents
                    documents = doc_processor.process_documents(uploaded_files)
                    
                    if documents:
                        # Initialize chatbot
                        chatbot = RAGChatbot(hf_token)
                        
                        # Create vectorstore
                        vectorstore = chatbot.create_vectorstore(documents)
                        
                        if vectorstore:
                            # Setup conversation chain
                            conversation = chatbot.setup_conversation_chain(vectorstore)
                            
                            if conversation:
                                st.session_state.conversation = conversation
                                st.session_state.vectorstore = vectorstore
                                st.session_state.processed_docs = [f.name for f in uploaded_files]
                                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                            else:
                                st.error("❌ Failed to setup conversation chain")
                        else:
                            st.error("❌ Failed to create vectorstore")
                    else:
                        st.error("❌ No documents were processed successfully")
        
        # Show processed documents
        if st.session_state.processed_docs:
            st.header("📋 Processed Documents")
            for doc in st.session_state.processed_docs:
                st.write(f"✓ {doc}")
            
            if st.button("🗑️ Clear All Documents"):
                st.session_state.conversation = None
                st.session_state.vectorstore = None
                st.session_state.processed_docs = []
                st.session_state.chat_history = []
                st.success("All documents cleared!")
                st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, chat_item in enumerate(st.session_state.chat_history):
                with st.container():
                    # Handle both old format (2 items) and new format (3 items)
                    if len(chat_item) == 2:
                        human_msg, ai_msg = chat_item
                        source_docs = None
                    else:
                        human_msg, ai_msg, source_docs = chat_item
                    
                    st.write("**You:**", human_msg)
                    st.write("**Assistant:**", ai_msg)
                    
                    # Show source documents if available
                    if source_docs:
                        with st.expander(f"📚 Source Documents (Question {i+1})"):
                            for j, doc in enumerate(source_docs[:3]):
                                st.write(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                                st.write(f"**Content:** {doc.page_content[:200]}...")
                                st.write("---")
                    
                    st.divider()
        
        # Chat input
        if st.session_state.conversation:
            # Use form to prevent automatic submission on every rerun
            with st.form(key='chat_form', clear_on_submit=True):
                user_question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What is this document about?",
                    key="question_input"
                )
                submit_button = st.form_submit_button("Ask Question")
            
            # Process the question only when form is submitted
            if submit_button and user_question.strip():
                with st.spinner("Thinking..."):
                    try:
                        # Get response from conversation chain
                        response = st.session_state.conversation({
                            "question": user_question
                        })
                        
                        # Extract answer
                        answer = response.get('answer', 'Sorry, I could not generate an answer.')
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, answer))
                        
                        # Store source documents for this response
                        if 'source_documents' in response and response['source_documents']:
                            # Store source docs with the last chat entry
                            st.session_state.chat_history[-1] = (
                                user_question, 
                                answer, 
                                response['source_documents']
                            )
                        
                        # Rerun to update display
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
        else:
            st.info("👆 Please upload and process documents first to start chatting!")
    
    with col2:
        st.header("ℹ️ Information")
        
        st.info("""
        **How to use:**
        1. Upload your documents (PDF, TXT, or images)
        2. Click 'Process Documents'
        3. Start asking questions!
        
        **Features:**
        - 📄 PDF support
        - 📝 Text file support  
        - 🖼️ Image OCR (if libraries installed)
        - 🔍 Semantic search
        - 💾 Local embeddings
        - 🤖 Hugging Face models
        """)
        
        if st.session_state.vectorstore:
            st.success("✅ Documents loaded and ready!")
            st.metric("Document Chunks", len(st.session_state.vectorstore.docstore._dict))
        else:
            st.warning("⚠️ No documents loaded")

if __name__ == "__main__":
    main()
