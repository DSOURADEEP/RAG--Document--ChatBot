import streamlit as st
import os
import tempfile
from typing import List
import warnings
warnings.filterwarnings("ignore")

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

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
    """Lightweight document processor for deployment"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for memory efficiency
            chunk_overlap=100,
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
    """Lightweight QA chain for deployment"""
    
    def __init__(self, vectorstore, hf_token):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Fewer docs
        self.hf_token = hf_token
        
        # Try to initialize HF model with lighter config
        try:
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",  # Smaller model for memory
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.1, "max_length": 256}  # Shorter responses
            )
            self.llm_available = True
        except Exception as e:
            st.warning("Using basic text extraction mode")
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
                # Create shorter context for memory efficiency
                context = "\n".join([doc.page_content[:500] for doc in docs[:2]])
                
                prompt = f"Based on this context, answer briefly: {context}\n\nQuestion: {question}\nAnswer:"
                
                answer = self.llm(prompt)
                
                return {
                    "answer": answer,
                    "source_documents": docs
                }
                
            except Exception as e:
                st.error(f"Error with AI model: {str(e)}")
        
        # Basic fallback: return most relevant text
        answer = "Based on the documents:\n\n"
        for i, doc in enumerate(docs[:2], 1):
            answer += f"**Section {i}:** {doc.page_content[:200]}...\n\n"
        
        return {
            "answer": answer,
            "source_documents": docs
        }

class RAGChatbot:
    """Lightweight RAG chatbot for deployment"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.embeddings = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize lightweight embedding model"""
        try:
            # Use smaller embedding model for free tiers
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
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
            conversation_chain = SimpleQAChain(vectorstore, self.hf_token)
            return conversation_chain
        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return None

def main():
    """Main application function"""
    
    st.title("🤖 RAG Chatbot - Chat with Your Documents")
    st.markdown("Upload your documents (PDF, TXT) and chat with them using AI!")
    
    # Get token from environment or Streamlit secrets
    hf_token = os.getenv('HUGGINGFACE_API_TOKEN') or st.secrets.get("HUGGINGFACE_API_TOKEN", "")
    
    if not hf_token:
        st.error("🔑 Please configure your HUGGINGFACE_API_TOKEN")
        st.info("Get a free token from: https://huggingface.co/settings/tokens")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader - PDF and TXT only for deployment
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files"
        )
        
        # Process documents button
        if st.button("🔄 Process Documents", disabled=not uploaded_files):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    doc_processor = DocumentProcessor()
                    documents = doc_processor.process_documents(uploaded_files)
                    
                    if documents:
                        chatbot = RAGChatbot(hf_token)
                        vectorstore = chatbot.create_vectorstore(documents)
                        
                        if vectorstore:
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
                    if len(chat_item) == 2:
                        human_msg, ai_msg = chat_item
                        source_docs = None
                    else:
                        human_msg, ai_msg, source_docs = chat_item
                    
                    st.write("**You:**", human_msg)
                    st.write("**Assistant:**", ai_msg)
                    
                    if source_docs:
                        with st.expander(f"📚 Sources"):
                            for j, doc in enumerate(source_docs[:2]):
                                st.write(f"**Source {j+1}:** {doc.page_content[:150]}...")
                    
                    st.divider()
        
        # Chat input
        if st.session_state.conversation:
            with st.form(key='chat_form', clear_on_submit=True):
                user_question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What is this document about?",
                    key="question_input"
                )
                submit_button = st.form_submit_button("Ask Question")
            
            if submit_button and user_question.strip():
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation({
                            "question": user_question
                        })
                        
                        answer = response.get('answer', 'Sorry, I could not generate an answer.')
                        st.session_state.chat_history.append((user_question, answer))
                        
                        if 'source_documents' in response and response['source_documents']:
                            st.session_state.chat_history[-1] = (
                                user_question, 
                                answer, 
                                response['source_documents']
                            )
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
        else:
            st.info("👆 Please upload and process documents first to start chatting!")
    
    with col2:
        st.header("ℹ️ Information")
        
        st.info("""
        **How to use:**
        1. Upload documents (PDF, TXT)
        2. Click 'Process Documents'
        3. Ask questions!
        
        **Features:**
        - 📄 PDF support
        - 📝 Text file support  
        - 🔍 Semantic search
        - 💾 Local embeddings
        - 🤖 HuggingFace models
        """)
        
        if st.session_state.vectorstore:
            st.success("✅ Documents loaded!")
            try:
                chunk_count = len(st.session_state.vectorstore.docstore._dict)
                st.metric("Document Chunks", chunk_count)
            except:
                st.metric("Status", "Ready")
        else:
            st.warning("⚠️ No documents loaded")

if __name__ == "__main__":
    main()
