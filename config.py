"""
Configuration settings for RAG Chatbot
Modify these settings to customize the behavior
"""

# Hugging Face Models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LANGUAGE_MODEL = "tiiuae/falcon-7b-instruct"

# Alternative models you can try:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller
# LANGUAGE_MODEL = "microsoft/DialoGPT-large"
# LANGUAGE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
# LANGUAGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Document Processing Settings
CHUNK_SIZE = 1000           # Size of text chunks for processing
CHUNK_OVERLAP = 200         # Overlap between chunks
MAX_CHUNKS_PER_QUERY = 3    # Number of relevant chunks to retrieve

# Model Settings
TEMPERATURE = 0.7           # Creativity of responses (0.0 to 1.0)
MAX_LENGTH = 512           # Maximum total length of response
MAX_NEW_TOKENS = 200       # Maximum new tokens to generate

# Streamlit Settings
PAGE_TITLE = "RAG Chatbot"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# File Upload Settings
SUPPORTED_EXTENSIONS = ['pdf', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# OCR Settings (for images)
OCR_LANGUAGE = 'eng'        # Tesseract language
OCR_CONFIG = '--psm 6'      # Page segmentation mode

# Vector Database Settings
VECTOR_SEARCH_K = 3         # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score

# UI Messages
WELCOME_MESSAGE = "Upload your documents (PDF, TXT, Images) and chat with them using AI!"
NO_DOCS_MESSAGE = "👆 Please upload and process documents first to start chatting!"
PROCESSING_MESSAGE = "Processing documents..."
THINKING_MESSAGE = "Thinking..."

# Error Messages
POPPLER_ERROR = "Poppler not found. Please install Poppler for Windows."
OCR_ERROR = "OCR libraries not available. Install opencv-python and pytesseract for image processing."
MODEL_ERROR = "Error loading models. Check your internet connection and Hugging Face token."

# Performance Settings
USE_GPU = False             # Set to True if you have a compatible GPU
DEVICE = 'cpu'             # 'cpu' or 'cuda'
NUM_THREADS = 4            # Number of CPU threads for processing

# Advanced LangChain Settings
VERBOSE_CHAINS = False      # Set to True for debugging
RETURN_SOURCE_DOCUMENTS = True
MEMORY_TYPE = "ConversationBufferMemory"  # Type of conversation memory

# Cache Settings
CACHE_EMBEDDINGS = True     # Cache embeddings to disk for faster reload
CACHE_DIR = ".cache"       # Directory for cached models and embeddings
