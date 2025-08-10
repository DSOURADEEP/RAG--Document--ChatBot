@echo off
echo 🤖 Starting RAG Chatbot...
echo ================================

:: Check if virtual environment exists
if not exist "venv\" (
    echo ❌ Virtual environment not found!
    echo Please run setup.py first:
    echo    python setup.py
    pause
    exit /b 1
)

:: Activate virtual environment and run app
echo 🚀 Activating virtual environment and starting application...
call venv\Scripts\activate && streamlit run app.py

pause
