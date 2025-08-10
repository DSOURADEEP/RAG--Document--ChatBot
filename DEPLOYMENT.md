# 🚀 Free Deployment Guide for RAG Chatbot

## 🆓 Best Free Hosting Options

### 1. **Streamlit Community Cloud** ⭐ (RECOMMENDED)
- **Cost**: Completely FREE
- **Best for**: Streamlit apps (perfect for our project!)
- **Limits**: 1GB RAM, 1 CPU core, 500MB storage
- **Pros**: Zero config, automatic HTTPS, custom domains

### 2. **Render**
- **Cost**: FREE tier available
- **Best for**: Full-stack applications
- **Limits**: 512MB RAM, sleeps after 15min inactivity
- **Pros**: Auto-deploy from GitHub, HTTPS included

### 3. **Railway**
- **Cost**: $5 free credits monthly (usually enough)
- **Best for**: Any application type
- **Limits**: Based on usage, auto-sleep
- **Pros**: Great performance, easy setup

### 4. **Google Cloud Run**
- **Cost**: FREE tier (2M requests/month)
- **Best for**: Containerized apps
- **Limits**: 1000 concurrent requests, 15min timeout
- **Pros**: Scales to zero, pay-per-use

---

## 🎯 Method 1: Streamlit Community Cloud (EASIEST & FREE)

### Step 1: Prepare Your Repository
Your GitHub repo is already ready! ✅

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign up/Login** with your GitHub account
3. **Click "New app"**
4. **Repository**: `DSOURADEEP/RAG--Document--ChatBot`
5. **Branch**: `main`
6. **Main file path**: `app.py`
7. **App URL**: Choose your custom URL

### Step 3: Add Environment Variables
In the **Advanced settings** section:
```
HUGGINGFACE_API_TOKEN = hf_your_actual_token_here
```

### Step 4: Deploy
Click **"Deploy!"** - Your app will be live in 2-3 minutes!

### 🎉 Result
Your app will be available at: `https://your-app-name.streamlit.app/`

---

## 🎯 Method 2: Render (FREE TIER)

### Step 1: Create Render Account
1. Go to: https://render.com/
2. Sign up with GitHub account

### Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repo: `RAG--Document--ChatBot`
3. **Name**: `rag-document-chatbot`
4. **Environment**: `Python 3`
5. **Build Command**: `pip install -r requirements.txt`
6. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Step 3: Environment Variables
Add in the **Environment** section:
```
HUGGINGFACE_API_TOKEN = hf_your_actual_token_here
PORT = 10000
```

### Step 4: Deploy
Click **"Create Web Service"** - Deploy in 3-5 minutes!

---

## 🎯 Method 3: Railway (FREE CREDITS)

### Step 1: Create Railway Account
1. Go to: https://railway.app/
2. Sign up with GitHub

### Step 2: Deploy from GitHub
1. Click **"Start a New Project"**
2. **"Deploy from GitHub repo"**
3. Select: `DSOURADEEP/RAG--Document--ChatBot`
4. Railway auto-detects it's a Python app

### Step 3: Add Environment Variables
1. Go to **Variables** tab
2. Add: `HUGGINGFACE_API_TOKEN = hf_your_actual_token_here`

### Step 4: Configure Start Command
1. Go to **Settings**
2. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🎯 Method 4: Google Cloud Run (ADVANCED)

### Step 1: Create Dockerfile (Already provided!)
```dockerfile
FROM python:3.10-slim
# ... (already in your project)
```

### Step 2: Deploy
1. **Install Google Cloud CLI**
2. **Login**: `gcloud auth login`
3. **Set project**: `gcloud config set project YOUR_PROJECT_ID`
4. **Deploy**:
```bash
gcloud run deploy rag-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HUGGINGFACE_API_TOKEN=your_token
```

---

## 🏆 RECOMMENDED: Streamlit Community Cloud

### ✅ Why Streamlit Cloud is Best for Your Project:

1. **🆓 Completely Free** - No credit card required
2. **⚡ Zero Configuration** - Works out of the box
3. **🔄 Auto-Deploy** - Updates when you push to GitHub
4. **🛡️ Secure** - Built-in secrets management
5. **🎯 Perfect for Streamlit** - Optimized for your framework

### 📋 Quick Deployment Checklist:

- [x] GitHub repository ready
- [x] All secrets removed from code  
- [x] Environment variables supported
- [x] Requirements.txt complete
- [ ] Get Hugging Face token ready
- [ ] Sign up for Streamlit Cloud
- [ ] Deploy in 5 minutes!

---

## 🔧 Troubleshooting Free Deployments

### Memory Issues (Common on Free Tiers)
**Problem**: App crashes due to memory limits

**Solution**: 
```python
# Add to app.py - reduce model size
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
```

### Slow Loading (First Request)
**Problem**: Cold starts on free tiers

**Solution**: Apps "wake up" after first request (normal behavior)

### API Rate Limits
**Problem**: Hugging Face API limits

**Solution**: App includes fallback text extraction

---

## 🎯 Next Steps

1. **Choose Streamlit Cloud** (recommended)
2. **Follow the Step-by-Step guide above**
3. **Your RAG chatbot will be live and FREE!**

### 🌟 Your Deployed App Will Have:
- ✅ Public URL anyone can access
- ✅ File upload functionality  
- ✅ Document processing (PDF, TXT, images)
- ✅ AI-powered chat responses
- ✅ Source document citations
- ✅ Completely free hosting!

**Ready to deploy? Choose Method 1 (Streamlit Cloud) for the easiest experience!** 🚀
