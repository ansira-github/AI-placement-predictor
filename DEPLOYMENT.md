# Deployment Guide for AI Placement Predictor

## ⚠️ Important: Vercel is NOT Suitable

**Vercel is designed for:**
- Static websites (React, Next.js static exports)
- Serverless functions (limited runtime)
- Edge functions

**This Streamlit app requires:**
- Full Python runtime
- Persistent server process
- Ability to load ML models

**❌ Vercel cannot deploy Streamlit apps**

---

## ✅ Recommended Deployment Platforms

### 1. **Streamlit Cloud** (Easiest - Recommended)
**Best for:** Quick deployment, free tier available

**Steps:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `ansira-github/AI-placement-predictor`
5. Set main file path: `frontend/app.py`
6. Click "Deploy"

**Advantages:**
- Free tier available
- Automatic deployments on git push
- Built specifically for Streamlit
- No configuration needed

---

### 2. **Railway** (Good for Beginners)
**Best for:** Easy deployment with good free tier

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python
6. Add build command: `pip install -r requirements.txt`
7. Add start command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`

**Create `railway.json` or use environment variables:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

### 3. **Render** (Good Free Tier)
**Best for:** Simple deployment with good documentation

**Steps:**
1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment:** Python 3

**Create `render.yaml`:**
```yaml
services:
  - type: web
    name: ai-placement-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
```

---

### 4. **Heroku** (Paid, but reliable)
**Note:** Heroku removed free tier, but still reliable

**Steps:**
1. Install Heroku CLI
2. Create `Procfile`:
```
web: streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0
```
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

---

### 5. **AWS/GCP/Azure** (For Production)
**Best for:** Enterprise/production deployments

These require more setup but offer better scalability and control.

---

## 📁 Files Included in Repository

✅ **Included (Required for deployment):**
- `backend/placement_model.pkl` - **CRITICAL** - Loaded via Git LFS
- All Python source files
- `requirements.txt`
- `.gitattributes` - Git LFS configuration

❌ **Excluded (Not needed for deployment):**
- `new.csv` - Training data (not needed at runtime)
- `.venv311/` - Virtual environment (recreated on deployment)
- Other `.pkl` files (if any)

---

## 🔧 Required Files for Deployment

Your repository now includes:
- ✅ Model file (`placement_model.pkl`) via Git LFS
- ✅ All source code
- ✅ Requirements file

**No additional files needed!** The deployment platform will:
1. Install dependencies from `requirements.txt`
2. Run `streamlit run frontend/app.py`
3. The model file will be automatically downloaded via Git LFS

---

## 🚀 Quick Start: Deploy to Streamlit Cloud (Recommended)

1. **Push your code** (already done ✅)
2. **Go to:** https://share.streamlit.io
3. **Sign in** with GitHub
4. **Click "New app"**
5. **Select repository:** `ansira-github/AI-placement-predictor`
6. **Main file path:** `frontend/app.py`
7. **Click "Deploy"**

**That's it!** Your app will be live in ~2 minutes.

---

## 📝 Notes

- **Model file:** Already included via Git LFS (1.6 KB uploaded)
- **CSV files:** Not needed for deployment (only for training)
- **Environment variables:** Not required for basic deployment
- **Port configuration:** Handled automatically by deployment platforms

---

## 🆘 Troubleshooting

**If model file not found:**
- Ensure Git LFS is installed: `git lfs install`
- Check file exists: `git lfs ls-files`
- Re-pull if needed: `git lfs pull`

**If deployment fails:**
- Check `requirements.txt` has all dependencies
- Verify `frontend/app.py` is the correct entry point
- Check platform logs for specific errors

---

## 📚 Additional Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)

