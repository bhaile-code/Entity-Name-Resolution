# Quick Start Guide

Get the Company Name Standardizer running in 5 minutes!

## Prerequisites Check

```bash
# Check Python version (need 3.9+)
python --version

# Check Node.js (need 18+, for frontend)
node --version

# Check npm
npm --version
```

If missing, install:
- **Python**: https://www.python.org/downloads/
- **Node.js**: https://nodejs.org/

## Backend Setup (2 minutes)

### 1. Navigate and Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows**:
```bash
venv\Scripts\activate
```

**macOS/Linux**:
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI (API framework)
- Uvicorn (ASGI server)
- Pandas (CSV processing)
- RapidFuzz (fuzzy matching)
- Pytest (testing)

Wait ~1-2 minutes for installation.

### 4. Create Logs Directory

```bash
mkdir logs
```

### 5. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend is ready!** ✅

Keep this terminal open and running.

**Test it**: Open http://localhost:8000 in your browser
- You should see: `{"status":"ok","message":"Company Name Standardizer API","version":"2.0.0"}`

**API Docs**: http://localhost:8000/docs

## Frontend Setup (2 minutes)

### 1. Open New Terminal

Navigate to frontend directory:

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

Wait ~1-2 minutes for installation.

### 3. Start Development Server

```bash
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

**Frontend is ready!** ✅

## Using the Application

### 1. Open Browser

Navigate to: http://localhost:3000

You should see the Company Name Standardizer interface.

### 2. Try with Sample Data

A sample CSV file is included: `sample_companies.csv` (in project root)

**Option A - Drag and Drop**:
1. Locate `sample_companies.csv` on your computer
2. Drag it to the upload area on the web page

**Option B - Click to Browse**:
1. Click the upload area
2. Select `sample_companies.csv`
3. Click "Open"

### 3. View Results

After upload (~1 second), you'll see:
- **Summary cards**: Total names, groups created, reduction %
- **Mappings table**: Original names → Canonical names with confidence scores
- **Audit log**: Detailed reasoning for each mapping

### 4. Explore Features

**Sort Table**:
- Click column headers to sort (Original Name, Canonical Name, Confidence, Group)

**Filter Results**:
- Type in the filter box to search for specific companies

**Export Data**:
- Click "Download CSV" to get mappings
- Switch to "Audit Log" tab and click "Download Audit Log" for JSON

### 5. Process Your Own Data

**Create a CSV file**:
```csv
company_name
Apple Inc.
Apple
Microsoft Corporation
Microsoft Corp
Google LLC
Google
```

**Requirements**:
- Must be a CSV file (.csv extension)
- Single column (any header name works)
- Max 10MB file size

**Upload and process!**

## Expected Results

Using `sample_companies.csv` (34 company names):
- **Groups Created**: ~13 (from 34 names)
- **Reduction**: ~60%
- **Processing Time**: <1 second
- **Confidence Scores**: 85-100% for obvious matches

## Quick Verification (Optional)

### Run Backend Tests

```bash
# In backend directory, with venv activated
cd backend
venv\Scripts\activate  # Windows
pytest -v
```

**Expected**: `5 passed in ~4s` ✅

### Run Integration Test

```bash
# In backend directory
./venv/Scripts/python test_workflow.py  # Windows
./venv/bin/python test_workflow.py      # macOS/Linux
```

**Expected**: `SUCCESS: All workflow tests passed!` ✅

## Troubleshooting

### Backend Issues

**"Module not found" error?**
```bash
# Make sure virtual environment is activated
# You should see (venv) in prompt
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**"Port 8000 already in use"?**
```bash
# Option 1: Stop other process using port 8000
# Option 2: Use different port
uvicorn app.main:app --reload --port 8001
```

**"logs directory not found"?**
```bash
mkdir logs
```

### Frontend Issues

**"Cannot connect to backend"?**
1. Make sure backend is running (check terminal)
2. Verify backend URL: http://localhost:8000
3. Check `frontend/src/config/api.config.js` - should point to `http://localhost:8000`

**"npm command not found"?**
- Install Node.js: https://nodejs.org/

**Build errors during `npm install`?**
```bash
# Clear cache and retry
npm cache clean --force
npm install
```

### Upload Issues

**"Only CSV files are supported"?**
- Make sure file has `.csv` extension
- Not `.txt`, `.xlsx`, or other formats

**"File size exceeds limit"?**
- Max file size: 10MB
- For larger files, split into multiple CSVs

## Next Steps

### Learn More

- **[README.md](README.md)** - Full project documentation
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for making changes
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Test results and coverage
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Architecture details

### Customize

**Change Similarity Threshold**:
Edit `backend/app/config/settings.py`:
```python
SIMILARITY_THRESHOLD: float = 85.0  # Increase for stricter matching
```

**Add Corporate Suffixes**:
Edit `backend/app/config/settings.py`:
```python
CORPORATE_SUFFIXES: List[str] = [
    'inc', 'corp', 'llc', 'ltd',
    'your-suffix-here',  # Add custom suffixes
]
```

Restart backend after changes.

### Development

**Make Changes**:
1. Backend changes: Auto-reload with `--reload` flag
2. Frontend changes: Hot reload automatic (Vite)

**Add Features**: See [CLAUDE.md](CLAUDE.md) for development patterns

## Stopping the Application

### Stop Backend

In backend terminal: `Ctrl+C`

### Stop Frontend

In frontend terminal: `Ctrl+C`

### Deactivate Virtual Environment

```bash
deactivate
```

## Summary

✅ **Backend**: http://localhost:8000 (API + Processing)
✅ **Frontend**: http://localhost:3000 (User Interface)
✅ **Sample Data**: `sample_companies.csv` (34 companies)
✅ **Tests**: 5/5 passing
✅ **Performance**: ~9,000 names/second

**You're ready to standardize company names!** 🚀

---

**Questions?** See [README.md](README.md) or [CLAUDE.md](CLAUDE.md) for detailed documentation.
