# Company Name Standardizer v2

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](TESTING_SUMMARY.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb)](https://reactjs.org/)

A lightweight, fully local web application that groups and normalizes similar company names from CSV input.

> **Status**: ✅ Backend tested and operational. Frontend ready for integration testing.

## Overview

The Company Name Standardizer automates the manual process analysts currently perform to deduplicate and standardize organization names across datasets. All processing happens locally - no data is sent to external servers.

**Key Principles**:
- 🔒 **Privacy First**: All processing is local - no data leaves your machine
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy iteration
- ⚡ **Fast**: Processes ~9,000 names per second
- 📊 **Transparent**: Complete audit trail with confidence scores

## Features

- **CSV Upload**: Upload a list of company names (single column CSV)
- **🆕 Semantic Embeddings**: LLM-powered understanding of company name context and meaning
  - OpenAI API integration (text-embedding-3-small/large)
  - Local privacy mode (sentence-transformers)
  - Fixes false matches from shared words (e.g., "American Express" vs "American Airlines")
- **Automatic Grouping**: Hybrid similarity scoring (fuzzy matching + semantic embeddings)
- **Adaptive Thresholding**: Optional GMM-based data-driven threshold calculation
- **Stratified Sampling**: Unbiased pair sampling for large datasets (3.6x faster)
- **Canonical Name Selection**: Assigns the simplest or most recognizable name as canonical
- **Confidence Scores**: Every mapping includes a confidence score (0-1)
- **Export Results**: Download mappings as CSV or audit log as JSON
- **Audit Log**: Complete audit trail of all processing decisions with reasoning

## Tech Stack

- **Frontend**: React 18 + Vite (fast HMR, modern tooling)
- **Backend**: Python 3.13 + FastAPI (async, high performance)
- **Matching Algorithm**:
  - RapidFuzz 3.14 (fast fuzzy string matching)
  - OpenAI text-embedding-3-small/large (semantic similarity)
  - sentence-transformers (local embeddings for privacy mode)
- **Data Processing**: Pandas 2.3 (CSV handling)
- **Machine Learning**: scikit-learn (GMM adaptive thresholding)

## Architecture

**Modular Design** - Clean separation of concerns:
```
Backend:  Config → Utils → Services → API → Main
Frontend: Constants → Utils → Services → Hooks → Components → App
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed architecture diagrams.

## Quick Start

### Prerequisites

- **Python 3.9+** (tested with 3.13.7)
- **Node.js 18+** (for frontend)
- **npm or yarn** (for frontend)

### Backend Setup (Tested ✅)

```bash
cd backend

# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 3. Install dependencies (verified working)
pip install -r requirements.txt

# 4. Configure environment variables (OPTIONAL - for embeddings)
cp .env.example .env
# Edit .env and add your OpenAI API key (or use Privacy Mode)
# OPENAI_API_KEY=sk-proj-your-key-here

# 5. Create logs directory
mkdir logs

# 6. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend runs on**: http://localhost:8000
**API Docs**: http://localhost:8000/docs

**🆕 For Embeddings Setup**: See [EMBEDDING_SETUP_GUIDE.md](EMBEDDING_SETUP_GUIDE.md) for detailed configuration

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

**Frontend runs on**: http://localhost:3000

### Testing

**Run backend tests** (50/50 passing ✅):
```bash
cd backend
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# All tests
pytest -v

# Specific test suites
pytest tests/test_name_matcher.py -v  # Name matching (12 tests)
pytest tests/test_gmm_threshold_service.py -v  # GMM (9 tests)
pytest tests/test_blocking_service.py -v  # Stratified sampling (29 tests)
```

**Run integration tests** ✅:
```bash
cd backend

# Compare fixed vs adaptive thresholds
python test_adaptive_workflow.py

# Performance validation (100, 300, 560 names)
python test_performance_validation.py
```

**Test Results**: See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for detailed test report.

## Usage

### 1. Try with Sample Data

A sample CSV file is included: [sample_companies.csv](sample_companies.csv)

Contains 34 company names including variants of Apple, Microsoft, Google, Amazon, etc.

### 2. Using the Application

1. **Prepare CSV**: Create a CSV file with a single column of company names
   ```csv
   company_name
   Apple Inc.
   Apple
   Microsoft Corporation
   Microsoft Corp
   Google LLC
   ```

2. **Start Servers**: Run both backend and frontend (in separate terminals)

3. **Upload**: Open http://localhost:3000 and drag/drop your CSV file

4. **🆕 Choose Embedding Quality** (NEW):
   - **Best Quality**: OpenAI 3-large (~90% accuracy, $0.13/1M tokens)
   - **Balanced** (Recommended): OpenAI 3-small (~85% accuracy, $0.02/1M tokens)
   - **Privacy Mode**: Local embeddings (~75% accuracy, no API calls)
   - **Disabled**: Fuzzy matching only (~61% accuracy, fastest)

5. **Choose Threshold Mode**:
   - **Fixed Threshold** (Default): Uses preset 85% threshold
   - **Adaptive GMM**: Data-driven thresholds (better for diverse datasets)

6. **Review**: Examine the standardized mappings with confidence scores

7. **Export**: Download results as CSV (mappings) or JSON (audit log)

### 3. Expected Results

**Example** - Processing these 3 names:
```
Apple Inc.
Apple
Apple Corporation
```

**Output**:
- **Groups Created**: 1 (all mapped to "Apple")
- **Canonical Name**: "Apple" (shortest name selected)
- **Confidence**: 100% for all (obvious matches)
- **Reduction**: 66.7% (3 names → 1 canonical name)

### 4. Advanced: Adaptive GMM Mode with Stratified Sampling

For large datasets or datasets with varying similarity patterns, enable adaptive mode:

**How it Works**:
1. **Stratified Sampling**: Groups similar names into blocks (by first token + phonetics)
2. **Proportional Allocation**: Samples pairs from each block (95% within, 5% across)
3. **GMM Fitting**: Fits 2-component Gaussian Mixture Model on similarity scores
4. **Adaptive Thresholds**: Calculates data-driven T_LOW, S_90, T_HIGH thresholds
5. **Smart Grouping**: Three-zone decision system with phonetic promotion

**Enable via Frontend**: Select "Adaptive GMM-Based" mode before uploading

**Enable via API**:
```bash
POST /api/process?use_adaptive_threshold=true
```

**Benefits**:
- 3.6x faster than old sequential sampling (560 names: 0.585s vs 2.124s)
- 4.3x better match representation (26% within-block vs 6%)
- Unbiased coverage of all names (eliminates first-320-names bias)
- Data-driven thresholds adapt to dataset characteristics

**Configuration**: See [Configuration](#configuration) section for tuning parameters.

## Project Structure

**Modular architecture** for easy iteration:

```
Entity Name Resolution v2/
├── backend/                   # Python FastAPI backend (✅ tested)
│   ├── app/
│   │   ├── main.py           # Application entry point
│   │   ├── models.py         # Pydantic data models
│   │   ├── config/           # Configuration layer
│   │   │   └── settings.py   # Centralized settings
│   │   ├── api/              # HTTP/API layer
│   │   │   └── routes.py     # Endpoint handlers
│   │   ├── services/         # Business logic layer
│   │   │   ├── name_matcher.py  # Core matching algorithm
│   │   │   ├── embedding_service.py  # 🆕 Semantic embeddings (OpenAI + local)
│   │   │   ├── gmm_threshold_service.py  # Adaptive thresholding
│   │   │   └── blocking_service.py  # Stratified sampling
│   │   └── utils/            # Utility layer
│   │       ├── logger.py     # Logging setup
│   │       └── csv_handler.py  # CSV utilities
│   ├── tests/                # Test suite (50/50 passing)
│   │   ├── test_name_matcher.py  # Name matching tests (12)
│   │   ├── test_gmm_threshold_service.py  # GMM tests (9)
│   │   └── test_blocking_service.py  # Stratified sampling tests (29)
│   ├── logs/                 # Application logs
│   ├── test_adaptive_workflow.py  # Compare fixed vs adaptive
│   ├── test_performance_validation.py  # Performance validation
│   └── requirements.txt      # Python dependencies
│
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── config/           # Configuration
│   │   ├── constants/        # App constants
│   │   ├── utils/            # Pure utilities
│   │   ├── hooks/            # Custom React hooks
│   │   ├── services/         # API & export services
│   │   ├── components/       # UI components
│   │   ├── App.jsx           # Main component
│   │   └── main.jsx          # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── sample_companies.csv       # Sample test data
├── README.md                  # This file
├── CLAUDE.md                  # Developer guide
├── PROJECT_STRUCTURE.md       # Architecture diagrams
└── TESTING_SUMMARY.md         # Test results
```

## Configuration

### Backend Configuration

Edit `backend/app/config/settings.py` or use environment variables via `.env`:

```python
# 🆕 Embedding Configuration (NEW)
OPENAI_API_KEY: str = ""                # Your OpenAI API key
DEFAULT_EMBEDDING_MODE: str = "openai-small"  # openai-large, openai-small, local, disabled
EMBEDDING_DIMENSIONS: int = 512         # Reduced from 1536 for speed
WRATIO_WEIGHT: float = 0.40             # Fuzzy matching (typos)
TOKEN_SET_WEIGHT: float = 0.15          # Token overlap (reduced from 0.40)
EMBEDDING_WEIGHT: float = 0.45          # Semantic similarity (NEW)

# Matching algorithm
SIMILARITY_THRESHOLD: float = 85.0      # Adjust grouping strictness (fixed mode)

# Adaptive GMM thresholding (optional)
USE_ADAPTIVE_THRESHOLD: bool = False    # Enable data-driven thresholds
GMM_MIN_SAMPLES: int = 50               # Minimum pairs for GMM
GMM_MAX_PAIRS: int = 50000              # Cap pairwise collection

# Stratified sampling (used in adaptive mode)
BLOCKING_MIN_BLOCK_SIZE: int = 2        # Skip singleton blocks
BLOCKING_MAX_BLOCK_PAIRS: int = 5000    # Cap per-block pairs
SAMPLING_WITHIN_BLOCK_PCT: float = 0.95 # 95% within blocks
SAMPLING_CROSS_BLOCK_PCT: float = 0.05  # 5% cross blocks
SAMPLING_PROPORTIONAL_PCT: float = 0.80 # 80% by block size
SAMPLING_FLOOR_PCT: float = 0.20        # 20% evenly distributed
SAMPLING_RNG_SEED: int = 42             # Fixed seed for reproducibility

# Corporate suffixes to normalize
CORPORATE_SUFFIXES: List[str] = [
    'inc', 'corp', 'llc', 'ltd', ...
]

# Server settings
HOST: str = "0.0.0.0"
PORT: int = 8000
```

See `.env.example` for complete configuration options.

**🆕 Embedding Setup**: See [EMBEDDING_SETUP_GUIDE.md](EMBEDDING_SETUP_GUIDE.md) for detailed instructions.

### Frontend Configuration

Edit `frontend/src/config/api.config.js`:

```javascript
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  TIMEOUT: 60000,
}
```

## Performance

**Tested Performance** (backend):

**Fixed Threshold Mode**:
- **Speed**: ~9,000 names per second
- **Accuracy**: 100% confidence on obvious matches
- **Efficiency**: 66.7% reduction in sample data

**Adaptive GMM Mode** (560 names):
- **Speed**: 0.585s total (3.6x faster than old sequential sampling)
- **Quality**: 4.3x better match representation (26% within-block vs 6%)
- **Threshold Separation**: 50% better (0.021 vs 0.015)
- **Coverage**: Unbiased sampling across all names

**Stratified Sampling Benefits**:
- Eliminates bias where first ~320 names got all comparisons
- Proportional representation from all name blocks
- 5% cross-block sampling captures rare matches
- Fixed RNG seed ensures reproducibility

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive developer guide for working with the codebase
- **🆕 [EMBEDDING_SETUP_GUIDE.md](EMBEDDING_SETUP_GUIDE.md)** - Setup and configuration guide for LLM embeddings
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed architecture diagrams and layer descriptions
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Complete test results and coverage report

## Development

### Running Tests

```bash
cd backend

# All unit tests (50 tests, ~16 seconds)
pytest -v

# Specific test suites
pytest tests/test_name_matcher.py -v       # 12 tests
pytest tests/test_gmm_threshold_service.py -v  # 9 tests
pytest tests/test_blocking_service.py -v   # 29 tests

# Integration tests
python test_adaptive_workflow.py           # Compare modes
python test_performance_validation.py      # Performance metrics

# Linting
black app/ tests/     # Format code
flake8 app/ tests/    # Check style
```

### Adding Features

See [CLAUDE.md](CLAUDE.md) for detailed guidance on:
- Adding new API endpoints
- Modifying the matching algorithm
- Adding export formats
- Customizing normalization rules

## Troubleshooting

**Backend won't start?**
```bash
# Check Python version (need 3.9+)
python --version

# Verify virtual environment is activated
# You should see (venv) in your terminal

# Reinstall dependencies
pip install -r requirements.txt

# Check logs directory exists
mkdir logs
```

**Tests failing?**
```bash
# Make sure you're in the backend directory
cd backend

# Activate virtual environment first
venv\Scripts\activate  # Windows

# Run tests with verbose output
pytest -v
```

**Port already in use?**
```bash
# Change port in backend/app/config/settings.py
PORT: int = 8001  # Use different port
```

See full troubleshooting guide in [CLAUDE.md](CLAUDE.md#troubleshooting).

## Contributing

This is a prototype project designed for easy iteration:
- ✅ Modular architecture for clean separation
- ✅ Comprehensive documentation
- ✅ Tested and verified functionality
- ✅ Clear code patterns and conventions

See [CLAUDE.md](CLAUDE.md) for development guidelines.

## License

This is a prototype project.
