# Company Name Standardizer v2

[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)](TESTING_SUMMARY.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb)](https://reactjs.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991)](https://openai.com/)

An intelligent, AI-enhanced web application that groups and normalizes similar company names using hybrid fuzzy matching, semantic embeddings, and optional LLM assessment for borderline cases.

> **Status**: ✅ Backend fully tested (76/76 tests passing). Frontend integration complete. Ready for production use.

## Overview

The Company Name Standardizer automates the manual process analysts currently perform to deduplicate and standardize organization names across datasets. It combines traditional fuzzy matching with modern AI capabilities to achieve high accuracy while maintaining transparency and explainability.

**Key Principles**:
- 🔒 **Privacy First**: All processing is local - no data leaves your machine (unless you opt-in to OpenAI API)
- 🤖 **AI-Enhanced**: Optional GPT-4o-mini assessment for ambiguous matches with anti-hallucination guardrails
- 🎯 **Hybrid Intelligence**: Combines fuzzy matching (typos) + semantic embeddings (meaning) + LLM reasoning (borderline cases)
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy iteration
- ⚡ **Fast**: Processes ~9,000 names per second (fixed mode) or ~100 names/sec (with LLM assessment)
- 📊 **Transparent**: Complete audit trail with confidence scores and reasoning

## Features

### Core Capabilities

- **CSV Upload**: Drag-and-drop interface for single column CSV files
- **Hybrid Similarity Scoring**: Three-layer intelligent matching
  - **40% Fuzzy Matching** (RapidFuzz): Handles typos, misspellings (e.g., "Mcrosoft" → "Microsoft")
  - **15% Token Overlap**: Accounts for word order variations
  - **45% Semantic Embeddings**: Understands meaning and context (distinguishes "American Express" from "American Airlines")
- **Phonetic Matching**: Double Metaphone algorithm for pronunciation-based bonus/penalty (±2-4%)
- **Canonical Name Selection**: Automatically selects simplest name (shortest → fewest words → best capitalization)
- **Confidence Scores**: Every mapping includes explainable confidence score (0-100%)
- **Flexible Export**: Download mappings as CSV or complete audit log as JSON

### 🆕 AI-Enhanced Assessment (NEW)

- **🤖 LLM Borderline Assessment**: GPT-4o-mini evaluates ambiguous matches (43-73% similarity)
  - **Pre-clustering approach**: Adjusts similarity matrix before clustering
  - **Async processing**: Non-blocking with real-time progress updates
  - **In-memory caching**: Prevents duplicate API calls within session
  - **Blending strategy**: LLM adjusts but doesn't override fuzzy scores
  - **Cost-effective**: ~$2.50 per 500 names

- **Anti-Hallucination Guardrails**: 5-level system ensures honest AI responses
  1. **Explicit "unknown" option**: LLM can admit uncertainty
  2. **Confidence threshold**: Responses below 60% converted to "unknown"
  3. **Reasoning validation**: Rejects generic phrases and short responses
  4. **Hallucination detection**: Flags responses using external knowledge
  5. **Response format validation**: Ensures valid JSON structure

- **Visual Indicators**: LLM-reviewed matches marked with badges
  - 🤖✓ = LLM confirmed "same company"
  - 🤖✗ = LLM confirmed "different companies"
  - 🤖❓ = LLM uncertain (unknown)

### Advanced Features

- **Semantic Embeddings**: Multiple modes for different needs
  - **OpenAI 3-large**: Best quality (~90% accuracy, $0.13/1M tokens)
  - **OpenAI 3-small**: Balanced (~85% accuracy, $0.02/1M tokens) - **Default**
  - **Local Mode**: Privacy-first using sentence-transformers (~75% accuracy, free)
  - **Disabled**: Fuzzy matching only (~61% accuracy, fastest)

- **Adaptive Thresholding**: Optional GMM-based data-driven threshold calculation
  - Stratified reservoir sampling (3.6x faster, unbiased)
  - Three-zone decision system (auto-accept/promotion/reject)
  - Phonetic promotion for borderline cases

- **Multiple Clustering Modes**:
  - **Fixed Threshold** (85%): Fast, deterministic, works for most datasets
  - **Adaptive GMM**: Data-driven thresholds adapt to dataset characteristics
  - **HAC (Hierarchical Agglomerative Clustering)**: Advanced multi-level grouping
  - **HAC + LLM**: Best accuracy for complex/ambiguous datasets

- **Complete Audit Trail**: Every decision explained with reasoning, confidence, and method used

## Tech Stack

- **Frontend**: React 18 + Vite (fast HMR, modern tooling)
- **Backend**: Python 3.13 + FastAPI (async, high performance)
- **AI/ML Stack**:
  - **OpenAI GPT-4o-mini**: LLM borderline assessment with guardrails
  - **OpenAI text-embedding-3-small/large**: Semantic similarity embeddings
  - **sentence-transformers**: Local privacy-mode embeddings
  - **RapidFuzz 3.14**: Fast fuzzy string matching (WRatio, token_set_ratio)
  - **Metaphone**: Phonetic matching algorithm
  - **scikit-learn**: GMM adaptive thresholding
- **Data Processing**: Pandas 2.3 (CSV handling)
- **HTTP Client**: OpenAI Python SDK + aiohttp (async API calls)

## Intended Users

This tool is designed for:

- **Data Analysts**: Cleaning company name data from multiple sources (CRM, financial reports, databases)
- **Financial Analysts**: Standardizing entity names for portfolio tracking, risk assessment, compliance
- **Research Teams**: Deduplicating organization names in academic or market research datasets
- **Operations Teams**: Normalizing vendor/supplier names across procurement systems
- **Data Scientists**: Preprocessing company data for ML models or graph analysis
- **Compliance Officers**: Entity resolution for sanctions screening, KYC, due diligence

### Use Cases

1. **Merging Datasets**: Combining customer/vendor lists from different systems with inconsistent naming
2. **Data Quality**: Identifying and resolving duplicate company records in databases
3. **Report Standardization**: Ensuring consistent company names across financial/operational reports
4. **Market Research**: Aggregating company mentions from multiple sources (news, filings, databases)
5. **Compliance Screening**: Resolving entity names for sanctions/watchlist matching
6. **Graph Analysis**: Creating clean entity nodes for knowledge graphs or relationship mapping

### When to Use Each Mode

- **Fixed Threshold**: Standard datasets with clear name similarities (default, fastest)
- **Adaptive GMM**: Large diverse datasets with varying similarity patterns (data-driven)
- **HAC**: Multi-level hierarchical grouping for complex organizational structures
- **HAC + LLM**: Maximum accuracy for datasets with many borderline/ambiguous cases

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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
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

**Run backend tests** (76/76 passing ✅):
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
pytest tests/test_llm_borderline_service.py -v  # 🆕 LLM assessment (26 tests)
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

**LLM Feature Tests** (26 tests covering):
- Initialization and configuration
- Anti-hallucination guardrails (5 levels)
- Response format validation
- Confidence threshold enforcement
- Reasoning quality validation
- Hallucination detection
- Caching functionality
- Async batch processing
- Realistic scenario validation

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

4. **Choose Embedding Mode** (affects similarity scoring):
   - **Best Quality**: OpenAI 3-large (~90% accuracy, $0.13/1M tokens)
   - **Balanced** (Recommended): OpenAI 3-small (~85% accuracy, $0.02/1M tokens)
   - **Privacy Mode**: Local embeddings (~75% accuracy, no API calls)
   - **Disabled**: Fuzzy matching only (~61% accuracy, fastest)

5. **Choose Clustering Mode**:
   - **Fixed Threshold** (Default): Fast, deterministic (85% threshold)
   - **Adaptive GMM**: Data-driven thresholds for diverse datasets
   - **HAC**: Hierarchical clustering (42% distance threshold)
   - **HAC + LLM**: HAC with AI assessment for borderline cases (best accuracy)

6. **🆕 Enable LLM Assessment** (Optional - HAC mode only):
   - Check "🤖 Enable AI Borderline Assessment" for ambiguous matches
   - GPT-4o-mini evaluates pairs in 43-73% similarity range
   - Adds ~30-60 seconds processing time for 500 names
   - Estimated cost: ~$2.50 per 500 names
   - Watch progress bar for real-time status updates

7. **Review Results**:
   - Examine standardized mappings with confidence scores
   - Look for 🤖 badges indicating LLM-reviewed matches
   - Click badges for tooltips showing LLM decisions
   - Review guardrail statistics in Summary section

8. **Export**: Download results as CSV (mappings) or JSON (audit log)

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

### 4. Process Flow

#### Standard Workflow (Fixed/Adaptive/HAC modes)

```
1. CSV Upload
   ↓
2. Name Extraction & Validation
   ↓
3. Normalization (lowercase, remove punctuation, strip suffixes)
   ↓
4. Similarity Calculation
   ├─ 40% Fuzzy Matching (RapidFuzz WRatio)
   ├─ 15% Token Overlap (token_set_ratio)
   ├─ 45% Semantic Embeddings (OpenAI/local/disabled)
   └─ ±2-4% Phonetic Bonus/Penalty (Metaphone)
   ↓
5. Clustering
   ├─ Fixed: Apply 85% threshold
   ├─ Adaptive GMM: Calculate T_LOW/S_90/T_HIGH, apply three-zone logic
   └─ HAC: Hierarchical clustering with 42% distance threshold
   ↓
6. Canonical Name Selection (shortest → fewest words → best caps)
   ↓
7. Confidence Score Assignment
   ↓
8. Results & Audit Log Generation
```

#### 🆕 HAC + LLM Workflow (Enhanced Accuracy)

```
1. CSV Upload
   ↓
2. Name Extraction & Validation
   ↓
3. Normalization
   ↓
4. Similarity Matrix Building (fuzzy + embeddings + phonetics)
   ↓
5. 🤖 LLM Borderline Assessment (NEW)
   ├─ Identify borderline pairs (0.27-0.57 distance / 43-73% similarity)
   ├─ Batch assessment (10 pairs at a time, async)
   ├─ Apply 5-level guardrails:
   │  1. Validate response structure
   │  2. Check confidence threshold (≥60%)
   │  3. Validate reasoning quality (no generic phrases)
   │  4. Detect hallucination (flag external knowledge)
   │  5. Flag guardrail violations
   ├─ LLM Decision: "same", "different", or "unknown"
   ├─ Adjust similarity scores:
   │  • same: +15% × confidence
   │  • different: -15% × confidence
   │  • unknown: no change (preserve fuzzy score)
   └─ Cache results (prevent duplicate API calls)
   ↓
6. HAC Clustering (on adjusted similarity matrix)
   ↓
7. Canonical Name Selection
   ↓
8. Confidence Score Assignment + LLM Metadata
   ├─ Mark llm_reviewed: true for assessed pairs
   ├─ Store llm_decision: same/different/unknown
   └─ Generate guardrail statistics
   ↓
9. Results with LLM Badges & Audit Log
```

#### Key Decision Points

- **Embedding Mode**: Affects similarity calculation accuracy (steps 4-5)
- **Clustering Mode**: Determines grouping algorithm (step 6)
- **LLM Assessment**: Only available in HAC mode for borderline cases
- **Guardrails**: Ensure LLM honesty by converting weak responses to "unknown"
- **Caching**: Speeds up processing by avoiding duplicate LLM calls

### 5. Advanced: Adaptive GMM Mode with Stratified Sampling

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
│   │   │   ├── llm_borderline_service.py  # 🆕 LLM assessment with guardrails (NEW)
│   │   │   ├── gmm_threshold_service.py  # Adaptive thresholding
│   │   │   ├── blocking_service.py  # Stratified sampling
│   │   │   └── hac_service.py  # Hierarchical clustering
│   │   └── utils/            # Utility layer
│   │       ├── logger.py     # Logging setup
│   │       └── csv_handler.py  # CSV utilities
│   ├── tests/                # Test suite (76/76 passing ✅)
│   │   ├── test_name_matcher.py  # Name matching tests (12)
│   │   ├── test_gmm_threshold_service.py  # GMM tests (9)
│   │   ├── test_blocking_service.py  # Stratified sampling tests (29)
│   │   └── test_llm_borderline_service.py  # 🆕 LLM assessment tests (26)
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
# OpenAI API Configuration
OPENAI_API_KEY: str = ""                # Your OpenAI API key (required for embeddings + LLM)

# 🆕 Embedding Configuration
DEFAULT_EMBEDDING_MODE: str = "openai-small"  # openai-large, openai-small, local, disabled
EMBEDDING_DIMENSIONS: int = 512         # Reduced from 1536 for speed
WRATIO_WEIGHT: float = 0.40             # Fuzzy matching (typos)
TOKEN_SET_WEIGHT: float = 0.15          # Token overlap (reduced from 0.40)
EMBEDDING_WEIGHT: float = 0.45          # Semantic similarity

# 🆕 LLM Borderline Assessment (NEW)
LLM_BORDERLINE_ENABLED: bool = False    # Enable GPT-4o-mini assessment
LLM_BORDERLINE_MODEL: str = "gpt-4o-mini"  # LLM model to use
LLM_BORDERLINE_DISTANCE_LOW: float = 0.27   # Lower bound (43% similarity)
LLM_BORDERLINE_DISTANCE_HIGH: float = 0.57  # Upper bound (73% similarity)
LLM_BORDERLINE_ADJUSTMENT_STRENGTH: float = 0.15  # ±15% adjustment

# 🆕 Anti-Hallucination Guardrails (NEW)
LLM_MIN_CONFIDENCE: float = 0.60        # Convert responses <60% to "unknown"
LLM_ALLOW_UNKNOWN: bool = True          # Allow LLM to admit uncertainty
LLM_MIN_REASONING_LENGTH: int = 30      # Reject reasoning <30 chars
LLM_MAX_API_RETRIES: int = 3            # Retry failed API calls
LLM_API_TIMEOUT: int = 30               # Timeout per API call (seconds)

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
- **Use Case**: Standard datasets, fast processing required

**Adaptive GMM Mode** (560 names):
- **Speed**: 0.585s total (3.6x faster than old sequential sampling)
- **Quality**: 4.3x better match representation (26% within-block vs 6%)
- **Threshold Separation**: 50% better (0.021 vs 0.015)
- **Coverage**: Unbiased sampling across all names
- **Use Case**: Large diverse datasets, data-driven thresholds

**🆕 HAC + LLM Mode** (500 names with embeddings):
- **Speed**: ~30-60 seconds total
  - Base processing: 1-2s
  - LLM assessment: 25-55s (depends on borderline pair count)
- **Borderline Pairs**: Typically 50-150 pairs (10-30% of total)
- **LLM Throughput**: ~10 pairs per API call (batched)
- **Cache Hit Rate**: 15-30% (reduces duplicate calls)
- **Cost**: ~$2.50 per 500 names (GPT-4o-mini pricing)
- **Accuracy**: Highest accuracy mode (best for ambiguous cases)
- **Guardrail Trigger Rate**: 5-15% (low confidence / weak reasoning flagged)
- **Use Case**: Complex datasets with many borderline similarities

**Stratified Sampling Benefits**:
- Eliminates bias where first ~320 names got all comparisons
- Proportional representation from all name blocks
- 5% cross-block sampling captures rare matches
- Fixed RNG seed ensures reproducibility

**LLM Assessment Benefits**:
- Resolves ambiguous matches fuzzy matching can't handle
- Understands business context (subsidiaries, acquisitions, DBA names)
- Anti-hallucination guardrails ensure honest responses
- Caching prevents duplicate API calls within session
- Progress tracking keeps users informed during processing

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive developer guide for working with the codebase
- **🆕 [EMBEDDING_SETUP_GUIDE.md](EMBEDDING_SETUP_GUIDE.md)** - Setup and configuration guide for LLM embeddings
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed architecture diagrams and layer descriptions
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Complete test results and coverage report

## Development

### Running Tests

```bash
cd backend

# All unit tests (76 tests, ~20 seconds)
pytest -v

# Specific test suites
pytest tests/test_name_matcher.py -v       # 12 tests - Core matching
pytest tests/test_gmm_threshold_service.py -v  # 9 tests - Adaptive thresholds
pytest tests/test_blocking_service.py -v   # 29 tests - Stratified sampling
pytest tests/test_llm_borderline_service.py -v  # 26 tests - LLM assessment (NEW)

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
