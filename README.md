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
- **Automatic Grouping**: Identify names referring to the same company using fuzzy matching
- **Canonical Name Selection**: Assigns the simplest or most recognizable name as canonical
- **Confidence Scores**: Every mapping includes a confidence score (0-1)
- **Export Results**: Download mappings as CSV or audit log as JSON
- **Audit Log**: Complete audit trail of all processing decisions with reasoning

## Tech Stack

- **Frontend**: React 18 + Vite (fast HMR, modern tooling)
- **Backend**: Python 3.13 + FastAPI (async, high performance)
- **Matching Algorithm**: RapidFuzz 3.14 (fast fuzzy string matching)
- **Data Processing**: Pandas 2.3 (CSV handling)

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

# 4. Create logs directory
mkdir logs

# 5. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend runs on**: http://localhost:8000
**API Docs**: http://localhost:8000/docs

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

**Run backend tests** (5/5 passing ✅):
```bash
cd backend
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pytest -v                      # All tests
pytest tests/test_name_matcher.py -v  # Specific tests
```

**Run workflow test** (integration test ✅):
```bash
cd backend
./venv/Scripts/python test_workflow.py  # Windows
./venv/bin/python test_workflow.py      # macOS/Linux
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

4. **Review**: Examine the standardized mappings with confidence scores

5. **Export**: Download results as CSV (mappings) or JSON (audit log)

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
│   │   │   └── name_matcher.py  # Core matching algorithm
│   │   └── utils/            # Utility layer
│   │       ├── logger.py     # Logging setup
│   │       └── csv_handler.py  # CSV utilities
│   ├── tests/                # Test suite (5/5 passing)
│   ├── logs/                 # Application logs
│   ├── test_workflow.py      # Integration test
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

Edit `backend/app/config/settings.py`:

```python
# Matching algorithm
SIMILARITY_THRESHOLD: float = 85.0    # Adjust grouping strictness

# Corporate suffixes to normalize
CORPORATE_SUFFIXES: List[str] = [
    'inc', 'corp', 'llc', 'ltd', ...
]

# Server settings
HOST: str = "0.0.0.0"
PORT: int = 8000
```

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
- **Speed**: ~9,000 names per second
- **Accuracy**: 100% confidence on obvious matches
- **Efficiency**: 66.7% reduction in sample data

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive developer guide for working with the codebase
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed architecture diagrams and layer descriptions
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Complete test results and coverage report

## Development

### Running Tests

```bash
# Backend unit tests (fast)
cd backend
pytest -v

# Backend integration test (comprehensive)
./venv/Scripts/python test_workflow.py

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
