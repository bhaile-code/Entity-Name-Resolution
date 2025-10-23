# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Company Name Standardizer is a full-stack web application that automates the deduplication and standardization of company names from CSV files. It uses fuzzy string matching to group similar company names and selects canonical names based on simplicity.

**Key Principles**:
- All processing happens locally - no data leaves the user's machine
- Modular architecture for easy iteration and testing
- Separation of concerns (config, services, API, UI)
- Prototype-level code hygiene without over-engineering

## Architecture

### Backend (Python + FastAPI)

**Location**: `backend/`

**Modular Structure**:

```
backend/
├── app/
│   ├── main.py              # FastAPI app setup and lifecycle
│   ├── config/
│   │   ├── settings.py      # Centralized configuration
│   │   └── __init__.py
│   ├── api/
│   │   ├── routes.py        # API endpoint handlers
│   │   └── __init__.py
│   ├── services/
│   │   ├── name_matcher.py  # Core matching business logic
│   │   └── __init__.py
│   ├── utils/
│   │   ├── logger.py        # Logging setup
│   │   ├── csv_handler.py   # CSV parsing and validation
│   │   └── __init__.py
│   ├── models.py            # Pydantic data models
│   └── __init__.py
├── tests/
│   ├── test_name_matcher.py
│   └── __init__.py
├── logs/                    # Application logs
└── requirements.txt
```

**Layer Responsibilities**:

1. **`config/settings.py`** - Single source of truth for configuration
   - All environment variables and defaults
   - Server, API, CORS, matching algorithm settings
   - Corporate suffixes list
   - Logging configuration
   - Access via `from app.config import settings`

2. **`api/routes.py`** - HTTP endpoint handlers
   - Route definitions and request/response handling
   - Input validation using Pydantic
   - Error handling with proper HTTP status codes
   - Separates HTTP concerns from business logic

3. **`services/name_matcher.py`** - Business logic layer
   - `NameMatcher` class with fuzzy matching algorithm
   - **Algorithm flow**:
     1. Normalize names (lowercase, remove punctuation/suffixes)
     2. Use RapidFuzz (ratio, token_sort_ratio, token_set_ratio)
     3. Group names with similarity >= threshold (default 85%)
     4. Select canonical name (shortest/fewest words/capitalization)
   - No direct HTTP dependencies - pure business logic
   - Can be used standalone or from API

4. **`utils/csv_handler.py`** - CSV file utilities
   - Parsing CSV bytes to DataFrame
   - Extracting company names (first column)
   - File validation (encoding, structure)
   - Separates I/O from business logic

5. **`utils/logger.py`** - Logging setup
   - Consistent logger configuration
   - File and console handlers
   - Configured via settings

6. **`models.py`** - Data contracts
   - Pydantic models for API validation
   - Type safety and documentation
   - `ProcessingResult`, `CompanyMapping`, `AuditLogEntry`

**Key Algorithm Details**:
- Common corporate suffixes stripped during normalization (configurable in settings)
- Canonical name selection: shortest length → fewest words → best capitalization
- Confidence scores: weighted average (30% ratio, 30% token_sort, 40% token_set)
- Each name gets an audit log entry with reasoning

### Frontend (React + Vite)

**Location**: `frontend/`

**Modular Structure**:

```
frontend/
├── src/
│   ├── main.jsx
│   ├── App.jsx              # Main component, orchestrates state
│   ├── components/          # UI components
│   │   ├── FileUpload.jsx
│   │   ├── ResultsTable.jsx
│   │   ├── Summary.jsx
│   │   └── AuditLog.jsx
│   ├── services/            # External integrations
│   │   ├── api.js           # Backend API client
│   │   └── export.js        # File export utilities
│   ├── hooks/               # Custom React hooks
│   │   ├── useFileUpload.js
│   │   ├── useTableSort.js
│   │   └── index.js
│   ├── utils/               # Pure utility functions
│   │   ├── validators.js    # File/data validation
│   │   ├── formatters.js    # Display formatting
│   │   └── index.js
│   ├── constants/           # Application constants
│   │   ├── confidence.js    # Confidence thresholds
│   │   └── index.js
│   ├── config/              # Configuration
│   │   └── api.config.js    # API endpoints and settings
│   ├── App.css
│   └── index.css
├── public/
├── index.html
├── package.json
└── vite.config.js
```

**Layer Responsibilities**:

1. **`config/api.config.js`** - API configuration
   - Base URLs, endpoints, timeouts
   - Single place to change API settings

2. **`constants/`** - Application constants
   - Confidence thresholds and colors
   - File constraints
   - Sort directions, tab names
   - Shared enums and lookup values

3. **`utils/`** - Pure utility functions
   - `validators.js`: File and data validation logic
   - `formatters.js`: Display formatting (percentages, dates, numbers)
   - No side effects, easy to test

4. **`hooks/`** - Custom React hooks
   - `useFileUpload`: File upload state and logic
   - `useTableSort`: Table sorting state and memoization
   - Reusable stateful logic extraction

5. **`services/`** - External service integration
   - `api.js`: Axios-based API client with error handling
   - `export.js`: CSV/JSON download utilities
   - Side effects contained here

6. **`components/`** - UI components
   - Presentational and container logic
   - Use hooks for complex state
   - Props for configuration

**State Management**:
- Single source of truth in `App.jsx`
- Props drilling for simplicity (no Redux/Context needed yet)
- Custom hooks for reusable stateful logic
- Component-level state for UI interactions

## Development Commands

### Backend

```bash
cd backend

# Setup (first time)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Run development server
python -m app.main
# Or with auto-reload:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest                                    # All tests
pytest tests/test_name_matcher.py -v     # Specific file
pytest -k test_normalize_name            # Specific test

# Linting and formatting
black app/ tests/                         # Format code
flake8 app/ tests/                        # Check style
```

### Frontend

```bash
cd frontend

# Setup (first time)
npm install

# Run development server
npm run dev          # Starts on http://localhost:3000

# Build for production
npm run build        # Output to dist/

# Preview production build
npm run preview

# Linting
npm run lint
```

### Running Full Stack

**Terminal 1** (Backend):
```bash
cd backend
venv\Scripts\activate  # or source venv/bin/activate
uvicorn app.main:app --reload
```

**Terminal 2** (Frontend):
```bash
cd frontend
npm run dev
```

Open http://localhost:3000

## Code Patterns & Conventions

### Backend Patterns

**Importing from Modules**:
```python
from app.config import settings          # Configuration
from app.services import NameMatcher     # Business logic
from app.utils import CSVHandler, setup_logger  # Utilities
from app.api import router               # API routes
```

**Adding New Configuration**:
1. Add to `app/config/settings.py` in the `Settings` class
2. Access via `settings.VARIABLE_NAME`
3. Can use environment variables with defaults

**Adding New API Endpoints**:
1. Add route handler to `app/api/routes.py`
2. Use existing router: `@router.get()` or `@router.post()`
3. Import dependencies from services/utils layers
4. Return Pydantic models for type safety

**Error Handling**:
```python
try:
    # Business logic
    result = service.do_something()
    return result
except ValueError as e:
    # Known validation errors → 400
    logger.warning(f"Validation error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    # Unexpected errors → 500
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
```

**Testing**:
- Test services independently from API layer
- Use descriptive test names: `test_<function>_<scenario>`
- Test edge cases: empty inputs, malformed data
- Mock external dependencies if needed

### Frontend Patterns

**Importing**:
```javascript
import { API_CONFIG } from '../config/api.config'
import { FILE_CONSTRAINTS, getConfidenceLevel } from '../constants'
import { validateFile, formatConfidence } from '../utils'
import { useFileUpload, useTableSort } from '../hooks'
import { uploadFile } from '../services/api'
```

**Adding New Constants**:
1. Add to appropriate file in `src/constants/`
2. Export from `src/constants/index.js`
3. Use throughout app for consistency

**Creating Custom Hooks**:
```javascript
// src/hooks/useMyHook.js
export const useMyHook = (initialValue) => {
  const [state, setState] = useState(initialValue)

  const doSomething = () => {
    // Logic here
  }

  return { state, doSomething }
}
```

**Adding New Components**:
1. Create file in `src/components/ComponentName.jsx`
2. Import required hooks, utils, constants
3. Add styles to `App.css`
4. Use in parent component

**API Calls**:
- All API calls through `services/api.js`
- Handle both server errors and network errors
- Display user-friendly messages

## Important Implementation Details

### Configuration Changes

**Backend**: Edit `backend/app/config/settings.py`
- Change port, host, CORS origins
- Adjust similarity threshold
- Modify corporate suffixes list
- Configure logging level

**Frontend**: Edit `frontend/src/config/api.config.js`
- Change API base URL
- Adjust timeout values
- Modify retry settings

### CSV Processing
- Only first column is used (any header name)
- Empty rows and duplicates filtered automatically
- Validation in both frontend (extension) and backend (content)

### Matching Algorithm Tuning

**Threshold** (in `settings.py`):
- Higher (e.g., 90) → more groups, stricter matching
- Lower (e.g., 75) → fewer groups, more aggressive grouping
- Default: 85% works well for most cases

**Canonical Selection** (in `services/name_matcher.py`):
```python
def score_name(name: str) -> Tuple[int, int, int]:
    return (
        len(name),           # Prefer shorter
        len(name.split()),   # Prefer fewer words
        -sum(1 for c in name if c.isupper())  # Prefer capitalization
    )
```
Adjust tuple weights to change selection criteria.

**Confidence Weights** (in `services/name_matcher.py`):
```python
score = (ratio * 0.3 + token_sort * 0.3 + token_set * 0.4)
```
Adjust weights (must sum to 1.0) to prioritize different fuzzy algorithms.

## API Contract

### POST /api/process

**Request**:
```
Content-Type: multipart/form-data
Field: file (CSV file)
```

**Success Response (200)**:
```json
{
  "mappings": [
    {
      "original_name": "Apple Inc.",
      "canonical_name": "Apple",
      "confidence_score": 0.95,
      "group_id": 0,
      "alternatives": ["Apple Inc.", "Apple Corporation"]
    }
  ],
  "audit_log": {
    "filename": "companies.csv",
    "processed_at": "2024-01-01T12:00:00",
    "total_names": 100,
    "total_groups": 85,
    "entries": [...]
  },
  "summary": {
    "total_input_names": 100,
    "total_groups_created": 85,
    "reduction_percentage": 15.0,
    "average_group_size": 1.18,
    "processing_time_seconds": 0.45
  }
}
```

**Error Responses**:
- 400: Invalid CSV (empty, malformed, wrong type)
- 500: Server processing error

### GET /api/health

Returns application health status and version.

## Common Development Scenarios

### Adding New Matching Strategy

1. Update `calculate_confidence()` in `services/name_matcher.py`
2. Add new fuzzy algorithm call
3. Update weighted average calculation
4. Add test in `tests/test_name_matcher.py`
5. Document in this file

### Making Threshold Configurable via API

1. Add query parameter to `POST /api/process` in `api/routes.py`:
   ```python
   async def process_companies(
       file: UploadFile,
       threshold: float = Query(default=None)
   ):
       matcher = NameMatcher(similarity_threshold=threshold)
   ```
2. Update frontend to allow threshold input
3. Pass parameter in API call

### Adding Export Formats

1. **Backend**: Create `app/services/export_service.py`
   ```python
   class ExportService:
       @staticmethod
       def to_excel(mappings): ...
   ```

2. **Frontend**: Add to `services/export.js`
   ```javascript
   export const downloadExcel = (data, filename) => { ... }
   ```

3. Add button in `ResultsTable.jsx`

### Adding Custom Normalization Rules

1. Edit `normalize_name()` in `services/name_matcher.py`
2. Add industry-specific rules (e.g., remove "&" → "and")
3. Update `CORPORATE_SUFFIXES` in `config/settings.py`
4. Test with representative data

## Testing Strategy

### Current Test Status ✅

**Backend**: Fully tested and operational (5/5 tests passing)
**Frontend**: Not yet tested (requires `npm install`)
**Integration**: Pending full stack setup

See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for complete test report.

### Backend Tests (Verified ✅)

**Unit Tests** (`tests/test_name_matcher.py`):
```bash
cd backend
venv\Scripts\activate  # Windows
pytest -v

# Results: 5/5 tests passing in 4.35s
# - test_normalize_name
# - test_select_canonical_name
# - test_calculate_confidence
# - test_group_similar_names
# - test_process_names
```

**Integration Test** (`test_workflow.py`):
```bash
cd backend
./venv/Scripts/python test_workflow.py

# Tests complete end-to-end workflow
# Verifies: config, processing, CSV handling, normalization
```

**What's Tested**:
- ✅ Name normalization (suffix removal, lowercase)
- ✅ Fuzzy matching with RapidFuzz
- ✅ Grouping algorithm (clustering)
- ✅ Canonical name selection
- ✅ Confidence score calculation
- ✅ Audit log generation
- ✅ CSV file validation
- ✅ Configuration loading
- ✅ All modular imports

**Test Coverage**:
- Services layer: Comprehensive
- Utils layer: Comprehensive
- Config layer: Verified
- API layer: Not yet tested (requires running server)
- Models: Validated via services

### Writing New Tests

**Pattern for Service Tests**:
```python
def test_my_feature():
    """Test description."""
    # Arrange
    matcher = NameMatcher()
    input_data = ["Test", "Test Inc"]

    # Act
    result = matcher.some_method(input_data)

    # Assert
    assert result is not None
    assert len(result) == expected_value
```

**Running Specific Tests**:
```bash
pytest tests/test_name_matcher.py::test_normalize_name  # Single test
pytest -k "normalize"  # All tests matching "normalize"
pytest --maxfail=1     # Stop after first failure
pytest -x              # Stop on first failure (short form)
```

### Frontend Tests (Future)

**Planned**:
- Component tests with React Testing Library
- Hook tests with @testing-library/react-hooks
- Integration tests with MSW for API mocking

**Setup** (when ready):
```bash
cd frontend
npm install
npm test
```

### Integration Testing

**Manual Test Workflow**:
1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Open http://localhost:3000
4. Upload [sample_companies.csv](../sample_companies.csv)
5. Verify results display correctly
6. Test export functionality

**Expected Result**:
- 34 input names → ~13 groups
- High confidence scores for obvious matches
- Fast processing (<1 second)

### Performance Testing

**Benchmark Test**:
```python
# Test with large dataset
import time
from app.services import NameMatcher

matcher = NameMatcher()
names = ["Company " + str(i) for i in range(10000)]

start = time.time()
result = matcher.process_names(names)
elapsed = time.time() - start

print(f"Processed {len(names)} names in {elapsed:.2f}s")
print(f"Throughput: {len(names)/elapsed:.0f} names/second")
```

**Current Performance** (tested):
- ~9,000 names per second
- Linear scaling with input size
- Minimal memory usage

### Test Data

**Sample Files**:
- `sample_companies.csv` - 34 company name variants
- Good for testing grouping logic
- Contains Apple, Microsoft, Google, Amazon variants

**Creating Test Data**:
```python
# Generate test CSV
import pandas as pd

data = {
    'company_name': [
        'Apple Inc.',
        'Apple',
        'Microsoft Corp',
        'Microsoft'
    ]
}
pd.DataFrame(data).to_csv('test_companies.csv', index=False)
```

## Known Limitations

- Single-column CSV only (by design for simplicity)
- String-based matching (no external company databases)
- Large files (>10k names) may take several seconds
- No result persistence (page refresh loses data)
- No manual override of groupings

## Future Enhancement Ideas

- Adjustable threshold slider in UI
- Manual grouping adjustments (drag-and-drop)
- Save/load sessions
- Industry-specific matching rules
- International character normalization
- Batch processing of multiple files
- Excel export with formatting

## Troubleshooting

**Import errors after refactoring**:
- Ensure in correct directory: `cd backend` or `cd frontend`
- Check imports use new paths: `from app.services import NameMatcher`
- Restart dev server after structural changes

**Backend won't start**:
- Check port 8000 availability
- Verify virtual environment activated
- Ensure all requirements installed: `pip install -r requirements.txt`
- Check logs directory exists: `mkdir logs`

**Frontend can't connect**:
- Verify backend running on http://localhost:8000
- Check CORS configuration in `config/settings.py`
- Verify `VITE_API_URL` in `.env` if using one

**Tests failing**:
- Ensure correct directory: `cd backend`
- Install test dependencies: `pip install pytest`
- Check import paths updated for new structure
- Run with verbose: `pytest -v`

**Poor matching results**:
- Lower `SIMILARITY_THRESHOLD` in settings for more grouping
- Check for unusual characters/encoding in input
- Review normalization logic - may need domain-specific rules
- Examine audit log to understand matching decisions

## Project Philosophy

This is a **prototype** with **good engineering practices**:

✅ **Do**:
- Separate concerns (config, services, API, UI)
- Write testable, modular code
- Use meaningful names and documentation
- Handle errors gracefully
- Log important operations
- Validate inputs

❌ **Don't**:
- Over-engineer with complex patterns
- Add unnecessary abstractions
- Implement features not needed now
- Optimize prematurely
- Add heavy dependencies for simple tasks

**Goal**: Balance between "quick prototype" and "maintainable codebase" for easy iteration.
