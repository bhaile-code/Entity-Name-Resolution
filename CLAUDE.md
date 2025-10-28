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
│   │   ├── name_matcher.py           # Core matching business logic
│   │   ├── embedding_service.py      # Semantic embeddings (OpenAI + local)
│   │   ├── gmm_threshold_service.py  # GMM-based adaptive thresholding
│   │   ├── blocking_service.py       # Stratified reservoir sampling
│   │   └── __init__.py
│   ├── utils/
│   │   ├── logger.py        # Logging setup
│   │   ├── csv_handler.py   # CSV parsing and validation
│   │   └── __init__.py
│   ├── models.py            # Pydantic data models
│   └── __init__.py
├── tests/
│   ├── test_name_matcher.py
│   ├── test_gmm_threshold_service.py
│   └── __init__.py
├── logs/                    # Application logs
├── test_adaptive_workflow.py
├── test_sample_data_500.py
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
   - **Supports two modes**:
     - **Fixed threshold mode** (default): Uses preset 85% similarity threshold
     - **Adaptive GMM mode** (opt-in): Data-driven thresholds using Gaussian Mixture Model
   - **Algorithm flow**:
     1. Normalize names (lowercase, remove punctuation/suffixes)
     2. Use RapidFuzz (WRatio, token_set_ratio)
     3. Apply phonetic matching bonus/penalty (Metaphone)
     4. Group names with similarity >= threshold
     5. Select canonical name (shortest/fewest words/capitalization)
   - **Adaptive mode workflow**:
     1. Collect pairwise similarity scores (max 50,000 pairs for performance)
     2. Fit 2-component GMM on score distribution
     3. Calculate T_LOW, S_90, T_HIGH from posterior probabilities
     4. Apply three-zone decision logic (auto-accept/promotion/reject)
     5. Adjust confidence scores based on zone and phonetic agreement
   - No direct HTTP dependencies - pure business logic
   - Can be used standalone or from API

4. **`services/embedding_service.py`** - Semantic embedding service
   - `EmbeddingService` class with three modes:
     - **OpenAI 3-large**: Best quality (~90% accuracy, $0.13/1M tokens)
     - **OpenAI 3-small**: Balanced (~85% accuracy, $0.02/1M tokens) - default
     - **Local**: Privacy mode (~75% accuracy, no API calls, uses sentence-transformers)
   - Intelligent caching to avoid re-computing embeddings
   - Graceful fallback when API unavailable
   - Batch processing for efficiency
   - Cosine similarity calculation
   - Factory function: `create_embedding_service(mode)`

5. **`services/gmm_threshold_service.py`** - GMM adaptive thresholding
   - `GMMThresholdCalculator` class
   - 2-component Gaussian Mixture Model fitting
   - Posterior probability calculation: P(same|score)
   - Threshold finding via binary search
   - Margin penalty calculation for promoted pairs
   - GMM metadata extraction (means, variances, weights)

6. **`utils/csv_handler.py`** - CSV file utilities
   - Parsing CSV bytes to DataFrame
   - Extracting company names (first column)
   - File validation (encoding, structure)
   - Separates I/O from business logic

6. **`utils/logger.py`** - Logging setup
   - Consistent logger configuration
   - File and console handlers
   - Configured via settings

7. **`models.py`** - Data contracts
   - Pydantic models for API validation
   - Type safety and documentation
   - `ProcessingResult`, `CompanyMapping`, `AuditLogEntry`
   - `ThresholdInfo` (method, thresholds, fallback reason)
   - `GMMMetadata` (cluster stats, pairs analyzed)

**Key Algorithm Details**:
- Common corporate suffixes stripped during normalization (configurable in settings)
- Canonical name selection: shortest length → fewest words → best capitalization
- **Hybrid confidence scoring** (when embeddings enabled):
  - **40% WRatio**: Fuzzy string matching (handles typos like "Mcrosoft" → "Microsoft")
  - **15% token_set**: Token overlap (reduced from 40% to fix shared-word problem)
  - **45% Semantic embeddings**: Understanding context (distinguishes "American Express" from "American Airlines")
  - **±2-4% Phonetic bonus/penalty**: Double Metaphone agreement/disagreement
- **Fallback confidence scoring** (embeddings disabled or failed):
  - 60% WRatio, 40% token_set (original algorithm)
- Phonetic matching: +4% if phonetics agree, -2% if disagree (using Double Metaphone)
- Phonetics skipped for: numbers, single chars, short acronyms (e.g., "3M", "IBM")
- Accent folding via unidecode for non-ASCII characters (e.g., "São" → "Sao")
- Each name gets an audit log entry with reasoning
- **Embedding modes**:
  - `openai-large`: Best quality (~90% accuracy, $0.13/1M tokens)
  - `openai-small`: Balanced (~85% accuracy, $0.02/1M tokens) - **default**
  - `local`: Privacy mode (~75% accuracy, uses sentence-transformers locally)
  - `disabled`: Fuzzy matching only (~61% accuracy, original algorithm)

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
- **NEW**: Configure embedding settings (see Embedding Configuration below)

**Frontend**: Edit `frontend/src/config/api.config.js`
- Change API base URL
- Adjust timeout values
- Modify retry settings

### Embedding Configuration (NEW Feature)

**Setup**:
1. Copy `backend/.env.example` to `backend/.env`
2. Add your OpenAI API key (get from https://platform.openai.com/api-keys):
   ```
   OPENAI_API_KEY=sk-proj-your-key-here
   ```
3. Choose default embedding mode (optional, defaults to `openai-small`):
   ```
   DEFAULT_EMBEDDING_MODE=openai-small  # or 'openai-large', 'local', 'disabled'
   ```

**Available Modes**:
- `openai-large`: Best quality (~90% accuracy, $0.13/1M tokens, ~3s for 500 names)
- `openai-small`: Balanced (~85% accuracy, $0.02/1M tokens, ~1s for 500 names) - **recommended**
- `local`: Privacy mode (~75% accuracy, $0, ~3s for 500 names + 2s model load)
- `disabled`: Fuzzy matching only (~61% accuracy, original algorithm, fastest)

**Cost Estimates** (OpenAI API):
- 100 names: ~$0.000008 (essentially free)
- 500 names: ~$0.00004 (essentially free)
- 1000 names: ~$0.00008 (essentially free)
- Monthly (100 uploads of 500 names): $0.20 - $1.30

**Tuning Weights** (in `settings.py`):
```python
WRATIO_WEIGHT = 0.40        # Fuzzy matching (typos)
TOKEN_SET_WEIGHT = 0.15     # Token overlap (reduced from 0.40)
EMBEDDING_WEIGHT = 0.45     # Semantic similarity (NEW)
```
Weights should sum to ~1.0. Increase `EMBEDDING_WEIGHT` for more semantic understanding, increase `WRATIO_WEIGHT` for better typo handling.

**When to Use Each Mode**:
- **Use `openai-small`** (default): Best balance for most cases
- **Use `openai-large`**: When accuracy is critical and cost is not a concern
- **Use `local`**: When data privacy is paramount or offline processing required
- **Use `disabled`**: Testing, comparison with original algorithm, or when embeddings unavailable

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

**Confidence Weights** (in `settings.py` - NEW hybrid approach):
```python
# When embeddings enabled (default)
base_score = (
    wratio * 0.40 +           # Fuzzy matching (handles typos)
    token_set * 0.15 +        # Token overlap (reduced from 0.40)
    embedding_score * 0.45    # Semantic similarity (NEW)
)
phonetic_bonus = _calculate_phonetic_bonus(norm1, norm2)  # +4, -2, or 0
final_score = max(0, min(100, base_score + phonetic_bonus))

# When embeddings disabled (fallback)
base_score = (wratio * 0.6 + token_set * 0.4)
```

**Why the change**:
- **Problem**: token_set_ratio at 40% weight caused "American Express" to match "American" with 100% confidence (shared word "American")
- **Solution**: Reduced token_set to 15%, added embeddings at 45% to understand semantic context
- **Result**: "American Express" vs "American" now scores ~68% (correctly rejected), while "American Express" vs "AmEx" scores ~94% (correctly grouped)

Adjust weights in `settings.py` (must sum to ~1.0). WRatio handles typos, token_set handles word order, embeddings understand meaning. Phonetic matching provides additional signal based on pronunciation similarity.

### GMM Adaptive Thresholding (Optional Feature)

**Overview**: Instead of using a fixed similarity threshold (85%), the system can automatically determine optimal thresholds by analyzing the distribution of similarity scores using a Gaussian Mixture Model.

**How to Enable**:
- **Frontend**: Select "Adaptive GMM-Based" mode before uploading
- **Backend**: Pass `use_adaptive_threshold=true` query parameter to `/api/process`
- **CLI**: `NameMatcher(use_adaptive_threshold=True)`

**Algorithm**:

1. **Pairwise Score Collection** (using Stratified Reservoir Sampling)
   - Uses block-based stratified sampling for unbiased pair selection
   - Samples up to 50,000 pairs from potentially millions of combinations
   - See "Stratified Reservoir Sampling" section below for detailed algorithm
   - Progress logged every 5,000 pairs (10% intervals)
   - Includes both composite score and phonetic agreement flag

2. **GMM Fitting**
   - Fit 2-component Gaussian Mixture Model on similarity scores
   - Identifies two clusters:
     - **Low cluster**: Different companies (mean ~0.2-0.3)
     - **High cluster**: Same companies (mean ~0.95-0.99)
   - Requires minimum 50 pairs (falls back to fixed threshold if insufficient)

3. **Threshold Calculation** (from posterior probabilities)
   - **T_LOW**: Score where P(same|score) = 0.02 (reject below this)
   - **S_90**: Score where P(same|score) = 0.90 (promotion eligibility)
   - **T_HIGH**: Score where P(same|score) = 0.98 (auto-accept above this)

4. **Three-Zone Decision System**
   - **Auto-accept zone** (score ≥ T_HIGH):
     - Always grouped
     - Confidence = 100 × P(same|score) + (+4 if phonetics agree)
     - Typically >89% confidence

   - **Promotion zone** (S_90 ≤ score < T_HIGH):
     - Grouped ONLY if phonetics agree
     - Confidence = 100 × P(same|score) + 4 - penalty(score)
     - Penalty = 10 × (1 - (score - S_90) / (T_HIGH - S_90))
     - Capped at 89% to distinguish from auto-accepts

   - **Reject zone** (score ≤ T_LOW):
     - Not grouped (treated as different companies)
     - If no matches found, single-member group with 100% confidence

**Configuration** (in `settings.py`):
```python
GMM_MIN_SAMPLES = 50       # Minimum pairs required for GMM
GMM_MAX_PAIRS = 50000      # Cap pairwise collection (performance)
GMM_FALLBACK_T_HIGH = 92.0 # Fallback if GMM unavailable
GMM_FALLBACK_T_LOW = 80.0  # Fallback reject threshold
```

**Performance**:
- 100 names: ~1 second
- 500 names: ~6-7 seconds (50,000 pairs analyzed)
- 1000 names: ~12-15 seconds (50,000 pairs analyzed)
- 2000+ names: ~15-20 seconds (50,000 pairs analyzed)

**Benefits**:
- ✅ Data-driven (adapts to dataset characteristics)
- ✅ Transparent (thresholds + GMM metadata returned)
- ✅ Phonetic promotion (borderline cases with phonetic agreement)
- ✅ Confidence gradation (promoted pairs ≤89%, auto-accepts >89%)
- ✅ Backward compatible (opt-in, default is fixed threshold)
- ✅ Robust fallback (insufficient data → fixed threshold)

**Returned Metadata**:
```python
{
  "threshold_info": {
    "method": "adaptive_gmm",
    "t_low": 0.85,
    "s_90": 0.91,
    "t_high": 0.94,
    "fallback_reason": null
  },
  "gmm_metadata": {
    "cluster_means": [0.25, 0.97],
    "cluster_variances": [0.015, 0.0001],
    "cluster_weights": [0.88, 0.12],
    "total_pairs_analyzed": 50000
  }
}
```

**When to Use**:
- **Use Adaptive** when:
  - You have diverse datasets with varying name similarity patterns
  - You want data-driven, self-calibrating thresholds
  - You need phonetic promotion for borderline cases
  - Processing time of 6-20 seconds is acceptable

- **Use Fixed** when:
  - You need fast processing (<1 second)
  - Dataset is small (<100 names)
  - You have a known threshold that works for your data
  - Consistency across runs is critical

**Example Usage**:
```python
# Fixed threshold mode (default)
matcher = NameMatcher()
result = matcher.process_names(names)

# Adaptive GMM mode
matcher = NameMatcher(use_adaptive_threshold=True)
result = matcher.process_names(names)
print(f"Thresholds: T_LOW={result['summary']['threshold_info']['t_low']:.3f}")
```

**Testing**:
```bash
# Test with sample data
cd backend
python test_adaptive_workflow.py  # Compare fixed vs adaptive
python test_sample_data_500.py    # Test with 500 names

# Unit tests
pytest tests/test_gmm_threshold_service.py -v  # 9 tests for GMM
pytest tests/test_name_matcher.py -v           # 12 tests for matcher
```

### Stratified Reservoir Sampling (Pairwise Score Collection)

**Overview**: When using adaptive GMM mode, the system needs to collect pairwise similarity scores to fit the Gaussian Mixture Model. Instead of comparing all possible pairs (O(n^2) complexity), the system uses stratified reservoir sampling to collect a representative sample of 50,000 pairs efficiently and without bias.

**Why Stratified Sampling?**

The naive sequential approach (compare pairs (0,1), (0,2)... until 50,000 pairs) has severe bias:
- First ~320 names get all comparisons
- Last 180+ names are ignored completely
- High-similarity pairs within later names never discovered
- GMM thresholds skewed toward early-file characteristics

Stratified reservoir sampling solves this by:
- Grouping similar names into blocks (strata)
- Sampling proportionally from each block
- Including cross-block pairs for rare matches
- Using fixed RNG seed for reproducibility

**Algorithm Components**:

1. **Blocking Key Generation** (`BlockingKeyGenerator`)
   - **Purpose**: Group potentially similar names together
   - **Strategy**: Hybrid token + phonetic approach
   - **Process**:
     1. Normalize name (lowercase, accent folding, remove punctuation)
     2. Extract first significant token (skip stopwords: "the", "a", "an")
     3. Generate Double Metaphone phonetic code
     4. Combine: `"token_PHONETIC"` (e.g., "Apple Inc." -> "apple_APL")
   - **Edge Cases**:
     - Stopword-only names: Use full normalized name
     - Digits/short tokens: Skip phonetics, use token only
     - Phonetic failure: Graceful fallback to token only
   - **Tracking**: Counts phonetic skips for systematic gap detection

2. **Block Creation**
   - Group names by blocking key
   - Filter singletons (blocks with size < 2)
   - Cap blocks at 5,000 pairs maximum (prevent giant blocks)
   - Log statistics (blocks created, singletons filtered, avg size)

3. **Budget Allocation** (Within-Block)
   - **Split**: 95% within-block, 5% cross-block
   - **Within-Block Allocation**: 80% proportional + 20% floor
     - **Proportional (80%)**: Larger blocks get more samples
       - `allocation[block] = (block_pairs / total_pairs) * proportional_budget`
     - **Floor (20%)**: Even distribution ensures small blocks represented
       - `floor_per_block = floor_budget / num_blocks`
   - **Caps**: Never exceed available pairs in block

4. **Within-Block Sampling** (Algorithm R)
   - **Purpose**: Uniform random sample from each block's pairs
   - **Algorithm**: Classical reservoir sampling (on-the-fly)
   - **Process**:
     1. Iterate through all possible pairs in block
     2. First `sample_size` pairs fill the reservoir
     3. Each subsequent pair replaces random reservoir item with probability `sample_size / pairs_seen`
   - **Guarantee**: Every pair has equal probability of being selected
   - **Memory**: Never stores more than `sample_size` pairs

5. **Cross-Block Sampling** (Two-Stage)
   - **Purpose**: Capture rare matches across different blocks
   - **Stage 1**: Sample block pairs uniformly
   - **Stage 2**: For each block pair, sample name pairs
   - **Final**: Reservoir sample from all cross-block candidates
   - **Constraint**: Only pairs from different blocks

**Configuration** (in `settings.py`):
```python
# Blocking Configuration
BLOCKING_MIN_BLOCK_SIZE = 2       # Skip singletons
BLOCKING_MAX_BLOCK_PAIRS = 5000   # Cap giant blocks

# Sampling Budget Split
SAMPLING_WITHIN_BLOCK_PCT = 0.95  # 95% within blocks
SAMPLING_CROSS_BLOCK_PCT = 0.05   # 5% cross blocks

# Within-Block Allocation
SAMPLING_PROPORTIONAL_PCT = 0.80  # 80% by size
SAMPLING_FLOOR_PCT = 0.20         # 20% evenly

# Reproducibility
SAMPLING_RNG_SEED = 42            # Fixed seed
```

**Performance Improvements**:

Validated with 560-name dataset (from test_performance_validation.py):
- **Speed**: 3.6x faster (0.585s vs 2.124s)
- **Quality**: 4.3x better match representation (26% within-block vs 6%)
- **Threshold Separation**: 50% better (0.021 vs 0.015)
- **Coverage**: Unbiased sampling across all names

**Returned Metadata**:
```python
{
  "sampling_metadata": {
    "total_blocks_created": 85,
    "singletons_filtered": 12,
    "avg_block_size": 3.2,
    "max_block_size": 15,
    "within_block_pairs_sampled": 47500,
    "cross_block_pairs_sampled": 2500,
    "total_pairs_sampled": 50000,
    "sampling_time_seconds": 0.005,
    "phonetic_stats": {
      "skipped_digit": 3,
      "skipped_short": 8
    }
  }
}
```

**Usage Example**:
```python
from app.services.blocking_service import BlockingKeyGenerator, StratifiedReservoirSampler

# Generate blocking keys
key_generator = BlockingKeyGenerator()
blocking_keys = {name: key_generator.generate_key(name) for name in names}

# Sample pairs
sampler = StratifiedReservoirSampler(
    max_pairs=50000,
    rng_seed=42,
    within_block_pct=0.95,
    cross_block_pct=0.05
)
result = sampler.sample_pairs(names, blocking_keys)

# Access results
sampled_pairs = result['pairs']  # List of (name1, name2) tuples
metadata = result['metadata']    # Sampling statistics
```

**Configuration Tuning**:

1. **Within/Cross-Block Split**:
   - **More within-block (0.98/0.02)**: Focus on obvious similarities
   - **More cross-block (0.90/0.10)**: Discover rare matches
   - **Default (0.95/0.05)**: Balanced coverage

2. **Proportional/Floor Split**:
   - **More proportional (0.90/0.10)**: Favor large blocks
   - **More floor (0.70/0.30)**: Ensure small block representation
   - **Default (0.80/0.20)**: Balanced allocation

3. **Block Size Caps**:
   - **Min block size (2)**: Filter singletons (no pairs possible)
   - **Max block pairs (5000)**: Prevent single block domination
   - Adjust based on dataset size distribution

4. **RNG Seed**:
   - **Fixed (42)**: Reproducible results for testing/debugging
   - **Change value**: Get different (but still reproducible) samples
   - **None**: Non-deterministic (not recommended for production)

**Testing**:
```bash
cd backend

# Unit tests (29 tests for blocking service)
pytest tests/test_blocking_service.py -v

# Performance validation
python test_performance_validation.py

# Integration with GMM
python test_adaptive_workflow.py
```

**Logging**:
- **DEBUG**: Detailed allocation, reservoir steps, cross-block selection
- **INFO**: Block statistics, sampling progress (10% intervals), final counts

**When to Adjust Configuration**:
- **Large datasets (>1000 names)**: Keep defaults, caps prevent slowdown
- **Small datasets (<100 names)**: May not benefit (few blocks)
- **Domain-specific**: Adjust token/phonetic logic in BlockingKeyGenerator
- **Reproducibility critical**: Keep fixed seed, validate with tests

## API Contract

### POST /api/process

**Request**:
```
Content-Type: multipart/form-data
Field: file (CSV file)
Query Parameter (optional): use_adaptive_threshold (boolean, default: false)
```

**Example**:
```bash
# Fixed threshold mode (default)
POST /api/process
Form data: file=companies.csv

# Adaptive GMM mode
POST /api/process?use_adaptive_threshold=true
Form data: file=companies.csv
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

**Backend**: Fully tested and operational (21 tests passing)
- **Name Matcher**: 12/12 tests passing (fuzzy matching, phonetics)
- **GMM Service**: 9/9 tests passing (adaptive thresholding)

**Frontend**: UI tested manually in browser
**Integration**: Full stack tested with sample datasets

### Backend Tests (Verified ✅)

**Unit Tests - Name Matcher** (`tests/test_name_matcher.py`):
```bash
cd backend
pytest tests/test_name_matcher.py -v

# Results: 12/12 tests passing
# - test_normalize_name
# - test_select_canonical_name
# - test_calculate_confidence
# - test_group_similar_names
# - test_process_names
# - test_phonetic_agreement
# - test_phonetic_disagreement
# - test_phonetic_skip_numbers
# - test_phonetic_skip_acronyms
# - test_phonetic_accent_folding
# - test_should_use_phonetics
# - test_calculate_phonetic_bonus
```

**Unit Tests - GMM Service** (`tests/test_gmm_threshold_service.py`):
```bash
cd backend
pytest tests/test_gmm_threshold_service.py -v

# Results: 9/9 tests passing in 16s
# - test_fit_gmm_with_sufficient_data
# - test_fit_gmm_with_insufficient_data
# - test_calculate_posterior_probability
# - test_find_threshold_for_posterior
# - test_calculate_adaptive_thresholds
# - test_calculate_margin_penalty
# - test_get_gmm_metadata
# - test_fallback_with_insufficient_samples
# - test_threshold_boundary_conditions
```

**Integration Tests**:
```bash
cd backend

# Compare fixed vs adaptive modes
python test_adaptive_workflow.py

# Test with 500-name dataset
python test_sample_data_500.py
```

**What's Tested**:
- ✅ Name normalization (suffix removal, lowercase)
- ✅ Fuzzy matching with RapidFuzz (WRatio, token_set)
- ✅ Phonetic matching (Metaphone, accent folding)
- ✅ Grouping algorithm (clustering)
- ✅ Canonical name selection
- ✅ Confidence score calculation
- ✅ GMM fitting (2-component model)
- ✅ Posterior probability calculation
- ✅ Threshold finding (T_LOW, S_90, T_HIGH)
- ✅ Margin penalty (promotion zone)
- ✅ Three-zone decision logic
- ✅ Fallback behavior (insufficient data)
- ✅ Audit log generation
- ✅ CSV file validation
- ✅ Configuration loading
- ✅ All modular imports

**Test Coverage**:
- Services layer: Comprehensive (name_matcher + gmm_threshold_service)
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
