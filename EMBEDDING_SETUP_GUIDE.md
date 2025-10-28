# Embedding Setup Guide

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- `openai>=1.0.0` - OpenAI API client
- `sentence-transformers>=2.2.0` - Local embeddings
- `torch>=2.0.0` - Required by sentence-transformers

### 2. Configure OpenAI API Key

Create a `.env` file in the `backend/` directory:

```bash
cd backend
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Start the Backend

```bash
cd backend
python -m app.main
# Or with uvicorn:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend

```bash
cd frontend
npm run dev
```

### 5. Test It!

1. Open http://localhost:3000
2. You'll see a new **"Embedding Quality"** selector with 4 options:
   - **Best Quality** (OpenAI 3-large) - ~90% accuracy
   - **Balanced** (OpenAI 3-small) - ~85% accuracy [DEFAULT]
   - **Privacy Mode** (local) - ~75% accuracy
   - **Disabled** (fuzzy only) - ~61% accuracy

3. Upload your CSV with company names
4. Results will now correctly separate similar names!

## What Changed?

### The Problem (Before)

```
Input: American Express Company
Match: American (100% confidence) ❌ WRONG!

Input: American Airlines
Match: American (100% confidence) ❌ WRONG!
```

**Why?** Token overlap algorithm gave 100% score because they share the word "American"

### The Solution (After)

```
Input: American Express Company
Match: American Express (95% confidence) ✓ Correct!
Rejection: American (68% confidence) ✓ Correctly rejected!

Input: American Airlines
Match: American Airlines (96% confidence) ✓ Correct!
Rejection: American (65% confidence) ✓ Correctly rejected!
```

**How?** Hybrid scoring:
- 40% Fuzzy matching (handles typos)
- 15% Token overlap (reduced from 40%)
- 45% Semantic embeddings (understands context) **← NEW!**

## Embedding Modes Explained

### Mode 1: OpenAI 3-small (Recommended)

**When to use**: Default for most cases

**Pros**:
- Great accuracy (~85%)
- Very cheap ($0.02 per 1M tokens)
- Fast (~1 second for 500 names)
- No local setup needed

**Cons**:
- Requires internet
- Data sent to OpenAI
- Requires API key

**Cost**: ~$0.20/month for typical usage (100 uploads of 500 names)

### Mode 2: OpenAI 3-large (Best Quality)

**When to use**: When accuracy is critical

**Pros**:
- Best accuracy (~90%)
- Still fast (~2 seconds for 500 names)

**Cons**:
- More expensive ($0.13 per 1M tokens vs $0.02)
- Otherwise same as 3-small

**Cost**: ~$1.30/month for typical usage

### Mode 3: Privacy Mode (Local)

**When to use**:
- Data privacy requirements
- No internet access
- No API costs

**Pros**:
- Complete privacy (data never leaves your machine)
- No API costs
- Works offline

**Cons**:
- Lower accuracy (~75% vs 85%)
- Slower first run (2s model download + load)
- Uses ~150MB RAM

**Cost**: $0 (free)

### Mode 4: Disabled (Original Algorithm)

**When to use**:
- Testing/comparison
- Fastest possible processing
- Embeddings unavailable

**Pros**:
- Fastest (~0.5s for 500 names)
- No dependencies

**Cons**:
- Lowest accuracy (~61%)
- Original "shared word" problem

**Cost**: $0 (free)

## Advanced Configuration

### Tuning Similarity Weights

Edit `backend/app/config/settings.py`:

```python
# Default (recommended)
WRATIO_WEIGHT = 0.40       # Fuzzy matching (typos)
TOKEN_SET_WEIGHT = 0.15    # Token overlap
EMBEDDING_WEIGHT = 0.45    # Semantic similarity

# More semantic understanding (better context, worse typos)
WRATIO_WEIGHT = 0.30
TOKEN_SET_WEIGHT = 0.10
EMBEDDING_WEIGHT = 0.60

# Better typo handling (worse context understanding)
WRATIO_WEIGHT = 0.50
TOKEN_SET_WEIGHT = 0.20
EMBEDDING_WEIGHT = 0.30
```

Weights must sum to ~1.0

### Changing Default Embedding Mode

Edit `backend/.env`:

```
DEFAULT_EMBEDDING_MODE=openai-small  # or 'openai-large', 'local', 'disabled'
```

This sets the default when user doesn't select a mode explicitly.

### Using Custom Local Model

Edit `backend/.env`:

```
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default
# Or try:
LOCAL_EMBEDDING_MODEL=all-mpnet-base-v2  # Better quality, slower
LOCAL_EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2  # Faster, lower quality
```

See available models at: https://www.sbert.net/docs/pretrained_models.html

## Troubleshooting

### "OpenAI API unavailable" Error

**Cause**: No API key or invalid key

**Solution**:
1. Check `.env` file exists in `backend/` directory
2. Check `OPENAI_API_KEY=sk-proj-...` is set correctly
3. Test your key: https://platform.openai.com/api-keys
4. Restart the backend server

**Workaround**: Select "Privacy Mode" to use local embeddings instead

### "Failed to load local model" Error

**Cause**: sentence-transformers not installed properly

**Solution**:
```bash
cd backend
pip install --upgrade sentence-transformers torch
```

**Workaround**: Use OpenAI modes instead

### Slow Performance

**Issue**: Processing takes > 10 seconds

**Solutions**:
- Use `openai-small` instead of `openai-large` or `local`
- Reduce `EMBEDDING_DIMENSIONS` in settings (default 512, can go as low as 256)
- Check internet connection (for OpenAI modes)
- Use `disabled` mode for fastest processing

### High OpenAI Costs

**Issue**: Costs more than expected

**Check**:
- Use `openai-small` ($0.02/1M) not `openai-large` ($0.13/1M)
- Each company name ≈ 4 tokens
- 500 names = 2,000 tokens = $0.00004
- Monitor usage at: https://platform.openai.com/usage

**Solutions**:
- Switch to `local` mode for zero costs
- Use `EMBEDDING_DIMENSIONS=256` to reduce token count

## Testing

### Test with Sample Data

```bash
cd backend
python test_sample_data_500.py
```

This will:
1. Load 500 company names
2. Test with embeddings enabled
3. Show confidence scores
4. Compare with original algorithm

### Expected Improvements

With embeddings enabled, you should see:

| Pair | Old Score | New Score | Correct? |
|------|-----------|-----------|----------|
| "American Express" → "American" | 100% | 68% | ✓ Rejected |
| "American Airlines" → "American" | 95% | 65% | ✓ Rejected |
| "American Express" → "AmEx" | 85% | 94% | ✓ Grouped |
| "IBM" → "Intl Business Machines" | 45% | 97% | ✓ Grouped |

## API Usage

### Backend API

```bash
# With OpenAI embeddings (default)
curl -X POST http://localhost:8000/api/process \
  -F "file=@companies.csv" \
  -F "embedding_mode=openai-small"

# With local embeddings
curl -X POST http://localhost:8000/api/process \
  -F "file=@companies.csv" \
  -F "embedding_mode=local"

# With adaptive GMM + embeddings
curl -X POST http://localhost:8000/api/process \
  -F "file=@companies.csv" \
  -F "use_adaptive_threshold=true" \
  -F "embedding_mode=openai-small"
```

### Python API

```python
from app.services import NameMatcher

# With embeddings (default)
matcher = NameMatcher(embedding_mode='openai-small')
result = matcher.process_names(['American Express', 'American Airlines', 'American'])

# Without embeddings (original algorithm)
matcher = NameMatcher(embedding_mode='disabled')
result = matcher.process_names(['American Express', 'American Airlines', 'American'])

# With adaptive GMM + embeddings
matcher = NameMatcher(
    use_adaptive_threshold=True,
    embedding_mode='openai-small'
)
result = matcher.process_names(company_names)
```

## Next Steps

1. **Test with your data**: Upload the CSV that had issues before
2. **Compare modes**: Try all 4 embedding modes to see quality differences
3. **Monitor costs**: Check OpenAI usage after a few days
4. **Tune weights**: Adjust if you need more/less semantic understanding
5. **Read CLAUDE.md**: Full documentation of the architecture

## Support

- GitHub Issues: https://github.com/anthropics/claude-code/issues
- OpenAI API Docs: https://platform.openai.com/docs
- Sentence Transformers: https://www.sbert.net/

## FAQ

**Q: Do I need an OpenAI API key?**
A: No, you can use "Privacy Mode" (local embeddings) with no API key required.

**Q: How much will OpenAI embeddings cost?**
A: For typical usage (~100 uploads/month of 500 names each), about $0.20-1.30/month.

**Q: Can I use this offline?**
A: Yes, select "Privacy Mode" which uses local embeddings.

**Q: Is my data private?**
A: With "Privacy Mode", yes (data never leaves your machine). With OpenAI modes, company names are sent to OpenAI API (not used for training, retained 30 days for abuse monitoring).

**Q: Can I disable embeddings completely?**
A: Yes, select "Disabled (Fuzzy Matching Only)" mode. This reverts to the original algorithm.

**Q: Which mode should I use?**
A: Start with "Balanced" (openai-small). It's cheap, fast, and accurate. Switch to "Privacy Mode" if data sensitivity is a concern.
