# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **serverless text analysis microservice** deployed on AWS Lambda that performs semantic clustering, sentiment analysis, and generates actionable insights from text data. It's designed for analyzing customer feedback and grouping similar sentences into thematic clusters.

**Deployed API Endpoint:** `https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze`

## Commonly Used Commands

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run specific test file
pytest tests/test_data_examples.py -v

# Run with markers
pytest -m "not slow" -v
```

### Development
```bash
# Set up virtual environment (first time)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt

# Local Lambda handler test
python src/lambda_function.py

# Build Docker image locally
docker build -t text-analysis .

# Run container locally
docker run -p 9000:8080 text-analysis
```

### Deployment
```bash
# Install infrastructure dependencies
pip install -r infrastructure/requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap

# Synthesize CloudFormation template
cdk synth

# Deploy to AWS
cdk deploy --require-approval never

# Destroy stack (cleanup)
cdk destroy
```

### Linting & Code Quality
```bash
# No explicit linters configured - relies on type hints and tests
# Consider adding: black, flake8, mypy for code quality
```

## High-Level Architecture

### Request Flow
```
API Gateway → Lambda Function (Docker) → CloudWatch Logs
```

The system follows a **pipeline architecture** that processes text in stages:

```
Input → Validation → Embeddings → Clustering → Sentiment → Insights → Response
```

### Key Components

1. **Entry Point** (`src/lambda_function.py`)
   - Main Lambda handler with lazy loading pattern
   - Caches ML models globally across warm invocations
   - Orchestrates the complete analysis pipeline
   - Critical optimization: Heavy ML imports deferred to invocation phase (not init phase) to avoid 10s timeout

2. **ML Pipeline** (4 independent stages)
   - **Embeddings** (`clustering/embeddings.py`): Converts sentences to 384-dim vectors using sentence-transformers
   - **Clustering** (`clustering/clusterer.py`): UMAP dimensionality reduction → HDBSCAN clustering → noise reassignment
   - **Sentiment** (`sentiment/analyzer.py`): VADER rule-based sentiment (cluster-level + sentence-level)
   - **Insights** (`clustering/insights.py`): TF-IDF keyword extraction + statistical insights

3. **Validation & Formatting** (`utils/`)
   - `validators.py`: Pydantic models for type-safe request validation
   - `formatters.py`: Response formatting with summary statistics

4. **Infrastructure** (`infrastructure/`)
   - CDK stack provisions Lambda + API Gateway + IAM + CloudWatch
   - Uses **Docker container deployment** (not layers) due to 250MB layer limit
   - 3GB memory, 900s timeout (15 minutes), 512MB ephemeral storage for ML workloads

### Critical Design Patterns

**Lazy Loading**: ML modules imported during invocation (900s timeout) not init (10s timeout). This reduced cold start from 10s+ to ~990ms.

**Global Caching**: Models loaded once per Lambda container, reused across warm invocations. Cache variables prefixed with `_` (e.g., `_embedder`, `_clusterer`).

**Noise Reassignment**: HDBSCAN produces "noise points" (cluster label -1). All noise is reassigned to nearest cluster so users see all sentences categorized.

**Ephemeral Storage Strategy**: All model caches redirect to `/tmp` (Lambda's only writable directory). Environment variables set `TRANSFORMERS_CACHE`, `HF_HOME`, etc.

## Input/Output Formats

### Input (Standalone Analysis)
```json
{
  "surveyTitle": "Product Feedback",
  "theme": "customer experience",
  "baseline": [
    {"sentence": "Great product!", "id": "1"},
    {"sentence": "Love the app", "id": "2"}
  ],
  "query": "overview"
}
```

### Input (Comparison Analysis)
```json
{
  "surveyTitle": "Product Feedback",
  "theme": "customer experience",
  "baseline": [...],
  "comparison": [...],  // Optional
  "query": "overview"
}
```

### Output
```json
{
  "clusters": [
    {
      "id": "baseline-cluster-0",
      "title": "Product Quality",
      "sentences": ["1", "2"],
      "size": 2,
      "sentiment": {
        "overall": "positive",
        "distribution": {"positive": 2, "neutral": 0, "negative": 0},
        "average_score": 0.8
      },
      "key_insights": ["100% positive sentiment", "Strong praise for quality"],
      "keywords": ["great", "product", "quality"],
      "source": "baseline"
    }
  ],
  "summary": {
    "total_sentences": 2,
    "clusters_found": 1,
    "overall_sentiment": "positive"
  },
  "request_id": "abc-123"
}
```

## Important Implementation Notes

### Validation Rules
- **Duplicate IDs rejected**: All example data files contain duplicate IDs and fail validation with 400 errors
- **Empty arrays allowed**: `"comparison": []` is valid (treated as no comparison mode)
- **All fields required**: `baseline`, `query`, `surveyTitle`, `theme` are mandatory

### Performance Characteristics
- **Cold start**: ~4-5s (model download + initialization)
- **Warm start**: <3s for 100 sentences, <10s for 500 sentences
- **Memory**: 3008 MB (3GB) optimal for ML workloads
- **Timeout**: 900s maximum (15 minutes)

### Test Data Issues
The provided example files in `data/` directory contain **duplicate sentence IDs** and will fail validation. Tests in `tests/test_data_examples.py` verify this validation correctly rejects invalid data.

### ML Model Choices
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU-optimized)
- **Clustering**: UMAP (384→5 dims) + HDBSCAN (min_cluster_size=2, max_clusters=10)
- **Sentiment**: VADER (rule-based, no model loading needed)

## Project Structure

```
src/
├── lambda_function.py       # Main handler with lazy loading
├── clustering/
│   ├── embeddings.py        # Sentence embedding generation
│   ├── clusterer.py         # UMAP + HDBSCAN clustering
│   └── insights.py          # TF-IDF keyword extraction
├── sentiment/
│   └── analyzer.py          # VADER sentiment analysis
└── utils/
    ├── validators.py        # Pydantic request validation
    └── formatters.py        # Response formatting

infrastructure/
├── app.py                   # CDK app entry point
└── stacks/
    └── lambda_stack.py      # Lambda + API Gateway stack

tests/
├── unit/                    # Unit tests with mocked ML models
├── integration/             # End-to-end Lambda handler tests
└── test_data_examples.py    # Tests against example data files
```

## Testing Strategy

- **Unit Tests**: Mock ML models to test logic without heavy dependencies
- **Integration Tests**: Full Lambda handler invocation with `moto` for AWS mocking
- **Coverage Target**: 70% minimum (configured in `pytest.ini`)
- **Markers**: `@pytest.mark.slow` for tests that take >2s

## Deployment Pipeline

**CI/CD**: GitHub Actions (`.github/workflows/deploy.yml`)

**Stages**:
1. **Test** (on push to main)
   - Install Python 3.12 + dependencies
   - Run pytest with coverage
   - Fail fast if tests don't pass

2. **Deploy** (after tests pass)
   - Set up Docker Buildx for x86_64
   - Configure AWS credentials
   - CDK synth → CDK deploy
   - Extract outputs (API endpoint, Lambda ARN)
   - Test deployed endpoint with curl

**Required Secrets**:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Key Learnings from Implementation

1. **Docker containers preferred over Lambda layers** when ML dependencies exceed 250MB
2. **Lazy loading critical** for Lambda init timeout (10s) - defer heavy imports to invocation
3. **HDBSCAN noise reassignment** improves UX - users expect all sentences categorized
4. **Pydantic validation** catches duplicate IDs and malformed requests early
5. **Global model caching** essential for warm start performance

## Known Issues & Limitations

1. **Example data files invalid**: All 3 files in `data/` have duplicate IDs
2. **Cold start performance**: First invocation ~4-5s (model download)
3. **Clustering non-determinism**: HDBSCAN/UMAP has slight randomness across runs
4. **No authentication**: API is public (should add API keys for production)

## Future Enhancements (Priority Order)

**High Priority**:
- Lambda SnapStart for Python 3.12 (80% cold start reduction)
- S3 model cache (2-3s faster cold starts)
- CloudWatch custom metrics for monitoring

**Medium Priority**:
- Memory right-sizing (potential 20-30% cost savings)
- ElastiCache for embedding cache (50%+ performance boost)
- LLM-generated insights (better than TF-IDF keywords)

**Low Priority**:
- Step Functions for 1000+ sentence datasets
- VPC integration for private data sources
- ARM64 migration (20% cost savings)

## Reference Documentation

- Comprehensive implementation details: `docs/SUMMARY.md`
- Architecture diagrams: `docs/ARCHITECTURE_DIAGRAMS.md`
- Original task requirements: `README.md`
