# Development Plan (4-Hour Time-Boxed)

## Time Allocation Strategy

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 1**: Setup & Research | 30 min | Environment, dependencies, research | Project structure, requirements.txt |
| **Phase 2**: Core ML Logic | 90 min | Clustering, sentiment, insights | Working pipeline (local) |
| **Phase 3**: Lambda Handler & API | 45 min | API contract, validation, handler | Lambda function code |
| **Phase 4**: Infrastructure | 45 min | CDK setup, deployment | Deployed API endpoint |
| **Phase 5**: Testing & Docs | 30 min | Tests, README, final polish | Test suite, documentation |
| **Buffer** | 30 min | Debugging, unforeseen issues | - |

**Total**: 4 hours

---

## Phase 1: Setup & Research (30 min)

### Objectives
- Set up project structure
- Install dependencies locally
- Research latest library versions and APIs
- Validate approach feasibility

### Tasks

**1.1 Project Structure** (5 min)
```bash
backend-task-2025/
├── src/
│   ├── lambda_function.py          # Main handler
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Sentence transformer wrapper
│   │   ├── clusterer.py            # HDBSCAN logic
│   │   └── insights.py             # Insight generation
│   ├── sentiment/
│   │   ├── __init__.py
│   │   └── analyzer.py             # VADER wrapper
│   └── utils/
│       ├── __init__.py
│       ├── validators.py           # Pydantic models
│       └── formatters.py           # Response formatting
├── infrastructure/
│   ├── app.py                      # CDK app
│   ├── stacks/
│   │   └── lambda_stack.py         # Lambda + API Gateway stack
│   └── requirements.txt
├── tests/
│   ├── unit/
│   │   ├── test_clustering.py
│   │   ├── test_sentiment.py
│   │   └── test_insights.py
│   └── integration/
│       └── test_pipeline.py
├── data/                           # Sample data (already exists)
├── docs/                           # Planning docs (already exists)
├── requirements.txt                # Python dependencies
├── pytest.ini
└── README.md
```

**1.2 Parallel Research** (25 min)
Launch 6 agents in parallel to fetch latest documentation:
- sentence-transformers: Latest models, API, Lambda optimization
- HDBSCAN: Parameters, best practices
- AWS Lambda Python 3.11: Layers, packaging, limits
- AWS CDK Python: Lambda + API Gateway constructs
- Pydantic v2: Validation patterns
- VADER sentiment: API and usage

---

## Phase 2: Core ML Logic (90 min)

### Objectives
- Build and test clustering pipeline locally
- Implement sentiment analysis
- Create insight generation logic
- Test with provided sample data

### Tasks

**2.1 Embeddings Module** (15 min)
File: `src/clustering/embeddings.py`

```python
# Implement:
- SentenceEmbedder class
- Model caching for Lambda
- Batch encoding
- Error handling

# Test with:
- Sample sentences from data/input_example.json
- Measure: embedding generation time
```

**2.2 Clustering Module** (25 min)
File: `src/clustering/clusterer.py`

```python
# Implement:
- HDBSCAN clustering with adaptive parameters
- KMeans fallback
- Post-processing (filter small clusters, limit to 10)
- Noise handling

# Test with:
- Different dataset sizes (10, 50, 100, 500 sentences)
- Edge cases (all similar, all different)
- Measure: cluster quality (silhouette score)
```

**2.3 Sentiment Analysis** (15 min)
File: `src/sentiment/analyzer.py`

```python
# Implement:
- VADER wrapper
- Sentiment classification (positive/neutral/negative)
- Aggregate cluster sentiment
- Confidence scores

# Test with:
- Known positive/negative sentences
- Measure: accuracy on sample data
```

**2.4 Insight Generation** (20 min)
File: `src/clustering/insights.py`

```python
# Implement:
- Keyword extraction (TF-IDF)
- Cluster title generation
- Template-based insights
- Comparative analysis logic

# Test with:
- Single cluster
- Baseline vs comparison
```

**2.5 Integration & Local Testing** (15 min)
File: `src/clustering/pipeline.py`

```python
# Implement:
- End-to-end pipeline class
- Combine all modules
- Handle baseline-only and comparative modes

# Test with:
- data/input_example.json
- data/input_comparison_example.json
- Validate output format matches spec
```

**Success Criteria**:
- ✅ Processes 100 sentences in <5s locally
- ✅ Generates 3-10 clusters
- ✅ Produces readable titles and insights
- ✅ Handles comparison mode

---

## Phase 3: Lambda Handler & API (45 min)

### Objectives
- Create Lambda handler function
- Implement request/response validation
- Add error handling and logging
- Test locally with SAM

### Tasks

**3.1 Request/Response Models** (10 min)
File: `src/utils/validators.py`

```python
# Implement Pydantic models:
- SentenceInput (sentence, id)
- AnalysisRequest (baseline, comparison, query, theme)
- ClusterOutput (title, sentences, sentiment, insights)
- AnalysisResponse (clusters, summary, comparison_insights)

# Validation rules:
- Max 1000 sentences per request
- No duplicate IDs
- Sentence length 1-1000 chars
```

**3.2 Lambda Handler** (20 min)
File: `src/lambda_function.py`

```python
# Implement:
def handler(event, context):
    # 1. Parse and validate request
    # 2. Run clustering pipeline
    # 3. Format response
    # 4. Error handling (return 400/500 with details)
    # 5. Logging (request ID, duration, cluster count)

# Environment variables:
- EMBEDDING_MODEL (default: all-MiniLM-L6-v2)
- MIN_CLUSTER_SIZE (default: 3)
- MAX_CLUSTERS (default: 10)
- LOG_LEVEL (default: INFO)
```

**3.3 Response Formatter** (10 min)
File: `src/utils/formatters.py`

```python
# Implement:
- Format clusters to match output spec
- Add summary statistics
- Handle comparison insights
- Ensure JSON serializable
```

**3.4 Local Testing with SAM** (5 min)
```bash
# Create template.yaml for SAM local testing
# Test handler locally:
sam local invoke -e test_event.json

# Measure cold start time
# Verify output format
```

**Success Criteria**:
- ✅ Handler parses API Gateway event correctly
- ✅ Validation rejects invalid input
- ✅ Returns properly formatted JSON response
- ✅ Logs contain useful debugging info

---

## Phase 4: Infrastructure (45 min)

### Objectives
- Create CDK stack for Lambda + API Gateway
- Build and package dependencies as Lambda layer
- Deploy to AWS
- Test deployed endpoint

### Tasks

**4.1 CDK Stack Setup** (15 min)
File: `infrastructure/stacks/lambda_stack.py`

```python
# Implement:
- Lambda function with layers
- API Gateway REST API
- API key for authentication
- CloudWatch log group
- IAM roles (minimal permissions)

# Configuration:
- Runtime: Python 3.11
- Memory: 3GB
- Timeout: 120s
- Environment variables
```

**4.2 Lambda Layer Build** (15 min)
File: `infrastructure/build_layer.sh`

```bash
#!/bin/bash
# Build ML dependencies layer in Docker

docker run --rm \
  -v $(pwd):/var/task \
  public.ecr.aws/lambda/python:3.11 \
  pip install \
    -t /var/task/python \
    sentence-transformers \
    hdbscan \
    scikit-learn \
    vaderSentiment \
    --no-cache-dir

# Zip layer
cd python && zip -r ../layer.zip . && cd ..

# Upload to S3 or include in CDK asset
```

**4.3 Deployment** (10 min)
```bash
# Bootstrap CDK (if first time)
cd infrastructure
cdk bootstrap

# Deploy stack
cdk deploy --require-approval never

# Capture outputs:
# - API endpoint URL
# - API key
```

**4.4 Smoke Test** (5 min)
```bash
# Test deployed endpoint
curl -X POST \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @../data/input_example.json \
  https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/analyze

# Verify:
# - Returns 200
# - Response format correct
# - Logs appear in CloudWatch
```

**Success Criteria**:
- ✅ CDK deploys without errors
- ✅ API endpoint accessible
- ✅ Returns valid response for sample data
- ✅ Meets <10s latency target

---

## Phase 5: Testing & Documentation (30 min)

### Objectives
- Write unit tests for core logic
- Write integration test for full pipeline
- Update README with deployment and usage instructions
- Document trade-offs and future improvements

### Tasks

**5.1 Unit Tests** (15 min)
Files: `tests/unit/test_*.py`

```python
# Test coverage:
- Embedding generation (mock model)
- Clustering logic (small dataset)
- Sentiment analysis (known inputs)
- Keyword extraction
- Insight generation
- Request validation (Pydantic)

# Run: pytest tests/unit --cov=src --cov-report=term
# Target: >70% coverage
```

**5.2 Integration Test** (5 min)
File: `tests/integration/test_pipeline.py`

```python
# Test:
- Full pipeline with data/input_example.json
- Comparison mode with data/input_comparison_example.json
- Verify output structure
- Assert cluster count in expected range

# Mock Lambda environment variables
```

**5.3 README Update** (10 min)
File: `README.md`

```markdown
# Add sections:
- Architecture overview
- Prerequisites (AWS account, CDK, Docker)
- Local development setup
- Deployment instructions
- API usage examples (curl, Python)
- Testing instructions
- Trade-offs and future improvements
- Cost estimates
```

**Success Criteria**:
- ✅ All tests pass
- ✅ Clear deployment instructions
- ✅ API usage documented with examples
- ✅ Trade-offs explicitly stated

---

## Buffer Phase (30 min)

**Use for**:
- Debugging deployment issues
- Performance optimization if >10s
- Improving test coverage
- Polishing documentation
- Handling unexpected library incompatibilities

---

## Priority Levels

### Must Have (P0) - Core Functionality
- [x] Clustering pipeline works
- [x] Sentiment analysis accurate
- [x] Lambda handler functional
- [x] Deployed API endpoint
- [x] Handles sample data correctly

### Should Have (P1) - Production Basics
- [ ] Comprehensive error handling
- [ ] Unit test coverage >70%
- [ ] Clear documentation
- [ ] Logging and observability
- [ ] Input validation

### Nice to Have (P2) - Polish
- [ ] Optimal HDBSCAN parameters
- [ ] Rich insight templates
- [ ] Performance optimizations
- [ ] Integration tests
- [ ] API documentation (OpenAPI spec)

### Won't Have (For MVP)
- LLM integration
- Caching layer
- CI/CD pipeline
- Multi-region deployment
- Advanced monitoring/alerting
- Rate limiting beyond API Gateway defaults

---

## Risk Mitigation

### High-Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Lambda layer too large (>250MB) | Cannot deploy | Use slim torch, remove unnecessary deps |
| Clustering too slow (>10s) | Fails requirement | Reduce max input to 500 sentences, optimize batch size |
| HDBSCAN produces poor clusters | Bad results | Implement KMeans fallback |
| Cold start >5s | Bad UX | Optimize imports, consider provisioned concurrency |
| CDK deployment fails | Cannot deploy | Test locally with SAM first, have manual Lambda upload ready |

### Fallback Plans

1. **If HDBSCAN fails**: Use KMeans with k=5
2. **If layer too large**: Deploy dependencies in function package instead
3. **If latency >10s**: Reduce max input size to 300 sentences
4. **If CDK issues**: Deploy manually via AWS Console (not ideal but works)
5. **If sentiment library unavailable**: Implement simple keyword-based sentiment

---

## Success Metrics

### Functional Requirements
- ✅ Accepts baseline + comparison input
- ✅ Returns 3-10 clusters
- ✅ Each cluster has title, sentiment, insights
- ✅ Handles comparison mode
- ✅ Output matches specification

### Non-Functional Requirements
- ✅ Response time <10s for 500 sentences
- ✅ Deterministic outputs
- ✅ Proper error messages
- ✅ Deployable via IaC

### Code Quality
- ✅ Type hints throughout
- ✅ Pydantic validation
- ✅ Unit tests for core logic
- ✅ Documented functions
- ✅ Follows Python best practices

---

## Definition of Done

For this MVP to be considered "done":

1. ✅ All P0 (Must Have) items completed
2. ✅ Deployed API endpoint accessible
3. ✅ Successfully processes both sample JSON files
4. ✅ Response format matches specification
5. ✅ README contains deployment and usage instructions
6. ✅ Core logic has unit tests
7. ✅ Trade-offs documented

**Not required for "done"**:
- 100% test coverage
- Production-grade monitoring
- CI/CD pipeline
- All edge cases handled

---

## Post-4-Hour Roadmap

If more time were available, next priorities would be:

### Week 1: Production Hardening
- Add DynamoDB for request caching
- Implement comprehensive logging/monitoring
- Set up CloudWatch alarms
- Add input sanitization for PII
- Implement rate limiting

### Week 2: Quality Improvements
- Fine-tune HDBSCAN parameters per domain
- Add LLM integration for richer insights (Bedrock)
- Improve keyword extraction (use noun phrases)
- Add support for multi-language

### Week 3: Scaling & Optimization
- Implement async processing for large datasets
- Add SQS queue for background jobs
- Use Step Functions for orchestration
- Optimize cold starts (provisioned concurrency)
- Add Redis caching layer

### Week 4: DevOps & Observability
- Full CI/CD pipeline (GitHub Actions)
- Automated testing in staging
- Canary deployments
- Distributed tracing (X-Ray)
- Custom metrics dashboard

---

## Tools & Resources

### Development Tools
- **IDE**: VS Code with Python extension
- **AWS**: AWS CLI, CDK CLI, SAM CLI
- **Testing**: pytest, moto
- **Docker**: For Lambda layer builds

### Key Documentation
- sentence-transformers: https://www.sbert.net/
- HDBSCAN: https://hdbscan.readthedocs.io/
- AWS Lambda Python: https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
- AWS CDK Python: https://docs.aws.amazon.com/cdk/api/v2/python/
- Pydantic: https://docs.pydantic.dev/2.0/
- VADER: https://github.com/cjhutto/vaderSentiment

### Sample Commands Reference

```bash
# Local development
python -m pip install -r requirements.txt
pytest tests/ -v
python -m src.lambda_function  # Local test

# Build layer
./infrastructure/build_layer.sh

# Deploy
cd infrastructure
cdk synth  # Preview CloudFormation
cdk deploy

# Test
curl -X POST $API_URL/analyze \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d @data/input_example.json

# Logs
aws logs tail /aws/lambda/TextAnalysisFunction --follow

# Cleanup
cdk destroy
```

---

## Time Tracking Checkpoints

Track progress against plan:

- [ ] **30 min**: Project structure created, dependencies installed, research complete
- [ ] **2h 00m**: Core ML pipeline working locally, tested with sample data
- [ ] **2h 45m**: Lambda handler implemented, SAM local testing successful
- [ ] **3h 30m**: Deployed to AWS, smoke test passing
- [ ] **4h 00m**: Tests written, README updated, MVP complete

If behind schedule at any checkpoint, cut P2 (Nice to Have) items and focus on P0/P1.

---

## Final Deliverables Checklist

- [ ] Working API endpoint (deployed)
- [ ] Source code (src/ directory)
- [ ] Infrastructure code (CDK)
- [ ] Tests (unit + integration)
- [ ] Documentation:
  - [ ] README.md (setup, deployment, usage)
  - [ ] TECH_STACK.md (complete)
  - [ ] ARCHITECTURE.md (complete)
  - [ ] IMPLEMENTATION_APPROACH.md (complete)
  - [ ] This file (DEVELOPMENT_PLAN.md)
- [ ] Sample requests and responses
- [ ] Trade-offs document

**What's deliberately NOT included**:
- CI/CD configuration
- Production monitoring setup
- Comprehensive integration tests
- Performance benchmarking suite
- Security audit
- Multi-environment setup (dev/staging/prod)
