# Technology Stack

## Core Requirements
- **Language**: Python 3.12 (chosen over 3.11 for SnapStart support)
- **Time Constraint**: 4 hours
- **Deployment Target**: AWS Lambda + API Gateway
- **Infrastructure as Code**: AWS CDK (Python)

## Python 3.12 Decision

**Why Python 3.12 instead of 3.11?**
- ✅ **SnapStart Support**: Sub-second cold starts (vs 2-4s on 3.11)
- ✅ **All dependencies supported**: PyTorch, sentence-transformers, hdbscan all have wheels
- ✅ **5-10% performance improvement** over 3.11
- ✅ **Better error messages** and debugging experience
- ✅ **AWS Lambda fully supports** Python 3.12 runtime

**Why NOT Python 3.13?**
- ❌ PyTorch doesn't officially support 3.13 yet
- ❌ sentence-transformers requires PyTorch (blocker)
- ⚠️ Only nightly PyTorch builds available (unstable)

## Technology Choices

### 1. NLP & Machine Learning

#### Embeddings
- **Library**: `sentence-transformers`
- **Model**: `all-MiniLM-L6-v2`
- **Rationale**:
  - Fast inference (~100ms for 100 sentences)
  - Small model size (~80MB) - fits in Lambda
  - Good semantic similarity performance
  - No external API calls = predictable latency & cost

#### Clustering
- **Library**: `hdbscan` or `scikit-learn` (KMeans as fallback)
- **Rationale**:
  - HDBSCAN: Handles variable cluster sizes, identifies outliers
  - No need to specify cluster count in advance
  - Works well with embedding spaces
  - Pure Python/NumPy = Lambda compatible

#### Sentiment Analysis
- **Library**: `vaderSentiment` or `textblob`
- **Rationale**:
  - Lightweight, no training required
  - Designed for social media/short text
  - Rule-based = fast and deterministic
  - No API calls or large models

### 2. API & Infrastructure

#### AWS Services
- **Lambda**: Python 3.12 runtime
  - Memory: 3GB (for ML libraries)
  - Timeout: 900s / 15 minutes (target <10s actual)
  - Container Image: ML dependencies (sentence-transformers, etc.)

- **API Gateway**: REST API
  - Request validation
  - API key authentication
  - CORS enabled

- **CloudWatch**: Logging and basic monitoring

#### Infrastructure as Code
- **AWS CDK**: Python constructs
- **Rationale**:
  - Type-safe, IDE autocomplete
  - Same language as application code
  - L2 constructs simplify common patterns
  - Easy to version and review

### 3. Development & Testing

#### Validation
- **Pydantic v2**: Request/response validation
- Strong typing
- Automatic JSON schema generation

#### Testing
- **pytest**: Unit and integration tests
- **moto**: AWS service mocking
- **pytest-cov**: Coverage reporting

#### Local Development
- **AWS SAM CLI**: Local Lambda testing
- **Docker**: Consistent environment

### 4. Dependencies Summary

```txt
# Core ML/NLP
sentence-transformers==2.2.2
hdbscan==0.8.33
scikit-learn==1.3.2
numpy==1.24.3
vaderSentiment==3.3.2

# API & Validation
pydantic==2.5.0
fastapi==0.104.1  # (optional, for local testing)

# AWS
aws-cdk-lib==2.110.0
constructs>=10.0.0

# Development
pytest==7.4.3
pytest-cov==4.1.0
moto[lambda,apigateway]==4.2.9
```

## Deployment Architecture

```
┌─────────────────┐
│   API Gateway   │ ← REST API with API key auth
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lambda Function│ ← Python 3.12, 3GB RAM, 900s timeout
│  ┌───────────┐  │
│  │  Handler  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Clustering│  │ ← sentence-transformers + HDBSCAN
│  │   Engine  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Sentiment │  │ ← VADER
│  │  Analyzer │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Insights  │  │ ← Template-based from statistics
│  │ Generator │  │
│  └───────────┘  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   CloudWatch    │ ← Logs & metrics
└─────────────────┘
```

## Trade-offs & Decisions

### No External LLM Integration
**Decision**: Use rule-based insight generation instead of LLM APIs
**Rationale**:
- ✅ Faster: No API latency
- ✅ Cheaper: No per-request costs
- ✅ Predictable: Deterministic outputs
- ✅ Simpler: Fewer dependencies, less error handling
- ❌ Less sophisticated insights
- **For MVP**: Good enough. Can add LLM later if needed.

### Lambda vs Container
**Decision**: Use Lambda with layers
**Rationale**:
- ✅ Simpler deployment
- ✅ Auto-scaling
- ✅ Pay-per-use
- ❌ 10GB layer size limit (manageable with optimized deps)
- ❌ Cold start (~2-3s with ML libs)

### HDBSCAN vs KMeans
**Decision**: Try HDBSCAN first, KMeans as fallback
**Rationale**:
- HDBSCAN better for real-world data (variable density)
- Can identify noise/outliers
- If too slow or unstable, fall back to KMeans

## Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| Response time | <10s | Batch embeddings, efficient clustering |
| Cold start | <5s | Optimized dependencies, Lambda layers |
| Max input size | 1000 sentences | Validate at API Gateway |
| Cluster count | 3-10 | Tune HDBSCAN parameters |

## Future Enhancements (Out of Scope for MVP)

- [ ] Redis caching for embeddings
- [ ] DynamoDB for request/result persistence
- [ ] LLM integration for richer insights (Bedrock/Claude)
- [ ] Real-time streaming for large inputs
- [ ] Multi-model ensemble
- [ ] Fine-tuned embeddings for domain-specific use
- [ ] A/B testing framework
