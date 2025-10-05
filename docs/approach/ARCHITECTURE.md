# System Architecture

## Overview

A serverless microservice that performs text clustering, sentiment analysis, and insight generation on customer feedback sentences.

## High-Level Architecture

```
┌──────────────┐
│   Client     │
│ Application  │
└──────┬───────┘
       │ HTTPS POST
       │ { baseline: [...], comparison: [...], query, theme }
       ▼
┌──────────────────────────────────────────────────────────┐
│              API Gateway (REST API)                       │
│  - Request validation (size, schema)                     │
│  - API key authentication                                 │
│  - CORS configuration                                     │
│  - Rate limiting (future)                                 │
└──────────────┬───────────────────────────────────────────┘
               │
               │ Invoke
               ▼
┌──────────────────────────────────────────────────────────┐
│           Lambda Function (Python 3.12)                   │
│  Runtime: 3GB RAM, 900s timeout (15 minutes)             │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │            lambda_function.handler()                │ │
│  │  1. Validate input (Pydantic)                      │ │
│  │  2. Combine baseline + comparison (if present)     │ │
│  │  3. Call clustering pipeline                        │ │
│  │  4. Format response                                 │ │
│  └─────────────────┬──────────────────────────────────┘ │
│                    │                                     │
│  ┌─────────────────▼──────────────────────────────────┐ │
│  │         Clustering Pipeline                         │ │
│  │  ┌──────────────────────────────────────────────┐  │ │
│  │  │  1. Embedding Generation                      │  │ │
│  │  │     - sentence-transformers                   │  │ │
│  │  │     - Model: all-MiniLM-L6-v2                │  │ │
│  │  │     - Batch processing                        │  │ │
│  │  └────────────────┬─────────────────────────────┘  │ │
│  │                   │                                 │ │
│  │  ┌────────────────▼─────────────────────────────┐  │ │
│  │  │  2. Clustering                                │  │ │
│  │  │     - HDBSCAN (primary)                      │  │ │
│  │  │     - KMeans (fallback)                      │  │ │
│  │  │     - Outlier detection                       │  │ │
│  │  └────────────────┬─────────────────────────────┘  │ │
│  │                   │                                 │ │
│  │  ┌────────────────▼─────────────────────────────┐  │ │
│  │  │  3. Post-processing                           │  │ │
│  │  │     - Filter small clusters (<3 items)       │  │ │
│  │  │     - Merge similar clusters (optional)       │  │ │
│  │  │     - Limit to max 10 clusters               │  │ │
│  │  └────────────────┬─────────────────────────────┘  │ │
│  │                   │                                 │ │
│  │  ┌────────────────▼─────────────────────────────┐  │ │
│  │  │  4. Per-Cluster Analysis                      │  │ │
│  │  │     a. Sentiment Analysis (VADER)            │  │ │
│  │  │     b. Keyword Extraction (TF-IDF)           │  │ │
│  │  │     c. Title Generation (top keywords)       │  │ │
│  │  │     d. Key Insights (templates)              │  │ │
│  │  └────────────────┬─────────────────────────────┘  │ │
│  │                   │                                 │ │
│  │  ┌────────────────▼─────────────────────────────┐  │ │
│  │  │  5. Comparative Analysis (if comparison)      │  │ │
│  │  │     - Identify baseline-only clusters        │  │ │
│  │  │     - Identify comparison-only clusters      │  │ │
│  │  │     - Identify shared themes                  │  │ │
│  │  └──────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Lambda Layers                               │ │
│  │  - sentence-transformers + dependencies            │ │
│  │  - scikit-learn, numpy, scipy                      │ │
│  │  - hdbscan                                          │ │
│  │  Total size: ~250MB (under 512MB limit)           │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────────────────┘
               │
               │ Logs
               ▼
┌──────────────────────────────────────────────────────────┐
│                    CloudWatch Logs                        │
│  - Request/response logging                              │
│  - Performance metrics (duration, memory)                │
│  - Error tracking                                         │
└──────────────────────────────────────────────────────────┘
```

## Request Flow

### 1. API Request Format

```json
{
  "baseline": [
    {
      "sentence": "The app crashes frequently",
      "id": "uuid-1"
    }
  ],
  "comparison": [
    {
      "sentence": "The new version is stable",
      "id": "uuid-2"
    }
  ],
  "query": "overview",
  "surveyTitle": "Product Feedback Q1 2025",
  "theme": "app stability"
}
```

### 2. Processing Steps

**Step 1: Validation** (API Gateway + Lambda)
- Check payload size (<6MB)
- Validate JSON schema
- Check sentence count (max 1000)
- Validate required fields

**Step 2: Embedding Generation** (~2-4s for 100 sentences)
```python
embeddings = model.encode(
    sentences,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True
)
```

**Step 3: Clustering** (~1-2s)
```python
# HDBSCAN approach
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=max(3, len(sentences) // 20),
    min_samples=2,
    metric='euclidean'
)
labels = clusterer.fit_predict(embeddings)
```

**Step 4: Cluster Analysis** (parallel per cluster, ~1-2s total)
For each cluster:
- Extract sentences
- Calculate sentiment scores
- Generate TF-IDF keywords
- Create title from top 3 keywords
- Generate insight templates

**Step 5: Response Formation**

### 3. Response Format

```json
{
  "clusters": [
    {
      "id": "cluster-1",
      "title": "App Crashes & Stability Issues",
      "sentences": [
        {
          "sentence": "The app crashes frequently",
          "id": "uuid-1",
          "sentiment": {
            "label": "negative",
            "score": -0.72
          }
        }
      ],
      "size": 15,
      "sentiment": {
        "overall": "negative",
        "distribution": {
          "positive": 2,
          "neutral": 1,
          "negative": 12
        },
        "average_score": -0.68
      },
      "key_insights": [
        "87% of feedback mentions crashes or freezing",
        "Most common issue: app unresponsive after login",
        "20% mention data loss during crash"
      ],
      "keywords": ["crash", "freeze", "unresponsive", "stability"],
      "source": "baseline"
    }
  ],
  "summary": {
    "total_sentences": 100,
    "clusters_found": 5,
    "unclustered": 8,
    "overall_sentiment": "negative",
    "query": "overview",
    "theme": "app stability"
  },
  "comparison_insights": {
    "baseline_only_themes": ["frequent crashes"],
    "comparison_only_themes": ["improved stability"],
    "shared_themes": ["loading times"]
  }
}
```

## AWS Infrastructure Components

### Lambda Function Configuration

```python
# CDK definition
lambda_function = _lambda.Function(
    self, "TextAnalysisFunction",
    runtime=_lambda.Runtime.PYTHON_3_11,
    handler="lambda_function.handler",
    code=_lambda.Code.from_asset("src"),
    memory_size=3008,  # 3GB
    timeout=Duration.seconds(900),  # 15 minutes
    environment={
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "MIN_CLUSTER_SIZE": "3",
        "MAX_CLUSTERS": "10",
        "LOG_LEVEL": "INFO"
    },
    layers=[ml_dependencies_layer]
)
```

### API Gateway Configuration

```python
api = apigateway.RestApi(
    self, "TextAnalysisAPI",
    rest_api_name="Text Analysis Service",
    description="Clustering and sentiment analysis for text data",
    deploy_options={
        "stage_name": "prod",
        "throttling_rate_limit": 10,
        "throttling_burst_limit": 20
    }
)

# Resource: /analyze
analyze_resource = api.root.add_resource("analyze")

# POST /analyze with API key
analyze_resource.add_method(
    "POST",
    apigateway.LambdaIntegration(lambda_function),
    api_key_required=True
)
```

### Lambda Layer (ML Dependencies)

**Contents**:
```
python/
├── sentence_transformers/
├── transformers/
├── torch/ (CPU-only)
├── sklearn/
├── hdbscan/
├── numpy/
└── scipy/
```

**Build process**:
```bash
# Build in Docker to match Lambda environment
docker run --rm -v $(pwd):/var/task \
  public.ecr.aws/lambda/python:3.11 \
  pip install -t /var/task/python \
  sentence-transformers hdbscan scikit-learn vaderSentiment
```

## Security Architecture

### Authentication & Authorization
- API Gateway API keys (for MVP)
- Future: AWS IAM, Cognito JWT tokens

### Data Security
- All data in transit encrypted (HTTPS)
- No data persistence (stateless processing)
- Logs scrubbed of PII

### IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "arn:aws:logs:*:*:*"
}
```

## Performance Characteristics

### Expected Latency

| Sentences | Embedding | Clustering | Analysis | Total  |
|-----------|-----------|------------|----------|--------|
| 50        | 1-2s      | 0.5s       | 0.5s     | 2-3s   |
| 100       | 2-3s      | 1s         | 1s       | 4-5s   |
| 500       | 5-7s      | 2s         | 1.5s     | 8-10s  |
| 1000      | 10-12s    | 3s         | 2s       | ~15s ⚠️ |

### Optimization Strategies
1. **Batch processing**: Encode all sentences in one call
2. **Lazy loading**: Import ML libs only when needed
3. **Caching**: Keep model in global scope (Lambda warm starts)
4. **Parallelization**: Process clusters concurrently

### Cold Start Mitigation
- Provisioned concurrency (not in MVP)
- Keep Lambda warm with scheduled pings (not in MVP)
- Optimize layer size

## Scalability

### Current Design
- **Concurrent executions**: 10 (default account limit: 1000)
- **Requests/second**: ~10-20 (with avg 1s duration)
- **Cost**: ~$0.0001 per request (3GB * 5s)

### Future Scaling
- Increase Lambda reserved concurrency
- Add SQS queue for async processing
- Use Step Functions for orchestration
- Add DynamoDB for request deduplication

## Monitoring & Observability

### CloudWatch Metrics
- Invocation count
- Duration (p50, p95, p99)
- Error rate
- Throttles

### Custom Metrics
- Sentences processed
- Clusters generated
- Average cluster size
- Unclustered ratio

### Alarms
- Error rate >5%
- Duration >10s (p95)
- Throttles >0

## Deployment Strategy

### Environments
- **dev**: Development testing
- **staging**: Pre-production validation
- **prod**: Production (MVP: just prod)

### CI/CD (Future)
1. PR → Unit tests
2. Merge → Deploy to staging
3. Manual approval → Deploy to prod
4. Rollback capability

### Rollout (MVP: Manual)
```bash
cd infrastructure
cdk bootstrap  # One-time
cdk deploy
```

## Disaster Recovery

### Backup Strategy
- Infrastructure as Code (CDK) in git
- No stateful resources to back up

### Recovery Plan
- Redeploy from CDK (RTO: <10 min)
- API Gateway URLs stable (no client changes)

## Cost Estimation

### Per Request (100 sentences, 5s duration, 3GB)
- Lambda: $0.00008
- API Gateway: $0.0000035
- **Total**: ~$0.00009 per request

### Monthly (10,000 requests)
- Lambda: $0.80
- API Gateway: $0.035
- **Total**: ~$0.84/month

**Note**: This excludes AWS Free Tier benefits.
