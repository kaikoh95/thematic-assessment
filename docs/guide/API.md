# API Documentation

Complete reference for the Text Analysis Microservice API.

## Base URL

```
https://{api-id}.execute-api.{region}.amazonaws.com/prod
```

Replace `{api-id}` and `{region}` with your deployment values (shown in CDK outputs).

## Authentication

Currently no authentication required. For production, consider adding:
- API Gateway API keys
- IAM authentication
- Lambda authorizers (Cognito, custom)

## Endpoints

### POST /analyze

Analyzes customer feedback sentences using semantic clustering and sentiment analysis.

**Endpoint:** `POST /analyze`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "baseline": [
    {
      "sentence": "Customer feedback text",
      "id": "unique-identifier"
    }
  ],
  "comparison": [
    {
      "sentence": "Optional comparison text",
      "id": "unique-identifier"
    }
  ],
  "query": "overview",
  "surveyTitle": "Optional survey title",
  "theme": "Optional theme/category"
}
```

**Request Schema:**

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `baseline` | Array<SentenceInput> | Yes | Main dataset to analyze | 1-1000 sentences |
| `comparison` | Array<SentenceInput> | No | Optional comparison dataset | 0-1000 sentences |
| `query` | String | No | Context/query for analysis | Default: "overview", Max: 100 chars |
| `surveyTitle` | String | No | Title of feedback collection | Max: 200 chars |
| `theme` | String | No | Theme/category of feedback | Max: 100 chars |

**SentenceInput Schema:**

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `sentence` | String | Yes | Feedback text | 1-1000 characters, non-empty after trim |
| `id` | String | Yes | Unique identifier | 1-100 characters, unique within dataset |

**Constraints:**

- **Total sentences:** Maximum 1000 (baseline + comparison combined)
- **Duplicate IDs:** Not allowed within baseline or comparison datasets
- **Empty sentences:** Not allowed (whitespace-only rejected)
- **Request size:** Recommended < 1MB

**Response (200 OK):**

```json
{
  "clusters": [
    {
      "id": "baseline-cluster-0",
      "title": "Money & Withdrawal Issues",
      "sentences": [
        {
          "sentence": "I want my money back",
          "id": "feedback-123",
          "sentiment": {
            "label": "negative",
            "score": -0.75
          }
        }
      ],
      "size": 10,
      "sentiment": {
        "overall": "negative",
        "distribution": {
          "positive": 0,
          "neutral": 1,
          "negative": 9
        },
        "average_score": -0.72
      },
      "key_insights": [
        "89% express negative sentiment - requires attention",
        "Most common phrase: 'my money' (8 mentions)"
      ],
      "keywords": ["money", "withdraw", "back", "funds"],
      "source": "baseline"
    }
  ],
  "summary": {
    "total_sentences": 100,
    "clusters_found": 5,
    "unclustered": 3,
    "overall_sentiment": "negative",
    "query": "overview",
    "theme": "product feedback"
  },
  "comparison_insights": {
    "baseline_only_themes": ["Money Issues", "Login Problems"],
    "comparison_only_themes": ["New Feature Praise"],
    "shared_themes": ["Customer Support", "App Performance"]
  },
  "request_id": "abc-123-def"
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `clusters` | Array<Cluster> | Analyzed feedback clusters |
| `summary` | Summary | Overall analysis statistics |
| `comparison_insights` | ComparisonInsights? | Comparison analysis (if comparison data provided) |
| `request_id` | String | AWS request ID for tracing |

**Cluster Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Unique cluster identifier |
| `title` | String | Human-readable cluster title |
| `sentences` | Array<Sentence> | Sentences in this cluster |
| `size` | Integer | Number of sentences in cluster |
| `sentiment` | ClusterSentiment | Aggregated sentiment analysis |
| `key_insights` | Array<String> | Actionable insights (0-3 items) |
| `keywords` | Array<String> | Top keywords (up to 10) |
| `source` | String | "baseline", "comparison", or "mixed" |

**Sentence Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `sentence` | String | Original text |
| `id` | String | Original identifier |
| `sentiment` | Sentiment | Sentence-level sentiment |

**Sentiment Schema:**

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `label` | String | Classification | "positive", "neutral", or "negative" |
| `score` | Float | Compound score | -1.0 to +1.0 |

**ClusterSentiment Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `overall` | String | Overall cluster sentiment |
| `distribution` | Object | Count of positive/neutral/negative |
| `average_score` | Float | Mean compound score (-1 to +1) |

**Summary Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `total_sentences` | Integer | Total sentences analyzed |
| `clusters_found` | Integer | Number of clusters identified |
| `unclustered` | Integer | Number of noise points |
| `overall_sentiment` | String | Overall sentiment across all data |
| `query` | String | Query/context from request |
| `theme` | String? | Theme from request |

**ComparisonInsights Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `baseline_only_themes` | Array<String> | Themes unique to baseline |
| `comparison_only_themes` | Array<String> | Themes unique to comparison |
| `shared_themes` | Array<String> | Themes in both datasets |

**Error Response (400 Bad Request):**

```json
{
  "error": "Request validation failed",
  "details": [
    {
      "field": "baseline.0.sentence",
      "message": "Sentence cannot be empty after stripping whitespace",
      "type": "value_error"
    }
  ],
  "request_id": "abc-123-def"
}
```

**Error Response (500 Internal Server Error):**

```json
{
  "error": "Internal server error: [error message]",
  "request_id": "abc-123-def"
}
```

## Examples

### Example 1: Basic Baseline Analysis

**Request:**

```bash
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "I want my money back", "id": "1"},
      {"sentence": "Cannot withdraw funds", "id": "2"},
      {"sentence": "Best investment app", "id": "3"},
      {"sentence": "Love the interface", "id": "4"}
    ],
    "query": "product feedback"
  }'
```

**Response:**

```json
{
  "clusters": [
    {
      "id": "baseline-cluster-0",
      "title": "Money & Withdrawal",
      "sentences": [
        {
          "sentence": "I want my money back",
          "id": "1",
          "sentiment": {"label": "negative", "score": -0.75}
        },
        {
          "sentence": "Cannot withdraw funds",
          "id": "2",
          "sentiment": {"label": "negative", "score": -0.68}
        }
      ],
      "size": 2,
      "sentiment": {
        "overall": "negative",
        "distribution": {"positive": 0, "neutral": 0, "negative": 2},
        "average_score": -0.72
      },
      "key_insights": [
        "100% express negative sentiment - requires attention"
      ],
      "keywords": ["money", "withdraw", "funds", "back"],
      "source": "baseline"
    },
    {
      "id": "baseline-cluster-1",
      "title": "Investment App & Interface",
      "sentences": [
        {
          "sentence": "Best investment app",
          "id": "3",
          "sentiment": {"label": "positive", "score": 0.68}
        },
        {
          "sentence": "Love the interface",
          "id": "4",
          "sentiment": {"label": "positive", "score": 0.62}
        }
      ],
      "size": 2,
      "sentiment": {
        "overall": "positive",
        "distribution": {"positive": 2, "neutral": 0, "negative": 0},
        "average_score": 0.65
      },
      "key_insights": [
        "Overwhelmingly positive (100%) - key strength area"
      ],
      "keywords": ["investment", "app", "interface", "love", "best"],
      "source": "baseline"
    }
  ],
  "summary": {
    "total_sentences": 4,
    "clusters_found": 2,
    "unclustered": 0,
    "overall_sentiment": "neutral",
    "query": "product feedback",
    "theme": null
  },
  "comparison_insights": null
}
```

### Example 2: Baseline vs Comparison Analysis

**Request:**

```bash
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Slow customer support", "id": "b1"},
      {"sentence": "Long wait times", "id": "b2"}
    ],
    "comparison": [
      {"sentence": "Fast customer support", "id": "c1"},
      {"sentence": "Quick responses", "id": "c2"}
    ],
    "query": "support comparison",
    "surveyTitle": "Q1 vs Q2 Support Feedback"
  }'
```

**Response:**

```json
{
  "clusters": [...],
  "summary": {
    "total_sentences": 4,
    "clusters_found": 2,
    "unclustered": 0,
    "overall_sentiment": "neutral",
    "query": "support comparison",
    "theme": null
  },
  "comparison_insights": {
    "baseline_only_themes": ["Slow Response Times"],
    "comparison_only_themes": ["Fast Response Times"],
    "shared_themes": ["Customer Support"]
  }
}
```

### Example 3: Validation Error

**Request:**

```bash
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Test 1", "id": "1"},
      {"sentence": "Test 2", "id": "1"}
    ]
  }'
```

**Response (400):**

```json
{
  "error": "Request validation failed",
  "details": [
    {
      "field": "baseline",
      "message": "Duplicate IDs found: 1",
      "type": "value_error"
    }
  ],
  "request_id": "abc-123"
}
```

## Performance Characteristics

| Scenario | Expected Latency | Notes |
|----------|------------------|-------|
| Cold start (first request) | 2-5 seconds | Model loading overhead |
| Warm start (10 sentences) | < 1 second | Cached models |
| Warm start (100 sentences) | 2-4 seconds | Optimal batch size |
| Warm start (500 sentences) | 6-10 seconds | Near maximum |
| Warm start (1000 sentences) | 10-15 seconds | Maximum allowed |

**Optimization tips:**

- Keep requests under 500 sentences for sub-10s latency
- Use provisioned concurrency for consistent latency
- Enable SnapStart for faster cold starts (Python 3.12+)

## Rate Limits

Default API Gateway throttling:

- **Rate limit:** 100 requests/second
- **Burst limit:** 200 requests

Contact AWS to increase limits if needed.

## Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 200 | Success | - |
| 400 | Bad Request | Validation error, malformed JSON, duplicate IDs |
| 500 | Internal Server Error | Lambda timeout, out of memory, model loading error |
| 502 | Bad Gateway | Lambda function error |
| 503 | Service Unavailable | API Gateway overload |
| 504 | Gateway Timeout | Lambda exceeded 29s API Gateway limit |

## CORS Support

The API includes full CORS support with the following headers:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Methods: POST, OPTIONS
```

Preflight OPTIONS requests are automatically handled by API Gateway.

## Best Practices

### Request Optimization

1. **Batch similar feedback:** Group related feedback in single requests
2. **Use meaningful IDs:** Helps with tracing and debugging
3. **Keep sentences concise:** 1-2 sentences per item (not paragraphs)
4. **Deduplicate input:** Remove exact duplicates before sending

### Error Handling

```python
import requests

response = requests.post(
    "https://your-api.com/analyze",
    json={"baseline": [...]},
    timeout=30
)

if response.status_code == 200:
    data = response.json()
    # Process successful response
elif response.status_code == 400:
    # Handle validation error
    error = response.json()
    print(f"Validation failed: {error['error']}")
elif response.status_code >= 500:
    # Handle server error, consider retry
    print("Server error, retrying...")
```

### Interpreting Results

1. **Cluster Size:** Larger clusters = more prevalent themes
2. **Sentiment Scores:**
   - `> 0.5`: Strongly positive
   - `0.05 to 0.5`: Mildly positive
   - `-0.05 to 0.05`: Neutral
   - `-0.5 to -0.05`: Mildly negative
   - `< -0.5`: Strongly negative

3. **Key Insights:** Focus on clusters with actionable insights
4. **Keywords:** Use for quick theme identification

## Monitoring

### CloudWatch Metrics

Key metrics to monitor:

- **Invocations:** Total request count
- **Duration:** Average processing time
- **Errors:** Failed requests (4xx/5xx)
- **Throttles:** Rate-limited requests
- **ConcurrentExecutions:** Simultaneous Lambda instances

### CloudWatch Logs

Lambda logs include:

```
[INFO] Processing request abc-123
[INFO] Request validated: 100 baseline, 50 comparison sentences
[INFO] Encoding embeddings for 150 sentences...
[INFO] Clustering completed: 5 clusters found
[INFO] Request abc-123 completed successfully in 3.45s
```

### X-Ray Tracing (Optional)

Enable AWS X-Ray for detailed performance tracing:

1. Add X-Ray SDK to dependencies
2. Enable in Lambda configuration
3. View traces in X-Ray console

## Changelog

### v1.0.0 (2025-01-05)

- Initial release
- Semantic clustering with HDBSCAN
- Sentiment analysis with VADER
- TF-IDF keyword extraction
- Template-based insights
- Comparison mode support
- Python 3.12 runtime
- ONNX-optimized inference

## Support

For issues or questions:

1. Check CloudWatch Logs for error details
2. Review validation error messages
3. Verify request format matches schema
4. Check AWS service quotas and limits

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Development Plan](docs/DEVELOPMENT_PLAN.md)
