# CURL Testing Guide

This guide provides sample CURL commands to test the deployed Lambda endpoint.

## Endpoint Information

**Production Endpoint:** `https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze`

**Timeout:** 15 minutes (900 seconds) for Lambda processing

**API Gateway Timeout:** 29 seconds (hard limit) - requests taking longer will timeout at API Gateway but Lambda continues processing

## Quick Reference

### 1. Basic Test (Minimal Payload)

```bash
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Great product", "id": "1"},
      {"sentence": "Poor service", "id": "2"},
      {"sentence": "Amazing experience", "id": "3"}
    ],
    "query": "overview"
  }'
```

### 2. Test with File (Recommended)

```bash
# Using test payload file
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json
```

### 3. Test with Response Formatting

```bash
# Pretty-print JSON response
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -s | python3 -m json.tool
```

### 4. Test with Timing Information

```bash
# Include HTTP status code and timing
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -w "\n\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
  -s | python3 -m json.tool
```

### 5. Test with Output File

```bash
# Save response to file
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -o response.json \
  -w "HTTP Status: %{http_code}\nTime: %{time_total}s\n"
```

## Complete Test Examples

### Example 1: Baseline-Only Analysis

```bash
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "I want my money back", "id": "1"},
      {"sentence": "Cannot withdraw funds", "id": "2"},
      {"sentence": "Best investment app ever", "id": "3"},
      {"sentence": "Great platform", "id": "4"},
      {"sentence": "Terrible customer service", "id": "5"}
    ],
    "query": "product feedback",
    "surveyTitle": "Q1 Product Feedback"
  }' \
  -s | python3 -m json.tool
```

### Example 2: Baseline + Comparison Analysis

```bash
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Love the new features", "id": "1"},
      {"sentence": "App crashes frequently", "id": "2"}
    ],
    "comparison": [
      {"sentence": "Much better than before", "id": "3"},
      {"sentence": "Still has bugs", "id": "4"}
    ],
    "query": "version comparison",
    "theme": "app updates"
  }' \
  -s | python3 -m json.tool
```

### Example 3: Large Dataset Test

```bash
# Test with larger dataset from file
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -w "\n\nHTTP Status: %{http_code}\nConnect Time: %{time_connect}s\nTotal Time: %{time_total}s\n" \
  -s -o large_response.json

# Then view the response
cat large_response.json | python3 -m json.tool | head -100
```

## Validation Tests

### Test 1: Duplicate ID Detection

```bash
# Should return 400 error with duplicate ID message
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "First sentence", "id": "1"},
      {"sentence": "Second sentence", "id": "1"}
    ],
    "query": "test"
  }' \
  -s | python3 -m json.tool
```

### Test 2: Missing Required Field

```bash
# Should return 400 error - missing 'query'
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Test", "id": "1"}
    ]
  }' \
  -s | python3 -m json.tool
```

### Test 3: Invalid JSON

```bash
# Should return 500 error - malformed JSON
curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{invalid json}' \
  -s | python3 -m json.tool
```

## Performance Testing

### Warm vs Cold Start Comparison

```bash
# First request (cold start - expect timeout for large datasets)
echo "=== COLD START TEST ==="
time curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -s -o /dev/null \
  -w "HTTP: %{http_code}, Time: %{time_total}s\n"

# Wait 5 seconds
sleep 5

# Second request (warm start - should be fast)
echo "=== WARM START TEST ==="
time curl -X POST https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d @data/input_example.json \
  -s -o /dev/null \
  -w "HTTP: %{http_code}, Time: %{time_total}s\n"
```

## Expected Response Structure

Successful response (HTTP 200):

```json
{
  "clusters": [
    {
      "id": "baseline-cluster-0",
      "title": "Money & Withdrawal Issues",
      "sentences": [...],
      "size": 3,
      "sentiment": {
        "overall": "negative",
        "distribution": {"positive": 0, "neutral": 1, "negative": 2},
        "average_score": -0.45
      },
      "key_insights": ["..."],
      "keywords": ["money", "withdraw", ...],
      "source": "baseline"
    }
  ],
  "summary": {
    "total_sentences": 15,
    "clusters_found": 6,
    "unclustered": 0,
    "overall_sentiment": "neutral",
    "query": "overview",
    "theme": null
  },
  "comparison_insights": null
}
```

Error response (HTTP 400/500):

```json
{
  "error": "Duplicate IDs found: 1, 2, 3",
  "request_id": "abc-123-def"
}
```

## Troubleshooting

### Timeout Issues

**Problem:** Request times out after 29 seconds
**Cause:** API Gateway has a hard 29-second timeout limit
**Solution:**
- Use smaller datasets for API Gateway
- Lambda continues processing (check CloudWatch logs)
- For large batches, consider direct Lambda invocation

### Connection Refused

**Problem:** `Connection refused` or network error
**Cause:** Incorrect endpoint URL
**Solution:** Verify endpoint from CDK outputs:

```bash
# Get current endpoint
aws cloudformation describe-stacks \
  --stack-name text-analysis-prod \
  --query 'Stacks[0].Outputs[?OutputKey==`AnalyzeEndpoint`].OutputValue' \
  --output text
```

### Invalid Response

**Problem:** Unexpected response format
**Cause:** API Gateway or Lambda error
**Solution:** Check CloudWatch logs (see LOG_CHECKS.md)

## Quick Test Script

Save this as `test_api.sh`:

```bash
#!/bin/bash
ENDPOINT="https://qs4om06hn8.execute-api.ap-southeast-2.amazonaws.com/prod/analyze"

echo "Testing Text Analysis API..."
echo "Endpoint: $ENDPOINT"
echo ""

# Test 1: Basic functionality
echo "Test 1: Basic Request"
curl -X POST $ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "Great product", "id": "1"},
      {"sentence": "Poor service", "id": "2"}
    ],
    "query": "test"
  }' \
  -s -w "\nStatus: %{http_code}\n" | python3 -m json.tool | head -50

echo -e "\n---\n"

# Test 2: Validation
echo "Test 2: Duplicate ID Validation"
curl -X POST $ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "First", "id": "1"},
      {"sentence": "Second", "id": "1"}
    ],
    "query": "test"
  }' \
  -s -w "\nStatus: %{http_code}\n" | python3 -m json.tool

echo -e "\nTests complete!"
```

Run with: `chmod +x test_api.sh && ./test_api.sh`
