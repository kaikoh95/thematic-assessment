# Deployment Guide

This guide covers deploying the text analysis microservice to AWS using CDK.

## Prerequisites

1. **AWS Account** with appropriate credentials configured
2. **AWS CLI** installed and configured (`aws configure`)
3. **Node.js** 18+ (for CDK CLI)
4. **Python 3.12** installed
5. **Docker** (for building Lambda layers)

## Step 1: Install Dependencies

### Application Dependencies

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install application dependencies
pip install -r requirements.txt
```

### Infrastructure Dependencies

```bash
# Install CDK dependencies
cd infrastructure
pip install -r requirements.txt
cd ..

# Install CDK CLI globally
npm install -g aws-cdk
```

## Step 2: Build Lambda Layer

The Lambda layer contains all ML/NLP dependencies (sentence-transformers, HDBSCAN, UMAP, VADER).

```bash
# Navigate to layer directory
cd layers/ml-dependencies

# Create python directory
mkdir -p python

# Install dependencies into python/ directory
pip install -r requirements.txt -t python/ --platform manylinux2014_x86_64 --only-binary=:all:

# Return to project root
cd ../..
```

**Important:** The `--platform manylinux2014_x86_64 --only-binary=:all:` flags ensure dependencies are compatible with Lambda's Linux environment.

### Layer Size Check

Verify the layer size is under AWS limits (250MB uncompressed):

```bash
du -sh layers/ml-dependencies/python/
# Should show ~200-240MB
```

## Step 3: Bootstrap CDK (First Time Only)

If this is your first time using CDK in this AWS account/region:

```bash
cdk bootstrap aws://ACCOUNT-NUMBER/REGION

# Example:
# cdk bootstrap aws://123456789012/us-east-1
```

This creates the necessary CDK infrastructure in your account.

## Step 4: Deploy to AWS

### Synthesize CloudFormation Template (Optional)

Preview the CloudFormation template:

```bash
cdk synth
```

### Show Deployment Diff (Optional)

See what will change:

```bash
cdk diff
```

### Deploy Stack

Deploy the complete stack:

```bash
cdk deploy
```

You'll be prompted to approve IAM changes. Review and approve by typing `y`.

**Deployment takes ~5-10 minutes.**

### Deployment Output

After successful deployment, you'll see outputs like:

```
✅  TextAnalysisStack

Outputs:
TextAnalysisStack.APIEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod/
TextAnalysisStack.AnalyzeEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod/analyze
TextAnalysisStack.LambdaFunctionName = text-analysis-prod-TextAnalysisFunction123ABC
TextAnalysisStack.LambdaFunctionArn = arn:aws:lambda:us-east-1:123456789012:function:text-analysis-prod-TextAnalysisFunction123ABC

Stack ARN:
arn:aws:cloudformation:us-east-1:123456789012:stack/TextAnalysisStack/...
```

## Step 5: Test the Deployment

### Using cURL

```bash
curl -X POST https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "This app is great!", "id": "1"},
      {"sentence": "Love the features", "id": "2"},
      {"sentence": "Terrible experience", "id": "3"}
    ],
    "query": "overview"
  }'
```

### Using Python

```python
import requests
import json

url = "https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/analyze"

payload = {
    "baseline": [
        {"sentence": "This app is great!", "id": "1"},
        {"sentence": "Love the features", "id": "2"},
        {"sentence": "Terrible experience", "id": "3"}
    ],
    "query": "overview"
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

## Step 6: Monitor and Debug

### CloudWatch Logs

View Lambda logs:

```bash
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunction --follow
```

### Metrics

Check Lambda metrics in CloudWatch:
- Invocations
- Duration
- Errors
- Throttles

### API Gateway Dashboard

Monitor API Gateway metrics:
- Request count
- Latency (should be <10s)
- 4xx/5xx errors

## Performance Optimization

### Enable SnapStart (Optional)

For faster cold starts (~500ms instead of 2-4s):

1. Uncomment the SnapStart configuration in `infrastructure/stacks/lambda_stack.py`
2. Redeploy: `cdk deploy`

**Note:** SnapStart requires Lambda published versions and adds complexity. Only enable if cold start performance is critical.

### Provisioned Concurrency (Optional)

For production with consistent traffic, consider provisioned concurrency:

```python
# Add to lambda_stack.py
text_analysis_lambda.add_alias(
    "live",
    provisioned_concurrent_executions=2  # Always-warm instances
)
```

## Troubleshooting

### Layer Size Exceeds 250MB

If the layer is too large:

1. Use `--only-binary=:all:` when installing dependencies
2. Remove unnecessary packages
3. Use PyTorch CPU-only version (already configured)

### Cold Start Timeout

If Lambda times out on cold start:

1. Increase timeout in `lambda_stack.py` (currently 120s)
2. Reduce memory (lower memory = slower but cheaper)
3. Enable SnapStart

### CORS Errors

If browser requests fail with CORS:

1. Check API Gateway CORS configuration in `lambda_stack.py`
2. Verify response headers in `formatters.py`

### Import Errors in Lambda

If Lambda can't find modules:

1. Check `PYTHONPATH` environment variable includes `/opt/python`
2. Verify layer is attached to function
3. Check layer compatibility (Python 3.12)

## Cost Estimation

Typical costs for moderate usage:

- **Lambda:** ~$0.20 per 1M requests (3GB memory, 5s avg duration)
- **API Gateway:** ~$3.50 per 1M requests
- **CloudWatch Logs:** ~$0.50/GB

**Monthly estimate for 100K requests:** ~$0.40

## Cleanup

To delete all AWS resources:

```bash
cdk destroy
```

Confirm deletion when prompted. This will remove:
- Lambda function
- API Gateway
- CloudWatch logs
- IAM roles

**Note:** The CDK bootstrap stack (CDKToolkit) will remain. Delete manually if no longer needed.

## Next Steps

1. **Add Custom Domain:** Use Route53 + API Gateway custom domain
2. **Add Authentication:** API Gateway authorizers (IAM, Cognito, Lambda)
3. **Add Monitoring:** CloudWatch dashboards, X-Ray tracing
4. **Add CI/CD:** GitHub Actions or CodePipeline for automated deployments
5. **Add Caching:** API Gateway caching for frequently requested analyses

## References

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)
