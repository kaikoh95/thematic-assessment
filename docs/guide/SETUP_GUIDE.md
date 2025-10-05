# Setup Guide - Before Deployment

## Current Status

✅ **Completed:**
- AWS CDK CLI installed
- CDK Python dependencies installed in virtual environment
- All source code and infrastructure code ready

❌ **Remaining Prerequisites:**
- AWS CLI installation
- AWS credentials configuration
- Lambda layer build (ML dependencies)
- CDK bootstrap (first-time setup)

---

## Step-by-Step Setup

### 1. Install AWS CLI

**macOS (using Homebrew):**
```bash
brew install awscli
```

**Alternative (official installer):**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

**Verify installation:**
```bash
aws --version
# Should show: aws-cli/2.x.x
```

---

### 2. Configure AWS Credentials

You need an AWS account with appropriate permissions (Lambda, API Gateway, IAM, CloudWatch).

**Option A: Configure with access keys**
```bash
aws configure
```

You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (json)

**Option B: Use AWS SSO**
```bash
aws configure sso
```

**Verify credentials:**
```bash
aws sts get-caller-identity
# Should show your AWS account ID and user ARN
```

---

### 3. Build Lambda Layer

The Lambda layer contains all ML/NLP dependencies (~200-240MB).

**IMPORTANT:** This must be done before deployment!

```bash
# Navigate to layer directory
cd layers/ml-dependencies

# Create python directory
mkdir -p python

# Install dependencies (must use Linux-compatible packages)
pip install -r requirements.txt -t python/ \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  --python-version 3.12

# Return to project root
cd ../..

# Verify layer size (should be ~200-240MB)
du -sh layers/ml-dependencies/python/
```

**Why this is needed:**
- Lambda runs on Linux (Amazon Linux 2)
- We need Linux-compatible binary packages
- The `--platform manylinux2014_x86_64` flag ensures compatibility

---

### 4. Bootstrap CDK (First-Time Only)

If this is your first time using CDK in your AWS account/region:

```bash
# Activate virtual environment
source .venv/bin/activate

# Bootstrap CDK
cdk bootstrap aws://ACCOUNT-NUMBER/REGION

# Example:
# cdk bootstrap aws://123456789012/us-east-1
```

**What this does:**
- Creates an S3 bucket for CDK assets
- Creates IAM roles for deployments
- Sets up necessary infrastructure

**Check your account number:**
```bash
aws sts get-caller-identity --query Account --output text
```

---

### 5. Deploy to AWS

Once all prerequisites are met:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Preview changes (optional)
cdk diff

# Deploy!
cdk deploy

# You'll be asked to approve IAM changes - review and type 'y'
```

**Expected deployment time:** 5-10 minutes

**Expected output:**
```
✅  TextAnalysisStack

Outputs:
TextAnalysisStack.APIEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod/
TextAnalysisStack.AnalyzeEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod/analyze
TextAnalysisStack.LambdaFunctionName = text-analysis-prod-TextAnalysisFunction123ABC
```

---

## Quick Commands Summary

```bash
# 1. Install AWS CLI
brew install awscli

# 2. Configure AWS credentials
aws configure

# 3. Build Lambda layer
cd layers/ml-dependencies
mkdir -p python
pip install -r requirements.txt -t python/ \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  --python-version 3.12
cd ../..

# 4. Bootstrap CDK (first time only)
source .venv/bin/activate
cdk bootstrap

# 5. Deploy
cdk deploy
```

---

## Testing After Deployment

Once deployed, test with:

```bash
# Get your API endpoint from CDK outputs
ENDPOINT="https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/analyze"

# Test request
curl -X POST $ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": [
      {"sentence": "This is amazing!", "id": "1"},
      {"sentence": "Terrible experience", "id": "2"}
    ],
    "query": "test"
  }'
```

---

## Troubleshooting

### "Unable to locate credentials"
- Run `aws configure` to set up credentials
- Or ensure your AWS SSO session is active

### "Account has not been bootstrapped"
- Run `cdk bootstrap` in your target region

### "Layer size exceeds limit"
- Make sure you used `--platform manylinux2014_x86_64`
- Check layer size: `du -sh layers/ml-dependencies/python/`
- Should be ~200-240MB (under 250MB limit)

### "Module not found" in Lambda
- Ensure layer was built correctly
- Check `PYTHONPATH` in Lambda environment variables
- Verify layer is attached to function

---

## Next Steps After Deployment

1. **Test the API** - Send sample requests
2. **Monitor CloudWatch Logs** - Check for errors
3. **Review CloudWatch Metrics** - Monitor performance
4. **Set up monitoring** - Add CloudWatch alarms
5. **Add authentication** - API keys or IAM auth
6. **Custom domain** - Route53 + API Gateway

---

## Alternative: Local Testing

If you want to test locally before deploying:

```bash
# Install dependencies locally
pip install -r requirements.txt

# Run Lambda handler locally
python src/lambda_function.py
```

This will run the test event included in `lambda_function.py`.

---

## Cost Reminder

**Expected costs:**
- Lambda: ~$0.20 per 1M requests (3GB memory, 5s avg)
- API Gateway: ~$3.50 per 1M requests
- CloudWatch Logs: ~$0.50/GB

**Monthly estimate for 100K requests:** ~$0.60

**Free tier eligible** for first 12 months (1M Lambda requests/month free)

---

## Need Help?

- AWS CLI setup: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- AWS credentials: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
- CDK documentation: https://docs.aws.amazon.com/cdk/
- Project documentation: See [DEPLOYMENT.md](DEPLOYMENT.md)
