# GitHub Actions Setup Guide

This guide explains how to configure GitHub Actions for automatic deployment of the text analysis Lambda function.

## Overview

The workflow (`.github/workflows/deploy.yml`) automatically deploys the CDK stack to AWS whenever code is pushed to the `main` branch.

## Workflow Steps

1. **Run Tests** - Validates code with pytest
2. **Deploy CDK Stack** - Builds Docker image and deploys to AWS Lambda
3. **Test API** - Verifies the deployed endpoint is working

## Required GitHub Secrets

You need to configure AWS credentials as GitHub repository secrets. There are two authentication methods:

### Option 1: OIDC (Recommended) ⭐

OIDC provides more secure authentication without long-lived credentials.

**Required Secret:**
- `AWS_ROLE_ARN` - ARN of the IAM role that GitHub Actions will assume

**Setup Steps:**

1. **Create an OIDC provider in AWS:**
   ```bash
   aws iam create-open-id-connect-provider \
     --url https://token.actions.githubusercontent.com \
     --client-id-list sts.amazonaws.com \
     --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
   ```

2. **Create an IAM role with trust policy:**

   Create `github-actions-trust-policy.json`:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
         },
         "Action": "sts:AssumeRoleWithWebIdentity",
         "Condition": {
           "StringEquals": {
             "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
           },
           "StringLike": {
             "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_ORG/YOUR_REPO:ref:refs/heads/main"
           }
         }
       }
     ]
   }
   ```

   Create the role:
   ```bash
   aws iam create-role \
     --role-name GitHubActionsDeployRole \
     --assume-role-policy-document file://github-actions-trust-policy.json
   ```

3. **Attach deployment permissions:**
   ```bash
   aws iam attach-role-policy \
     --role-name GitHubActionsDeployRole \
     --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
   ```

   For production, create a more restrictive policy with only required permissions:
   - CloudFormation full access
   - Lambda full access
   - IAM role creation/management
   - API Gateway full access
   - ECR full access (for Docker images)
   - S3 access (for CDK bootstrap bucket)
   - CloudWatch Logs access

4. **Add secret to GitHub:**
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `AWS_ROLE_ARN`
   - Value: `arn:aws:iam::YOUR_ACCOUNT_ID:role/GitHubActionsDeployRole`

### Option 2: Access Keys (Simpler but less secure)

**Required Secrets:**
- `AWS_ACCESS_KEY_ID` - Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret access key

**Setup Steps:**

1. **Create an IAM user for GitHub Actions:**
   ```bash
   aws iam create-user --user-name github-actions-deploy
   ```

2. **Attach deployment permissions:**
   ```bash
   aws iam attach-user-policy \
     --user-name github-actions-deploy \
     --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
   ```

3. **Create access keys:**
   ```bash
   aws iam create-access-key --user-name github-actions-deploy
   ```

   Save the `AccessKeyId` and `SecretAccessKey` from the output.

4. **Add secrets to GitHub:**
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add two secrets:
     - Name: `AWS_ACCESS_KEY_ID`, Value: Your access key ID
     - Name: `AWS_SECRET_ACCESS_KEY`, Value: Your secret access key

5. **Update workflow file:**

   Edit `.github/workflows/deploy.yml` and comment out the OIDC section, uncomment the access keys section:
   ```yaml
   - name: Configure AWS credentials
     uses: aws-actions/configure-aws-credentials@v4
     with:
       # Option 2: Use access keys
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: ${{ env.AWS_REGION }}
   ```

## Testing the Workflow

1. **Push to main branch:**
   ```bash
   git add .
   git commit -m "Add GitHub Actions workflow"
   git push origin main
   ```

2. **Monitor the deployment:**
   - Go to your repository → Actions tab
   - Click on the running workflow
   - Watch the logs for each step

3. **Check deployment output:**
   - The workflow will print the API endpoint and Lambda function name
   - It will also test the API endpoint with a sample request

## Workflow Configuration

The workflow is configured with these settings:

- **Python Version:** 3.12
- **Node Version:** 20 (for CDK)
- **AWS Region:** ap-southeast-2
- **Stack Name:** text-analysis-prod

To change these, edit the `env` section in `.github/workflows/deploy.yml`.

## Troubleshooting

### Deployment fails with permission errors

- Ensure the IAM role/user has sufficient permissions
- Check CloudFormation and Lambda permissions specifically

### Docker build fails

- Ensure Docker Buildx is available (handled by the workflow)
- Check Dockerfile syntax and dependencies

### Tests fail

- Run tests locally first: `pytest tests/ -v`
- Check for missing dependencies in requirements.txt

### CDK bootstrap not completed

If you see bootstrap errors, manually run:
```bash
aws configure  # Set your credentials
cdk bootstrap aws://YOUR_ACCOUNT_ID/ap-southeast-2
```

## Security Best Practices

1. ✅ Use OIDC authentication (Option 1) for better security
2. ✅ Limit IAM permissions to only what's needed for deployment
3. ✅ Enable branch protection on `main` to require PR reviews
4. ✅ Use environment-specific deployments (dev/staging/prod)
5. ✅ Enable CloudTrail logging for audit trails
6. ✅ Rotate access keys regularly (if using Option 2)
7. ✅ Use separate AWS accounts for production

## Advanced: Multi-Environment Deployment

To deploy to multiple environments (dev/staging/prod), modify the workflow:

```yaml
on:
  push:
    branches:
      - main        # deploys to prod
      - develop     # deploys to dev
      - staging     # deploys to staging

jobs:
  deploy:
    # ... existing steps ...

    - name: Determine environment
      id: env
      run: |
        if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
          echo "environment=prod" >> $GITHUB_OUTPUT
        elif [[ "${{ github.ref }}" == "refs/heads/staging" ]]; then
          echo "environment=staging" >> $GITHUB_OUTPUT
        else
          echo "environment=dev" >> $GITHUB_OUTPUT
        fi

    - name: CDK Deploy
      run: |
        cdk deploy --require-approval never \
          --context environment=${{ steps.env.outputs.environment }}
```

Then update your CDK stack to use the environment context.

## Support

For issues with the workflow:
1. Check the Actions tab for detailed error logs
2. Review AWS CloudFormation events in the AWS Console
3. Check Lambda CloudWatch logs for runtime errors
