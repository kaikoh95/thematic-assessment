#!/usr/bin/env python3
"""
AWS CDK App - Text Analysis Microservice

Entry point for CDK deployment.

Usage:
    cdk deploy           # Deploy to AWS
    cdk diff             # Show changes
    cdk synth            # Generate CloudFormation
    cdk destroy          # Delete stack
"""

import os
from aws_cdk import App, Environment
from stacks.lambda_stack import TextAnalysisStack

# Initialize CDK app
app = App()

# Get AWS account and region from environment or use defaults
env = Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "ap-southeast-2"),
)

# Create stack
TextAnalysisStack(
    app,
    "TextAnalysisStack",
    env=env,
    description="Text analysis microservice with semantic clustering and sentiment analysis",
    stack_name="text-analysis-prod",
)

# Synthesize CloudFormation template
app.synth()
