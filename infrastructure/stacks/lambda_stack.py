"""
CDK Stack for Text Analysis Microservice

Provisions:
- Lambda function with ML dependencies layer
- API Gateway REST API
- IAM roles and permissions
- CloudWatch logging
- CORS configuration

Architecture:
API Gateway → Lambda (with Layer) → CloudWatch Logs

Key configurations:
- Python 3.12 runtime (Container Image deployment)
- 3GB memory (optimal for ML workloads)
- 900s timeout (15 minutes - handles large batches)
- Ephemeral storage: 2GB (model caching)
"""

from aws_cdk import (
    Stack,
    Duration,
    Size,
    CfnOutput,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_logs as logs,
    aws_iam as iam,
)
from constructs import Construct
import os


class TextAnalysisStack(Stack):
    """
    CDK Stack for text analysis Lambda + API Gateway
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ====================================================================
        # LAMBDA FUNCTION (Container Image)
        # ====================================================================
        # Using container images instead of layers to support large ML dependencies (>250MB)
        # Container images support up to 10GB vs 250MB for layers

        # Lambda execution role
        lambda_role = iam.Role(
            self,
            "TextAnalysisLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Main Lambda function with Docker container
        # Docker layer caching is handled via Docker Buildx in GitHub Actions
        # The ghaction-github-runtime and buildx setup in CI enables automatic caching
        text_analysis_lambda = lambda_.DockerImageFunction(
            self,
            "TextAnalysisFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=".",
                file="Dockerfile",
                # Use CMD from Dockerfile - don't override here
            ),
            role=lambda_role,
            timeout=Duration.seconds(900),  # 15 minutes (for large batches)
            memory_size=3008,  # 3GB (optimal for ML)
            ephemeral_storage_size=Size.mebibytes(2048),  # 2GB (for model caching)
            environment={
                "LOG_LEVEL": "INFO",
                "NUMBA_CACHE_DIR": "/tmp",  # Use /tmp for numba caching (only writable dir in Lambda)
                "TRANSFORMERS_CACHE": "/tmp/transformers",  # HuggingFace transformers cache
                "HF_HOME": "/tmp/hf",  # HuggingFace home directory
                "SENTENCE_TRANSFORMERS_HOME": "/tmp/sentence_transformers",  # Sentence transformers cache
            },
            description="Text analysis microservice with semantic clustering and sentiment analysis (Container Image)",
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # SnapStart CANNOT be enabled - incompatible with Container Images
        # AWS Lambda SnapStart limitations:
        # - Only supports ZIP package deployments (not DockerImageFunction)
        # - Does not support ephemeral storage >512MB
        #
        # Our architecture uses Container Images because:
        # - ML dependencies exceed 250MB ZIP limit
        # - Container images support up to 10GB
        #
        # Cold start mitigation strategies instead:
        # 1. Global model caching (models persist across warm invocations)
        # 2. Lazy loading (defer heavy imports to invocation phase)
        # 3. Provisioned concurrency (if needed for production)

        # ====================================================================
        # API GATEWAY
        # ====================================================================

        # REST API
        api = apigw.RestApi(
            self,
            "TextAnalysisAPI",
            rest_api_name="Text Analysis Service",
            description="Semantic clustering and sentiment analysis API",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization"],
            ),
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,  # requests per second
                throttling_burst_limit=200,  # burst capacity
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                metrics_enabled=True,
            ),
        )

        # Lambda integration
        lambda_integration = apigw.LambdaIntegration(
            text_analysis_lambda,
            proxy=True,  # Proxy integration (event contains full request)
            integration_responses=[
                apigw.IntegrationResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": "'*'"
                    },
                )
            ],
        )

        # POST /analyze endpoint
        analyze_resource = api.root.add_resource("analyze")
        analyze_resource.add_method(
            "POST",
            lambda_integration,
            method_responses=[
                apigw.MethodResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": True
                    },
                )
            ],
        )

        # ====================================================================
        # OUTPUTS
        # ====================================================================

        CfnOutput(
            self,
            "APIEndpoint",
            value=api.url,
            description="API Gateway endpoint URL",
            export_name=f"{self.stack_name}-APIEndpoint",
        )

        CfnOutput(
            self,
            "AnalyzeEndpoint",
            value=f"{api.url}analyze",
            description="Full endpoint for text analysis",
            export_name=f"{self.stack_name}-AnalyzeEndpoint",
        )

        CfnOutput(
            self,
            "LambdaFunctionName",
            value=text_analysis_lambda.function_name,
            description="Lambda function name",
            export_name=f"{self.stack_name}-LambdaFunctionName",
        )

        CfnOutput(
            self,
            "LambdaFunctionArn",
            value=text_analysis_lambda.function_arn,
            description="Lambda function ARN",
            export_name=f"{self.stack_name}-LambdaFunctionArn",
        )
