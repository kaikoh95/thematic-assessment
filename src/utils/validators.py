"""
Pydantic Validators for Request/Response Models

Type-safe validation for API Gateway requests and Lambda responses.

Key features:
- Strong typing with Pydantic v2
- Custom validators for business logic
- Automatic JSON schema generation
- Clear error messages
- Support for baseline + comparison mode
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SentenceInput(BaseModel):
    """
    Individual sentence input model

    Example:
        {
            "sentence": "This app is great!",
            "id": "uuid-123"
        }
    """
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer feedback sentence"
    )
    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the sentence"
    )

    @field_validator('sentence')
    @classmethod
    def validate_sentence(cls, v: str) -> str:
        """Validate and clean sentence"""
        # Strip whitespace
        v = v.strip()

        if not v:
            raise ValueError("Sentence cannot be empty after stripping whitespace")

        return v


class AnalysisRequest(BaseModel):
    """
    Main API request model

    Example:
        {
            "baseline": [...],
            "comparison": [...],  # optional
            "query": "overview",
            "surveyTitle": "Product Feedback Q1",
            "theme": "user experience"
        }
    """
    baseline: List[SentenceInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Baseline sentences for analysis"
    )

    comparison: Optional[List[SentenceInput]] = Field(
        default=None,
        max_length=1000,
        description="Optional comparison sentences"
    )

    query: str = Field(
        default="overview",
        max_length=100,
        description="Query/context for the analysis"
    )

    surveyTitle: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Title of the survey/feedback collection"
    )

    theme: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Theme/category of the feedback"
    )

    @field_validator('baseline', 'comparison')
    @classmethod
    def validate_no_duplicate_ids(cls, v: Optional[List[SentenceInput]]) -> Optional[List[SentenceInput]]:
        """Ensure no duplicate IDs within a dataset"""
        if v is None:
            return v

        ids = [item.id for item in v]

        if len(ids) != len(set(ids)):
            # Find duplicates
            seen = set()
            duplicates = set()
            for id_ in ids:
                if id_ in seen:
                    duplicates.add(id_)
                seen.add(id_)

            raise ValueError(
                f"Duplicate IDs found: {', '.join(list(duplicates)[:5])}"
            )

        return v

    @model_validator(mode='after')
    def validate_total_size(self) -> 'AnalysisRequest':
        """Validate total sentence count doesn't exceed limits"""
        baseline_count = len(self.baseline)
        comparison_count = len(self.comparison) if self.comparison else 0
        total = baseline_count + comparison_count

        # Soft limit warning
        if total > 500:
            logger.warning(
                f"Large request: {total} sentences (may exceed 10s latency target)"
            )

        # Hard limit
        if total > 1000:
            raise ValueError(
                f"Total sentences ({total}) exceeds maximum (1000). "
                f"Baseline: {baseline_count}, Comparison: {comparison_count}"
            )

        return self


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class SentimentData(BaseModel):
    """Sentiment analysis results"""
    label: str = Field(
        ...,
        description="Sentiment classification: positive, neutral, or negative"
    )
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Compound sentiment score (-1 to +1)"
    )


class SentenceOutput(BaseModel):
    """Individual sentence in cluster response"""
    sentence: str = Field(..., description="Original sentence text")
    id: str = Field(..., description="Sentence ID from input")
    sentiment: SentimentData = Field(..., description="Sentence-level sentiment")


class ClusterSentiment(BaseModel):
    """Cluster-level sentiment aggregation"""
    overall: str = Field(
        ...,
        description="Overall cluster sentiment"
    )
    distribution: Dict[str, int] = Field(
        ...,
        description="Sentiment distribution counts"
    )
    average_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Average compound score"
    )


class ClusterOutput(BaseModel):
    """Individual cluster in the response"""
    id: str = Field(..., description="Cluster identifier")
    title: str = Field(..., description="Human-readable cluster title")
    sentences: List[SentenceOutput] = Field(
        ...,
        description="Sentences in this cluster"
    )
    size: int = Field(..., ge=0, description="Number of sentences in cluster")

    sentiment: ClusterSentiment = Field(
        ...,
        description="Cluster-level sentiment analysis"
    )

    key_insights: List[str] = Field(
        ...,
        description="Actionable insights for this cluster"
    )

    keywords: List[str] = Field(
        ...,
        description="Top keywords characterizing this cluster"
    )

    source: Optional[str] = Field(
        default=None,
        description="Data source: baseline, comparison, or mixed"
    )


class ComparisonInsights(BaseModel):
    """Insights from baseline vs comparison analysis"""
    baseline_only_themes: List[str] = Field(
        default_factory=list,
        description="Themes only in baseline data"
    )
    comparison_only_themes: List[str] = Field(
        default_factory=list,
        description="Themes only in comparison data"
    )
    shared_themes: List[str] = Field(
        default_factory=list,
        description="Themes present in both datasets"
    )


class AnalysisSummary(BaseModel):
    """Summary statistics for the analysis"""
    total_sentences: int = Field(..., ge=0)
    clusters_found: int = Field(..., ge=0)
    unclustered: int = Field(..., ge=0)
    overall_sentiment: str = Field(...)
    query: str = Field(...)
    theme: Optional[str] = None


class AnalysisResponse(BaseModel):
    """
    Complete API response model

    Example:
        {
            "clusters": [...],
            "summary": {...},
            "comparison_insights": {...}  # if comparison mode
        }
    """
    clusters: List[ClusterOutput] = Field(
        ...,
        description="Analyzed clusters with insights"
    )

    summary: AnalysisSummary = Field(
        ...,
        description="Overall analysis summary"
    )

    comparison_insights: Optional[ComparisonInsights] = Field(
        default=None,
        description="Comparative analysis (if comparison data provided)"
    )


# ============================================================================
# ERROR RESPONSE MODEL
# ============================================================================

class ErrorDetail(BaseModel):
    """Individual validation error"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ErrorResponse(BaseModel):
    """Error response for validation failures"""
    error: str = Field(..., description="High-level error description")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed validation errors"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="AWS request ID for tracing"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_request(data: Dict[str, Any]) -> AnalysisRequest:
    """
    Validate incoming request data

    Args:
        data: Raw request data from API Gateway

    Returns:
        Validated AnalysisRequest

    Raises:
        ValidationError: If validation fails
    """
    return AnalysisRequest.model_validate(data)


def format_validation_errors(validation_error: Exception) -> ErrorResponse:
    """
    Format Pydantic validation errors for API response

    Args:
        validation_error: Pydantic ValidationError

    Returns:
        Formatted ErrorResponse
    """
    errors = []

    for error in validation_error.errors():
        errors.append(
            ErrorDetail(
                field='.'.join(str(loc) for loc in error['loc']),
                message=error['msg'],
                type=error['type']
            )
        )

    return ErrorResponse(
        error="Request validation failed",
        details=errors
    )


if __name__ == "__main__":
    # Example usage and testing
    import json
    from pydantic import ValidationError

    print("=" * 60)
    print("Pydantic Validators Test")
    print("=" * 60)

    # Test 1: Valid request
    print("\nTest 1: Valid Request")
    print("-" * 60)

    valid_data = {
        "baseline": [
            {"sentence": "This is great!", "id": "1"},
            {"sentence": "I love it", "id": "2"},
            {"sentence": "Amazing experience", "id": "3"}
        ],
        "query": "overview",
        "surveyTitle": "Product Feedback",
        "theme": "user experience"
    }

    try:
        request = AnalysisRequest.model_validate(valid_data)
        print(f"✓ Valid request accepted")
        print(f"  Baseline sentences: {len(request.baseline)}")
        print(f"  Query: {request.query}")
        print(f"  Theme: {request.theme}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")

    # Test 2: Duplicate IDs
    print("\nTest 2: Duplicate IDs (Should Fail)")
    print("-" * 60)

    duplicate_data = {
        "baseline": [
            {"sentence": "First", "id": "1"},
            {"sentence": "Second", "id": "1"},  # Duplicate ID
        ]
    }

    try:
        request = AnalysisRequest.model_validate(duplicate_data)
        print(f"✗ Should have failed!")
    except ValidationError as e:
        print(f"✓ Correctly rejected duplicate IDs")
        error_response = format_validation_errors(e)
        print(f"  Error: {error_response.error}")
        print(f"  Details: {error_response.details[0].message}")

    # Test 3: Empty sentence
    print("\nTest 3: Empty Sentence (Should Fail)")
    print("-" * 60)

    empty_data = {
        "baseline": [
            {"sentence": "   ", "id": "1"},  # Empty after strip
        ]
    }

    try:
        request = AnalysisRequest.model_validate(empty_data)
        print(f"✗ Should have failed!")
    except ValidationError as e:
        print(f"✓ Correctly rejected empty sentence")
        print(f"  Error: {e.errors()[0]['msg']}")

    # Test 4: Too many sentences
    print("\nTest 4: Size Limit (1001 sentences)")
    print("-" * 60)

    large_data = {
        "baseline": [
            {"sentence": f"Sentence {i}", "id": f"id-{i}"}
            for i in range(1001)
        ]
    }

    try:
        request = AnalysisRequest.model_validate(large_data)
        print(f"✗ Should have failed!")
    except ValidationError as e:
        print(f"✓ Correctly rejected oversized request")
        print(f"  Error: {e.errors()[0]['msg']}")

    # Test 5: Valid response serialization
    print("\nTest 5: Response Serialization")
    print("-" * 60)

    response = AnalysisResponse(
        clusters=[
            ClusterOutput(
                id="cluster-1",
                title="Money Issues",
                sentences=[
                    SentenceOutput(
                        sentence="I want my money back",
                        id="1",
                        sentiment=SentimentData(label="negative", score=-0.75)
                    )
                ],
                size=1,
                sentiment=ClusterSentiment(
                    overall="negative",
                    distribution={"positive": 0, "neutral": 0, "negative": 1},
                    average_score=-0.75
                ),
                key_insights=["High negative sentiment"],
                keywords=["money", "back"],
                source="baseline"
            )
        ],
        summary=AnalysisSummary(
            total_sentences=1,
            clusters_found=1,
            unclustered=0,
            overall_sentiment="negative",
            query="overview"
        )
    )

    json_output = response.model_dump_json(indent=2)
    print("✓ Response serialized to JSON")
    print(f"  Length: {len(json_output)} characters")
    print(f"  Clusters: {len(response.clusters)}")
