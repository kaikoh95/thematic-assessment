"""
Unit Tests for Pydantic Validators

Tests request validation, response serialization, and error handling.
"""

import pytest
from pydantic import ValidationError

from src.utils.validators import (
    SentenceInput,
    AnalysisRequest,
    AnalysisResponse,
    ClusterOutput,
    ClusterSentiment,
    SentenceOutput,
    SentimentData,
    AnalysisSummary,
    validate_request,
    format_validation_errors
)


class TestSentenceInput:
    """Test SentenceInput validation"""

    def test_valid_sentence(self):
        """Valid sentence should pass"""
        sentence = SentenceInput(sentence="This is a test", id="test-1")
        assert sentence.sentence == "This is a test"
        assert sentence.id == "test-1"

    def test_whitespace_stripping(self):
        """Whitespace should be stripped"""
        sentence = SentenceInput(sentence="  Test  ", id="test-1")
        assert sentence.sentence == "Test"

    def test_empty_after_strip_fails(self):
        """Empty sentence after strip should fail"""
        with pytest.raises(ValidationError) as exc_info:
            SentenceInput(sentence="   ", id="test-1")
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_sentence_too_long_fails(self):
        """Sentence exceeding max_length should fail"""
        long_sentence = "a" * 1001
        with pytest.raises(ValidationError):
            SentenceInput(sentence=long_sentence, id="test-1")

    def test_id_too_long_fails(self):
        """ID exceeding max_length should fail"""
        long_id = "a" * 101
        with pytest.raises(ValidationError):
            SentenceInput(sentence="Test", id=long_id)


class TestAnalysisRequest:
    """Test AnalysisRequest validation"""

    def test_valid_baseline_only(self):
        """Valid baseline-only request should pass"""
        request = AnalysisRequest(
            baseline=[
                SentenceInput(sentence="Test 1", id="1"),
                SentenceInput(sentence="Test 2", id="2")
            ],
            query="overview"
        )
        assert len(request.baseline) == 2
        assert request.comparison is None
        assert request.query == "overview"

    def test_valid_with_comparison(self):
        """Valid request with comparison should pass"""
        request = AnalysisRequest(
            baseline=[SentenceInput(sentence="Test 1", id="1")],
            comparison=[SentenceInput(sentence="Test 2", id="2")],
            query="comparison"
        )
        assert len(request.baseline) == 1
        assert len(request.comparison) == 1

    def test_duplicate_baseline_ids_fails(self):
        """Duplicate IDs in baseline should fail"""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                baseline=[
                    SentenceInput(sentence="Test 1", id="1"),
                    SentenceInput(sentence="Test 2", id="1")  # Duplicate ID
                ]
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_duplicate_comparison_ids_fails(self):
        """Duplicate IDs in comparison should fail"""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                baseline=[SentenceInput(sentence="Test 1", id="1")],
                comparison=[
                    SentenceInput(sentence="Test 2", id="2"),
                    SentenceInput(sentence="Test 3", id="2")  # Duplicate
                ]
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_total_size_exceeds_limit_fails(self):
        """Total sentences exceeding 1000 should fail"""
        baseline = [
            SentenceInput(sentence=f"Test {i}", id=f"id-{i}")
            for i in range(600)
        ]
        comparison = [
            SentenceInput(sentence=f"Test {i}", id=f"comp-{i}")
            for i in range(401)
        ]

        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(baseline=baseline, comparison=comparison)
        assert "1000" in str(exc_info.value)

    def test_empty_baseline_fails(self):
        """Empty baseline should fail"""
        with pytest.raises(ValidationError):
            AnalysisRequest(baseline=[])

    def test_default_query(self):
        """Default query should be 'overview'"""
        request = AnalysisRequest(
            baseline=[SentenceInput(sentence="Test", id="1")]
        )
        assert request.query == "overview"

    def test_optional_fields(self):
        """Optional fields should have None defaults"""
        request = AnalysisRequest(
            baseline=[SentenceInput(sentence="Test", id="1")]
        )
        assert request.surveyTitle is None
        assert request.theme is None
        assert request.comparison is None


class TestAnalysisResponse:
    """Test AnalysisResponse serialization"""

    def test_valid_response_serialization(self):
        """Valid response should serialize to JSON"""
        response = AnalysisResponse(
            clusters=[
                ClusterOutput(
                    id="cluster-1",
                    title="Test Cluster",
                    sentences=[
                        SentenceOutput(
                            sentence="Test sentence",
                            id="1",
                            sentiment=SentimentData(label="positive", score=0.5)
                        )
                    ],
                    size=1,
                    sentiment=ClusterSentiment(
                        overall="positive",
                        distribution={"positive": 1, "neutral": 0, "negative": 0},
                        average_score=0.5
                    ),
                    key_insights=["Test insight"],
                    keywords=["test", "keyword"]
                )
            ],
            summary=AnalysisSummary(
                total_sentences=1,
                clusters_found=1,
                unclustered=0,
                overall_sentiment="positive",
                query="overview"
            )
        )

        # Test serialization
        json_data = response.model_dump_json()
        assert "cluster-1" in json_data
        assert "positive" in json_data

        # Test deserialization
        response2 = AnalysisResponse.model_validate_json(json_data)
        assert response2.summary.total_sentences == 1

    def test_response_with_comparison_insights(self):
        """Response with comparison insights should serialize"""
        from src.utils.validators import ComparisonInsights

        response = AnalysisResponse(
            clusters=[],
            summary=AnalysisSummary(
                total_sentences=0,
                clusters_found=0,
                unclustered=0,
                overall_sentiment="neutral",
                query="test"
            ),
            comparison_insights=ComparisonInsights(
                baseline_only_themes=["Theme A"],
                comparison_only_themes=["Theme B"],
                shared_themes=["Theme C"]
            )
        )

        json_data = response.model_dump()
        assert json_data["comparison_insights"]["baseline_only_themes"] == ["Theme A"]


class TestValidateRequest:
    """Test validate_request helper function"""

    def test_valid_dict_input(self):
        """Valid dict should be validated"""
        data = {
            "baseline": [
                {"sentence": "Test", "id": "1"}
            ],
            "query": "overview"
        }

        request = validate_request(data)
        assert isinstance(request, AnalysisRequest)
        assert len(request.baseline) == 1

    def test_invalid_dict_raises_validation_error(self):
        """Invalid dict should raise ValidationError"""
        data = {
            "baseline": [],  # Empty baseline
            "query": "test"
        }

        with pytest.raises(ValidationError):
            validate_request(data)


class TestFormatValidationErrors:
    """Test format_validation_errors helper"""

    def test_format_single_error(self):
        """Single validation error should be formatted"""
        try:
            SentenceInput(sentence="", id="1")
        except ValidationError as e:
            error_response = format_validation_errors(e)
            assert error_response.error == "Request validation failed"
            assert len(error_response.details) > 0
            assert error_response.details[0].field == "sentence"

    def test_format_multiple_errors(self):
        """Multiple validation errors should be formatted"""
        try:
            AnalysisRequest(
                baseline=[
                    SentenceInput(sentence="Test 1", id="1"),
                    SentenceInput(sentence="Test 2", id="1")  # Duplicate ID
                ]
            )
        except ValidationError as e:
            error_response = format_validation_errors(e)
            assert error_response.error == "Request validation failed"
            assert len(error_response.details) > 0


class TestSentimentDataConstraints:
    """Test SentimentData score constraints"""

    def test_valid_score_range(self):
        """Score within [-1, 1] should pass"""
        sentiment = SentimentData(label="positive", score=0.75)
        assert sentiment.score == 0.75

    def test_score_too_high_fails(self):
        """Score > 1 should fail"""
        with pytest.raises(ValidationError):
            SentimentData(label="positive", score=1.5)

    def test_score_too_low_fails(self):
        """Score < -1 should fail"""
        with pytest.raises(ValidationError):
            SentimentData(label="negative", score=-1.5)

    def test_boundary_scores(self):
        """Boundary scores (-1, 1) should pass"""
        pos = SentimentData(label="positive", score=1.0)
        neg = SentimentData(label="negative", score=-1.0)
        assert pos.score == 1.0
        assert neg.score == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
