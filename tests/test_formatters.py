"""
Unit Tests for Response Formatters

Tests transformation of raw analysis results into API responses.
"""

import pytest
import json
import numpy as np

from src.utils.formatters import (
    AnalysisFormatter,
    format_success_response,
    format_error_response
)
from src.utils.validators import (
    SentenceInput,
    AnalysisResponse
)


class TestAnalysisFormatter:
    """Test AnalysisFormatter class"""

    @pytest.fixture
    def formatter(self):
        """Create formatter instance"""
        return AnalysisFormatter()

    @pytest.fixture
    def mock_baseline_data(self):
        """Mock baseline analysis data"""
        return {
            'sentences': [
                SentenceInput(sentence="I want my money back", id="1"),
                SentenceInput(sentence="Cannot withdraw funds", id="2"),
                SentenceInput(sentence="Great app", id="3"),
                SentenceInput(sentence="Love it", id="4")
            ],
            'labels': np.array([0, 0, 1, 1]),  # 2 clusters
            'cluster_sentiments': {
                0: {
                    'overall': 'negative',
                    'average_score': -0.70,
                    'distribution': {'positive': 0, 'neutral': 0, 'negative': 2}
                },
                1: {
                    'overall': 'positive',
                    'average_score': 0.65,
                    'distribution': {'positive': 2, 'neutral': 0, 'negative': 0}
                }
            },
            'cluster_insights': {
                0: {
                    'title': 'Money & Withdrawal',
                    'keywords': ['money', 'withdraw', 'funds'],
                    'key_insights': ['High negative sentiment - requires attention']
                },
                1: {
                    'title': 'App Experience',
                    'keywords': ['app', 'love', 'great'],
                    'key_insights': ['Overwhelmingly positive feedback']
                }
            },
            'sentence_sentiments': {
                "1": {'label': 'negative', 'score': -0.75},
                "2": {'label': 'negative', 'score': -0.65},
                "3": {'label': 'positive', 'score': 0.60},
                "4": {'label': 'positive', 'score': 0.70}
            }
        }

    def test_format_baseline_only_response(self, formatter, mock_baseline_data):
        """Format response with baseline data only"""
        response = formatter.format_response(
            baseline_sentences=mock_baseline_data['sentences'],
            cluster_labels=mock_baseline_data['labels'],
            cluster_sentiment_results=mock_baseline_data['cluster_sentiments'],
            cluster_insights_results=mock_baseline_data['cluster_insights'],
            sentence_level_sentiments=mock_baseline_data['sentence_sentiments'],
            query="overview",
            theme="feedback"
        )

        # Check response type
        assert isinstance(response, AnalysisResponse)

        # Check summary
        assert response.summary.total_sentences == 4
        assert response.summary.clusters_found == 2
        assert response.summary.query == "overview"
        assert response.summary.theme == "feedback"

        # Check clusters
        assert len(response.clusters) == 2

        # Clusters should be sorted by size (both have size 2, so order may vary)
        for cluster in response.clusters:
            assert cluster.size == 2
            assert cluster.source == "baseline"
            assert len(cluster.sentences) == 2
            assert len(cluster.keywords) > 0
            assert len(cluster.key_insights) > 0

        # Check no comparison insights
        assert response.comparison_insights is None

    def test_format_with_comparison(self, formatter):
        """Format response with both baseline and comparison data"""
        baseline = [SentenceInput(sentence="Test baseline", id="b1")]
        comparison = [SentenceInput(sentence="Test comparison", id="c1")]

        baseline_labels = np.array([0])
        comparison_labels = np.array([0])

        cluster_sentiments = {
            0: {
                'overall': 'neutral',
                'average_score': 0.0,
                'distribution': {'positive': 0, 'neutral': 1, 'negative': 0}
            }
        }

        cluster_insights = {
            0: {
                'title': 'Test Cluster',
                'keywords': ['test'],
                'key_insights': ['Test insight']
            }
        }

        sentence_sentiments = {
            "b1": {'label': 'neutral', 'score': 0.0},
            "c1": {'label': 'neutral', 'score': 0.0}
        }

        response = formatter.format_response(
            baseline_sentences=baseline,
            cluster_labels=baseline_labels,
            cluster_sentiment_results=cluster_sentiments,
            cluster_insights_results=cluster_insights,
            sentence_level_sentiments=sentence_sentiments,
            query="test",
            comparison_sentences=comparison,
            comparison_labels=comparison_labels
        )

        # Should have 2 clusters (1 baseline + 1 comparison)
        assert len(response.clusters) == 2
        assert response.summary.total_sentences == 2

        # Should have comparison insights
        assert response.comparison_insights is not None

    def test_clusters_sorted_by_size(self, formatter):
        """Clusters should be sorted by size (largest first)"""
        sentences = [
            SentenceInput(sentence=f"Test {i}", id=f"id-{i}")
            for i in range(10)
        ]

        # Cluster 0: 7 items, Cluster 1: 3 items
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        cluster_sentiments = {
            0: {
                'overall': 'neutral',
                'average_score': 0.0,
                'distribution': {'positive': 0, 'neutral': 7, 'negative': 0}
            },
            1: {
                'overall': 'neutral',
                'average_score': 0.0,
                'distribution': {'positive': 0, 'neutral': 3, 'negative': 0}
            }
        }

        cluster_insights = {
            i: {
                'title': f'Cluster {i}',
                'keywords': ['test'],
                'key_insights': []
            }
            for i in [0, 1]
        }

        sentence_sentiments = {
            f"id-{i}": {'label': 'neutral', 'score': 0.0}
            for i in range(10)
        }

        response = formatter.format_response(
            baseline_sentences=sentences,
            cluster_labels=labels,
            cluster_sentiment_results=cluster_sentiments,
            cluster_insights_results=cluster_insights,
            sentence_level_sentiments=sentence_sentiments,
            query="test"
        )

        # First cluster should be largest
        assert response.clusters[0].size == 7
        assert response.clusters[1].size == 3

    def test_noise_points_excluded(self, formatter):
        """Noise points (-1 label) should be excluded from clusters"""
        sentences = [
            SentenceInput(sentence="Test 1", id="1"),
            SentenceInput(sentence="Test 2", id="2"),  # Noise
            SentenceInput(sentence="Test 3", id="3")
        ]

        # Label 2 is noise (-1)
        labels = np.array([0, -1, 0])

        cluster_sentiments = {
            0: {
                'overall': 'neutral',
                'average_score': 0.0,
                'distribution': {'positive': 0, 'neutral': 2, 'negative': 0}
            }
        }

        cluster_insights = {
            0: {
                'title': 'Test Cluster',
                'keywords': ['test'],
                'key_insights': []
            }
        }

        sentence_sentiments = {
            str(i): {'label': 'neutral', 'score': 0.0}
            for i in range(1, 4)
        }

        response = formatter.format_response(
            baseline_sentences=sentences,
            cluster_labels=labels,
            cluster_sentiment_results=cluster_sentiments,
            cluster_insights_results=cluster_insights,
            sentence_level_sentiments=sentence_sentiments,
            query="test"
        )

        # Should have 1 cluster with 2 sentences (noise excluded)
        assert len(response.clusters) == 1
        assert response.clusters[0].size == 2

        # Unclustered count should be 1
        assert response.summary.unclustered == 1

    def test_overall_sentiment_calculation(self, formatter):
        """Overall sentiment should be weighted by cluster size"""
        sentences = [
            SentenceInput(sentence=f"Test {i}", id=f"id-{i}")
            for i in range(10)
        ]

        # Cluster 0: 8 positive, Cluster 1: 2 negative
        labels = np.array([0] * 8 + [1] * 2)

        cluster_sentiments = {
            0: {
                'overall': 'positive',
                'average_score': 0.70,
                'distribution': {'positive': 8, 'neutral': 0, 'negative': 0}
            },
            1: {
                'overall': 'negative',
                'average_score': -0.60,
                'distribution': {'positive': 0, 'neutral': 0, 'negative': 2}
            }
        }

        cluster_insights = {
            i: {
                'title': f'Cluster {i}',
                'keywords': ['test'],
                'key_insights': []
            }
            for i in [0, 1]
        }

        sentence_sentiments = {
            f"id-{i}": {
                'label': 'positive' if i < 8 else 'negative',
                'score': 0.70 if i < 8 else -0.60
            }
            for i in range(10)
        }

        response = formatter.format_response(
            baseline_sentences=sentences,
            cluster_labels=labels,
            cluster_sentiment_results=cluster_sentiments,
            cluster_insights_results=cluster_insights,
            sentence_level_sentiments=sentence_sentiments,
            query="test"
        )

        # Overall should be positive (8 positive > 2 negative)
        assert response.summary.overall_sentiment == 'positive'

    def test_comparison_insights_generation(self, formatter):
        """Comparison insights should identify unique and shared themes"""
        # Create clusters with different titles
        baseline_clusters = []
        comparison_clusters = []

        # Baseline: Theme A, Theme B
        # Comparison: Theme B, Theme C
        # Shared: Theme B

        # This would require more complex mocking, so let's test the helper directly
        from src.utils.validators import ClusterOutput, ClusterSentiment, SentimentData

        baseline = [
            ClusterOutput(
                id="b1",
                title="Theme A",
                sentences=[],
                size=1,
                sentiment=ClusterSentiment(
                    overall="neutral",
                    distribution={"positive": 0, "neutral": 1, "negative": 0},
                    average_score=0.0
                ),
                key_insights=[],
                keywords=[]
            ),
            ClusterOutput(
                id="b2",
                title="Theme B",
                sentences=[],
                size=1,
                sentiment=ClusterSentiment(
                    overall="neutral",
                    distribution={"positive": 0, "neutral": 1, "negative": 0},
                    average_score=0.0
                ),
                key_insights=[],
                keywords=[]
            )
        ]

        comparison = [
            ClusterOutput(
                id="c1",
                title="Theme B",
                sentences=[],
                size=1,
                sentiment=ClusterSentiment(
                    overall="neutral",
                    distribution={"positive": 0, "neutral": 1, "negative": 0},
                    average_score=0.0
                ),
                key_insights=[],
                keywords=[]
            ),
            ClusterOutput(
                id="c2",
                title="Theme C",
                sentences=[],
                size=1,
                sentiment=ClusterSentiment(
                    overall="neutral",
                    distribution={"positive": 0, "neutral": 1, "negative": 0},
                    average_score=0.0
                ),
                key_insights=[],
                keywords=[]
            )
        ]

        insights = formatter._build_comparison_insights(baseline, comparison)

        assert "Theme A" in insights.baseline_only_themes
        assert "Theme C" in insights.comparison_only_themes
        assert "Theme B" in insights.shared_themes


class TestAPIGatewayFormatters:
    """Test API Gateway response formatters"""

    def test_format_success_response(self):
        """Success response should have correct structure"""
        # Create minimal valid response
        from src.utils.validators import AnalysisSummary

        response = AnalysisResponse(
            clusters=[],
            summary=AnalysisSummary(
                total_sentences=0,
                clusters_found=0,
                unclustered=0,
                overall_sentiment="neutral",
                query="test"
            )
        )

        api_response = format_success_response(response, request_id="test-123")

        assert api_response['statusCode'] == 200
        assert 'headers' in api_response
        assert api_response['headers']['Content-Type'] == 'application/json'
        assert 'Access-Control-Allow-Origin' in api_response['headers']
        body = json.loads(api_response['body'])
        assert body['request_id'] == "test-123"

    def test_format_error_response(self):
        """Error response should have correct structure"""
        api_response = format_error_response(
            error_message="Test error",
            status_code=400,
            request_id="test-456"
        )

        assert api_response['statusCode'] == 400
        body = json.loads(api_response['body'])
        assert body['error'] == "Test error"
        assert body['request_id'] == "test-456"
        assert 'Access-Control-Allow-Origin' in api_response['headers']

    def test_cors_headers_present(self):
        """Both success and error responses should have CORS headers"""
        from src.utils.validators import AnalysisSummary

        success = format_success_response(
            AnalysisResponse(
                clusters=[],
                summary=AnalysisSummary(
                    total_sentences=0,
                    clusters_found=0,
                    unclustered=0,
                    overall_sentiment="neutral",
                    query="test"
                )
            )
        )

        error = format_error_response("Test error")

        # Both should have CORS headers
        for response in [success, error]:
            headers = response['headers']
            assert headers['Access-Control-Allow-Origin'] == '*'
            assert 'Access-Control-Allow-Headers' in headers
            assert 'Access-Control-Allow-Methods' in headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
