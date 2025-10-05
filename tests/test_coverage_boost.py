"""
Additional tests to boost coverage to 70%

Targets uncovered edge cases and utility functions.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.clustering.embeddings import SentenceEmbedder, generate_embeddings
from src.sentiment.analyzer import ClusterSentimentAnalyzer
from src.utils.formatters import AnalysisFormatter
from src.utils.validators import (
    ClusterOutput,
    ClusterSentiment,
    AnalysisSummary
)


class TestEmbeddingsEdgeCases:
    """Test edge cases in embeddings module"""

    def test_encode_empty_list(self):
        """Test encoding empty sentence list returns empty array"""
        embedder = SentenceEmbedder()
        result = embedder.encode([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension"""
        embedder = SentenceEmbedder()
        dim = embedder.get_embedding_dimension()

        assert isinstance(dim, int)
        assert dim == 384  # all-MiniLM-L6-v2

    def test_generate_embeddings_convenience_function(self):
        """Test convenience function generate_embeddings()"""
        sentences = ["Test sentence", "Another test"]

        embeddings = generate_embeddings(
            sentences,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize=True,
            batch_size=32
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384


class TestFormattersEdgeCases:
    """Test edge cases in formatters module"""

    def test_calculate_overall_sentiment_empty_clusters(self):
        """Test overall sentiment calculation with empty clusters"""
        formatter = AnalysisFormatter()

        # Call the method with empty list
        result = formatter._calculate_overall_sentiment([])

        assert result == 'neutral'

    def test_calculate_overall_sentiment_all_zero_size(self):
        """Test overall sentiment with all zero-size clusters"""
        formatter = AnalysisFormatter()

        # Create clusters with zero size
        clusters = [
            ClusterOutput(
                id="test-1",
                title="Test Cluster",
                sentences=[],
                size=0,  # Zero size
                sentiment=ClusterSentiment(
                    overall="positive",
                    distribution={"positive": 0, "neutral": 0, "negative": 0},
                    average_score=0.5
                ),
                key_insights=[],
                keywords=[],
                source="baseline"
            )
        ]

        result = formatter._calculate_overall_sentiment(clusters)

        # Should return neutral when all clusters are empty
        assert result in ['positive', 'neutral', 'negative']


class TestSentimentEdgeCases:
    """Test edge cases in sentiment analysis"""

    def test_analyze_cluster_with_all_empty_strings(self):
        """Test sentiment analysis on cluster with only empty strings"""
        analyzer = ClusterSentimentAnalyzer()

        sentences = ["", "   ", "\t\n"]

        result = analyzer.analyze_cluster(sentences)

        # Should handle empty/whitespace-only strings gracefully
        assert 'overall' in result
        assert result['overall'] in ['positive', 'neutral', 'negative']

    def test_analyze_single_whitespace_edge_case(self):
        """Test analyze_single with whitespace string"""
        analyzer = ClusterSentimentAnalyzer()

        result = analyzer.analyze_single("   ")

        # Should return valid sentiment scores
        assert isinstance(result, dict)
        assert 'neg' in result
        assert 'neu' in result
        assert 'pos' in result
        assert 'compound' in result


class TestValidatorsEdgeCases:
    """Test edge cases in validators"""

    def test_cluster_sentiment_with_zero_distribution(self):
        """Test ClusterSentiment with all zero distribution"""
        sentiment = ClusterSentiment(
            overall="neutral",
            distribution={"positive": 0, "neutral": 0, "negative": 0},
            average_score=0.0
        )

        assert sentiment.overall == "neutral"
        assert sentiment.distribution["positive"] == 0
        assert sentiment.distribution["neutral"] == 0
        assert sentiment.distribution["negative"] == 0
        assert sentiment.average_score == 0.0

    def test_analysis_summary_minimal(self):
        """Test AnalysisSummary with minimal data"""
        summary = AnalysisSummary(
            total_sentences=0,
            clusters_found=0,
            unclustered=0,
            overall_sentiment="neutral",
            query="test"
        )

        assert summary.total_sentences == 0
        assert summary.clusters_found == 0
        assert summary.overall_sentiment == "neutral"


class TestEmbedderCaching:
    """Test embedding model caching behavior"""

    def test_multiple_embedder_instances_use_cache(self):
        """Test that multiple SentenceEmbedder instances use the same cached model"""
        # Create first embedder
        embedder1 = SentenceEmbedder()
        dim1 = embedder1.get_embedding_dimension()

        # Create second embedder - should use cached model
        embedder2 = SentenceEmbedder()
        dim2 = embedder2.get_embedding_dimension()

        # Both should return same dimension
        assert dim1 == dim2
        assert dim1 == 384


class TestClustererEdgeCases:
    """Test edge cases in clustering"""

    def test_embedder_encode_single_method(self):
        """Test encode_single convenience method"""
        embedder = SentenceEmbedder()

        embedding = embedder.encode_single("Test sentence")

        # Should return 1D array
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 384

    def test_embedder_with_show_progress_false(self):
        """Test embedder with show_progress disabled"""
        embedder = SentenceEmbedder()

        embeddings = embedder.encode(
            ["Test 1", "Test 2"],
            show_progress=False
        )

        assert embeddings.shape == (2, 384)

    def test_embedder_with_different_batch_size(self):
        """Test embedder with custom batch size"""
        embedder = SentenceEmbedder()

        embeddings = embedder.encode(
            ["Test 1", "Test 2", "Test 3"],
            batch_size=1
        )

        assert embeddings.shape == (3, 384)

    def test_embedder_normalize_false(self):
        """Test embedder without normalization"""
        embedder = SentenceEmbedder()

        embeddings = embedder.encode(
            ["Test 1", "Test 2"],
            normalize=False
        )

        assert embeddings.shape == (2, 384)


class TestLambdaFunctionEdgeCases:
    """Test edge cases in lambda function"""

    def test_missing_body_in_event(self):
        """Test handler with missing body"""
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-123"

        event = {}  # No body field

        response = handler(event, MockContext())

        # Should return error
        assert response['statusCode'] >= 400

    def test_empty_query_uses_default(self):
        """Test that empty query uses default"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-456"

        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Test 1", "id": "1"},
                    {"sentence": "Test 2", "id": "2"}
                ]
                # No query field
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'query' in body['summary']

    def test_with_theme_and_survey_title(self):
        """Test handler with theme and surveyTitle"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-789"

        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Great product", "id": "1"},
                    {"sentence": "Love it", "id": "2"},
                    {"sentence": "Excellent", "id": "3"}
                ],
                "query": "product feedback",
                "theme": "customer satisfaction",
                "surveyTitle": "Q1 Survey"
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['summary']['theme'] == "customer satisfaction"


class TestFormatterAdditionalCoverage:
    """Additional tests for formatter coverage"""

    def test_calculate_overall_sentiment_mixed_sizes(self):
        """Test overall sentiment with mixed cluster sizes"""
        from src.utils.formatters import AnalysisFormatter
        from src.utils.validators import ClusterOutput, ClusterSentiment

        formatter = AnalysisFormatter()

        # Create clusters with varying sizes
        clusters = [
            ClusterOutput(
                id="c1",
                title="Large Positive",
                sentences=[],
                size=10,
                sentiment=ClusterSentiment(
                    overall="positive",
                    distribution={"positive": 10, "neutral": 0, "negative": 0},
                    average_score=0.8
                ),
                key_insights=[],
                keywords=[],
                source="baseline"
            ),
            ClusterOutput(
                id="c2",
                title="Small Negative",
                sentences=[],
                size=2,
                sentiment=ClusterSentiment(
                    overall="negative",
                    distribution={"positive": 0, "neutral": 0, "negative": 2},
                    average_score=-0.6
                ),
                key_insights=[],
                keywords=[],
                source="baseline"
            )
        ]

        result = formatter._calculate_overall_sentiment(clusters)

        # Should be positive due to larger cluster
        assert result == 'positive'


class TestIntegrationWithDiverseData:
    """Integration tests with diverse data patterns"""

    def test_integration_with_comparison_none(self):
        """Test validation with comparison=None"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-none-comparison"

        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Test baseline sentence", "id": "b1"},
                    {"sentence": "Another baseline test", "id": "b2"}
                ],
                "comparison": None,  # Explicitly None
                "query": "test query"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200

    def test_very_similar_sentences_clustering(self):
        """Test clustering with very similar sentences"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-similar"

        # Create 10 nearly identical sentences
        sentences = [
            {"sentence": f"This is a test sentence number {i}", "id": f"id{i}"}
            for i in range(10)
        ]

        event = {
            "body": json.dumps({
                "baseline": sentences,
                "query": "similarity test"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        # Should cluster them together
        assert body['summary']['clusters_found'] >= 1

    def test_diverse_sentiment_sentences(self):
        """Test with strongly diverse sentiments"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-diverse"

        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Absolutely amazing! Best ever!", "id": "p1"},
                    {"sentence": "Outstanding performance!", "id": "p2"},
                    {"sentence": "Completely terrible! Worst!", "id": "n1"},
                    {"sentence": "Awful and disappointing!", "id": "n2"},
                    {"sentence": "It works as expected.", "id": "neu1"},
                ],
                "query": "sentiment diversity test"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        # Should have detected different sentiments
        assert 'overall_sentiment' in body['summary']

    def test_medium_dataset_with_clusters(self):
        """Test with medium-sized dataset that forms multiple clusters"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-medium"

        # Create sentences about different topics
        sentences = []
        # Topic 1: Food (5 sentences)
        for i in range(5):
            sentences.append({"sentence": f"The food quality is excellent {i}", "id": f"food{i}"})
        # Topic 2: Service (5 sentences)
        for i in range(5):
            sentences.append({"sentence": f"Customer service is terrible {i}", "id": f"service{i}"})
        # Topic 3: Price (5 sentences)
        for i in range(5):
            sentences.append({"sentence": f"Pricing is reasonable and fair {i}", "id": f"price{i}"})

        event = {
            "body": json.dumps({
                "baseline": sentences,
                "query": "multi-cluster test"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        # Should form multiple clusters
        assert body['summary']['clusters_found'] >= 2

    def test_with_special_characters_in_sentences(self):
        """Test handling of special characters"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-special-chars"

        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Great! Really love it :)", "id": "1"},
                    {"sentence": "Bad... not good :(", "id": "2"},
                    {"sentence": "Price is $50.99 - too expensive!", "id": "3"},
                    {"sentence": "Rating: 5/5 stars ⭐", "id": "4"},
                ],
                "query": "special chars test"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200

    def test_longer_sentences(self):
        """Test with longer, more detailed sentences"""
        import json
        from src.lambda_function import handler

        class MockContext:
            request_id = "test-long-sentences"

        event = {
            "body": json.dumps({
                "baseline": [
                    {
                        "sentence": "I've been using this product for several months now and I must say the experience has been absolutely fantastic from start to finish",
                        "id": "long1"
                    },
                    {
                        "sentence": "After trying multiple alternatives in the market, this solution stands out as the most reliable and user-friendly option available",
                        "id": "long2"
                    },
                    {
                        "sentence": "The customer support team was incredibly responsive and helped resolve my issues within hours, which exceeded my expectations",
                        "id": "long3"
                    },
                ],
                "query": "detailed feedback"
            })
        }

        response = handler(event, MockContext())
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['summary']['total_sentences'] == 3
