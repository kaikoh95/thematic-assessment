"""
Integration Tests

Tests the complete analysis pipeline from request to response.
Requires ML models to be loaded (slower tests).
"""

import pytest
import json
from src.lambda_function import handler


class MockContext:
    """Mock Lambda context for testing"""
    request_id = "test-integration-123"
    function_name = "text-analysis-test"
    memory_limit_in_mb = 3008


class TestEndToEndAnalysis:
    """Test complete analysis pipeline"""

    def test_baseline_only_analysis(self):
        """Complete analysis with baseline data only"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "I want my money back", "id": "1"},
                    {"sentence": "Cannot withdraw my funds", "id": "2"},
                    {"sentence": "Terrible customer service", "id": "3"},
                    {"sentence": "Best investment app ever", "id": "4"},
                    {"sentence": "Love the interface", "id": "5"},
                    {"sentence": "Great platform to trade", "id": "6"},
                    {"sentence": "Easy to use", "id": "7"},
                    {"sentence": "Support is unhelpful", "id": "8"}
                ],
                "query": "overview",
                "surveyTitle": "Q1 Feedback",
                "theme": "user experience"
            })
        }

        response = handler(event, MockContext())

        # Check successful response
        assert response['statusCode'] == 200
        assert 'body' in response

        body = json.loads(response['body'])

        # Check summary
        assert body['summary']['total_sentences'] == 8
        assert body['summary']['clusters_found'] > 0
        assert body['summary']['query'] == "overview"
        assert body['summary']['theme'] == "user experience"

        # Check clusters exist
        assert len(body['clusters']) > 0

        # Each cluster should have required fields
        for cluster in body['clusters']:
            assert 'id' in cluster
            assert 'title' in cluster
            assert 'sentences' in cluster
            assert 'size' in cluster
            assert 'sentiment' in cluster
            assert 'key_insights' in cluster
            assert 'keywords' in cluster

            # Each sentence should have required fields
            for sentence in cluster['sentences']:
                assert 'sentence' in sentence
                assert 'id' in sentence
                assert 'sentiment' in sentence
                assert sentence['sentiment']['label'] in ['positive', 'neutral', 'negative']
                assert -1 <= sentence['sentiment']['score'] <= 1

    def test_baseline_and_comparison_analysis(self):
        """Analysis with both baseline and comparison data"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Poor customer service", "id": "b1"},
                    {"sentence": "Long wait times", "id": "b2"},
                    {"sentence": "Support is unhelpful", "id": "b3"}
                ],
                "comparison": [
                    {"sentence": "Excellent support team", "id": "c1"},
                    {"sentence": "Quick responses", "id": "c2"},
                    {"sentence": "Very helpful staff", "id": "c3"}
                ],
                "query": "support comparison"
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 200
        body = json.loads(response['body'])

        # Should have both baseline and comparison clusters
        assert body['summary']['total_sentences'] == 6

        # Should have comparison insights
        assert body['comparison_insights'] is not None
        assert 'baseline_only_themes' in body['comparison_insights']
        assert 'comparison_only_themes' in body['comparison_insights']
        assert 'shared_themes' in body['comparison_insights']

    def test_validation_error_handling(self):
        """Invalid request should return 400 error"""
        event = {
            "body": json.dumps({
                "baseline": [],  # Empty baseline (invalid)
                "query": "test"
            })
        }

        response = handler(event, MockContext())

        # Should return validation error
        assert response['statusCode'] == 400
        assert 'error' in response['body']

    def test_duplicate_id_error(self):
        """Duplicate IDs should be rejected"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Test 1", "id": "1"},
                    {"sentence": "Test 2", "id": "1"}  # Duplicate ID
                ],
                "query": "test"
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'validation' in body['error'].lower()

    def test_malformed_json_error(self):
        """Malformed JSON should return error"""
        event = {
            "body": "{'invalid': json}"  # Invalid JSON
        }

        response = handler(event, MockContext())

        # Should handle parsing error
        assert response['statusCode'] >= 400

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test with larger dataset (100 sentences)"""
        import time

        baseline = [
            {"sentence": f"This is test sentence number {i}", "id": f"id-{i}"}
            for i in range(100)
        ]

        event = {
            "body": json.dumps({
                "baseline": baseline,
                "query": "performance test"
            })
        }

        start = time.time()
        response = handler(event, MockContext())
        duration = time.time() - start

        # Should complete successfully
        assert response['statusCode'] == 200

        # Should complete reasonably fast (< 10s for 100 sentences)
        assert duration < 10.0, f"Processing took {duration:.2f}s (expected <10s)"

        body = json.loads(response['body'])
        assert body['summary']['total_sentences'] == 100

    def test_sentiment_distribution_accuracy(self):
        """Sentiment analysis should correctly classify clear sentiments"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "This is absolutely amazing! Love it!", "id": "p1"},
                    {"sentence": "Best product ever! Highly recommend!", "id": "p2"},
                    {"sentence": "Outstanding quality and service!", "id": "p3"},
                    {"sentence": "Terrible! Worst experience ever!", "id": "n1"},
                    {"sentence": "Awful service! Complete waste of money!", "id": "n2"},
                ],
                "query": "sentiment test"
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 200
        body = json.loads(response['body'])

        # Count sentiments across all clusters
        positive_count = 0
        negative_count = 0

        for cluster in body['clusters']:
            for sentence in cluster['sentences']:
                if sentence['sentiment']['label'] == 'positive':
                    positive_count += 1
                elif sentence['sentiment']['label'] == 'negative':
                    negative_count += 1

        # Should have 3 positive and 2 negative
        assert positive_count == 3, f"Expected 3 positive, got {positive_count}"
        assert negative_count == 2, f"Expected 2 negative, got {negative_count}"

    def test_cluster_keywords_relevance(self):
        """Cluster keywords should be relevant to content"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "The food was delicious and amazing", "id": "1"},
                    {"sentence": "Great food, loved the taste", "id": "2"},
                    {"sentence": "Food quality was excellent", "id": "3"},
                ],
                "query": "food feedback"
            })
        }

        response = handler(event, MockContext())

        assert response['statusCode'] == 200
        body = json.loads(response['body'])

        # Should have at least 1 cluster
        assert len(body['clusters']) > 0

        # First cluster should have "food" as keyword
        cluster = body['clusters'][0]
        keywords_str = ' '.join(cluster['keywords']).lower()
        assert 'food' in keywords_str, f"Expected 'food' in keywords, got {cluster['keywords']}"


class TestResponseStructure:
    """Test response structure compliance"""

    def test_response_has_all_required_fields(self):
        """Response should have all required fields per API spec"""
        event = {
            "body": json.dumps({
                "baseline": [
                    {"sentence": "Test sentence", "id": "1"}
                ],
                "query": "test"
            })
        }

        response = handler(event, MockContext())
        body = json.loads(response['body'])

        # Top-level fields
        assert 'clusters' in body
        assert 'summary' in body

        # Summary fields
        summary = body['summary']
        assert 'total_sentences' in summary
        assert 'clusters_found' in summary
        assert 'unclustered' in summary
        assert 'overall_sentiment' in summary
        assert 'query' in summary

        # If there are clusters
        if body['clusters']:
            cluster = body['clusters'][0]

            # Cluster fields
            assert 'id' in cluster
            assert 'title' in cluster
            assert 'sentences' in cluster
            assert 'size' in cluster
            assert 'sentiment' in cluster
            assert 'key_insights' in cluster
            assert 'keywords' in cluster

            # Cluster sentiment fields
            assert 'overall' in cluster['sentiment']
            assert 'distribution' in cluster['sentiment']
            assert 'average_score' in cluster['sentiment']

    def test_cors_headers_in_response(self):
        """Response should include CORS headers"""
        event = {
            "body": json.dumps({
                "baseline": [{"sentence": "Test", "id": "1"}],
                "query": "test"
            })
        }

        response = handler(event, MockContext())

        headers = response['headers']
        assert headers['Access-Control-Allow-Origin'] == '*'
        assert 'Content-Type' in headers


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
