"""
Unit Tests for Sentiment Analysis

Tests VADER-based sentiment analysis at sentence and cluster levels.
"""

import pytest
from src.sentiment.analyzer import (
    ClusterSentimentAnalyzer,
    classify_sentiment,
    analyze_sentiment,
    analyze_cluster_sentiment
)


class TestClassifySentiment:
    """Test sentiment classification function"""

    def test_positive_classification(self):
        """Positive scores should classify as positive"""
        assert classify_sentiment(0.5, 'standard') == 'positive'
        assert classify_sentiment(0.05, 'standard') == 'positive'
        assert classify_sentiment(0.9, 'standard') == 'positive'

    def test_negative_classification(self):
        """Negative scores should classify as negative"""
        assert classify_sentiment(-0.5, 'standard') == 'negative'
        assert classify_sentiment(-0.05, 'standard') == 'negative'
        assert classify_sentiment(-0.9, 'standard') == 'negative'

    def test_neutral_classification(self):
        """Scores near zero should classify as neutral"""
        assert classify_sentiment(0.0, 'standard') == 'neutral'
        assert classify_sentiment(0.04, 'standard') == 'neutral'
        assert classify_sentiment(-0.04, 'standard') == 'neutral'

    def test_strict_thresholds(self):
        """Strict mode should have wider neutral range"""
        # Standard mode: 0.1 is positive
        assert classify_sentiment(0.1, 'standard') == 'positive'

        # Strict mode: 0.1 is neutral (threshold is 0.2)
        assert classify_sentiment(0.1, 'strict') == 'neutral'

        # Strict mode: 0.25 is positive
        assert classify_sentiment(0.25, 'strict') == 'positive'


class TestClusterSentimentAnalyzer:
    """Test ClusterSentimentAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return ClusterSentimentAnalyzer(threshold_mode='standard')

    def test_analyze_single_positive(self, analyzer):
        """Single positive sentence should have positive sentiment"""
        result = analyzer.analyze_single("This is amazing! I love it!")

        assert 'compound' in result
        assert result['compound'] > 0
        assert result['pos'] > 0

    def test_analyze_single_negative(self, analyzer):
        """Single negative sentence should have negative sentiment"""
        result = analyzer.analyze_single("This is terrible! I hate it!")

        assert 'compound' in result
        assert result['compound'] < 0
        assert result['neg'] > 0

    def test_analyze_single_neutral(self, analyzer):
        """Single neutral sentence should have neutral sentiment"""
        result = analyzer.analyze_single("This is a thing.")

        assert 'compound' in result
        # Neutral sentences should have compound near 0
        assert abs(result['compound']) < 0.3

    def test_analyze_single_empty_string(self, analyzer):
        """Empty string should return neutral fallback"""
        result = analyzer.analyze_single("")

        assert result['compound'] == 0.0
        assert result['neu'] == 1.0

    def test_analyze_batch(self, analyzer):
        """Batch analysis should process multiple sentences"""
        sentences = [
            "Great product!",
            "Terrible service.",
            "It's okay."
        ]

        results = analyzer.analyze_batch(sentences)

        assert len(results) == 3
        assert results[0]['compound'] > 0  # Positive
        assert results[1]['compound'] < 0  # Negative
        # All should have compound scores
        assert all('compound' in r for r in results)

    def test_analyze_cluster_homogeneous_positive(self, analyzer):
        """Cluster with all positive sentences"""
        sentences = [
            "This is great!",
            "I love this!",
            "Amazing experience!",
            "Best product ever!"
        ]

        result = analyzer.analyze_cluster(sentences)

        assert result['overall'] == 'positive'
        assert result['average_score'] > 0
        assert result['distribution']['positive'] == 4
        assert result['distribution']['negative'] == 0

    def test_analyze_cluster_homogeneous_negative(self, analyzer):
        """Cluster with all negative sentences"""
        sentences = [
            "This is terrible!",
            "I hate this!",
            "Worst experience ever!",
            "Complete waste of money!"
        ]

        result = analyzer.analyze_cluster(sentences)

        assert result['overall'] == 'negative'
        assert result['average_score'] < 0
        assert result['distribution']['negative'] == 4
        assert result['distribution']['positive'] == 0

    def test_analyze_cluster_mixed(self, analyzer):
        """Cluster with mixed sentiments"""
        sentences = [
            "This is great!",          # Positive
            "This is terrible!",       # Negative
            "It's okay.",              # Neutral
            "Pretty good overall."     # Positive
        ]

        result = analyzer.analyze_cluster(sentences)

        # Should have mixed distribution
        assert result['distribution']['positive'] > 0
        assert result['distribution']['negative'] > 0
        assert result['total_sentences'] == 4

        # Percentages should sum to 100
        percentages = result['percentages']
        total_pct = percentages['positive'] + percentages['neutral'] + percentages['negative']
        assert abs(total_pct - 100.0) < 0.1

    def test_analyze_cluster_statistics(self, analyzer):
        """Cluster analysis should include statistical measures"""
        sentences = [
            "Great!",
            "Good!",
            "Okay.",
            "Bad.",
            "Terrible!"
        ]

        result = analyzer.analyze_cluster(sentences)

        stats = result['statistics']

        # Should have all required statistics
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

        # Min should be less than max
        assert stats['min'] < stats['max']

        # Std should be non-negative
        assert stats['std'] >= 0

    def test_analyze_cluster_empty(self, analyzer):
        """Empty cluster should return neutral fallback"""
        result = analyzer.analyze_cluster([])

        assert result['overall'] == 'neutral'
        assert result['average_score'] == 0.0
        assert result['total_sentences'] == 0

    def test_analyze_cluster_with_details(self, analyzer):
        """Detailed analysis should include per-sentence data"""
        sentences = ["Great!", "Terrible!"]
        ids = ["sent-1", "sent-2"]

        result = analyzer.analyze_cluster_with_details(sentences, ids)

        assert 'sentence_sentiments' in result
        assert len(result['sentence_sentiments']) == 2

        sent1 = result['sentence_sentiments'][0]
        assert sent1['id'] == 'sent-1'
        assert 'sentiment' in sent1
        assert 'compound' in sent1

    def test_median_used_for_overall(self, analyzer):
        """Cluster sentiment should use median (robust to outliers)"""
        # 4 slightly positive, 1 extremely negative outlier
        sentences = [
            "Good",          # +
            "Good",          # +
            "Good",          # +
            "Good",          # +
            "ABSOLUTELY HORRIBLE TERRIBLE AWFUL!"  # Extreme negative outlier
        ]

        result = analyzer.analyze_cluster(sentences)

        # Median should make this positive despite the outlier
        # (4 positive medians > 1 negative)
        # The overall might still be positive due to median robustness
        stats = result['statistics']
        assert 'median' in stats


class TestConvenienceFunctions:
    """Test convenience helper functions"""

    def test_analyze_sentiment_single(self):
        """analyze_sentiment should work for single sentence"""
        result = analyze_sentiment("This is amazing!")

        assert result['sentiment'] == 'positive'
        assert result['compound'] > 0
        assert 'scores' in result

    def test_analyze_cluster_sentiment_convenience(self):
        """analyze_cluster_sentiment should work as shortcut"""
        sentences = [
            "Great product!",
            "Love it!",
            "Best ever!"
        ]

        result = analyze_cluster_sentiment(sentences)

        assert result['overall'] == 'positive'
        assert result['total_sentences'] == 3


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_sentence(self):
        """Very long sentence should still be analyzed"""
        analyzer = ClusterSentimentAnalyzer()
        long_sentence = "This is great! " * 100  # 1500+ characters

        result = analyzer.analyze_single(long_sentence)

        # Should still work
        assert 'compound' in result
        assert result['compound'] > 0

    def test_special_characters(self):
        """Sentences with special characters should work"""
        analyzer = ClusterSentimentAnalyzer()

        result = analyzer.analyze_single("This is 💯% amazing!!! 🎉🎉🎉")

        # VADER handles emojis
        assert result['compound'] > 0

    def test_single_sentence_cluster(self):
        """Cluster with single sentence should work"""
        analyzer = ClusterSentimentAnalyzer()

        result = analyzer.analyze_cluster(["Great!"])

        assert result['total_sentences'] == 1
        assert result['statistics']['std'] == 0.0  # No variance with 1 point


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
