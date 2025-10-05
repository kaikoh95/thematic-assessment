"""
Cluster Insights Generation Module

TF-IDF-based keyword extraction and template-based insights.
Generates cluster titles and actionable insights without LLM.

Features:
- Keyword extraction using TF-IDF (unigrams + bigrams)
- Smart cluster title generation
- Template-based insights from statistics
- Common phrase detection
- Production-optimized for customer feedback

Strategy: Fast, deterministic, no external API calls.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ClusterInsightsGenerator:
    """
    Generate insights for text clusters using TF-IDF and statistical analysis

    No LLM required - uses keyword extraction and template-based insights.

    Usage:
        generator = ClusterInsightsGenerator()
        insights = generator.generate_insights(cluster_sentences, sentiment_data)
    """

    def __init__(
        self,
        max_keywords: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        max_df: float = 0.85,
        min_df: int = 1,
        stop_words: str = 'english'
    ):
        """
        Initialize insights generator

        Args:
            max_keywords: Maximum keywords to extract per cluster
            ngram_range: N-gram range (1,2) = unigrams + bigrams
            max_df: Ignore terms in >X% of documents
            min_df: Ignore terms in <X documents
            stop_words: Stop words to filter
        """
        self.max_keywords = max_keywords
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.stop_words = stop_words

    def generate_insights(
        self,
        cluster_sentences: List[str],
        sentiment_data: Optional[Dict] = None,
        total_sentences: Optional[int] = None
    ) -> Dict:
        """
        Generate comprehensive insights for a cluster

        Args:
            cluster_sentences: List of sentences in the cluster
            sentiment_data: Optional sentiment analysis results
            total_sentences: Total sentences in dataset (for prevalence)

        Returns:
            {
                'title': str,
                'keywords': List[str],
                'key_insights': List[str],
                'common_phrases': List[Tuple[str, int]],
                'size': int
            }
        """
        if not cluster_sentences:
            return self._empty_insights()

        cluster_size = len(cluster_sentences)

        # Extract keywords
        keywords = self._extract_keywords(cluster_sentences)

        # Generate cluster title
        title = self._generate_title(keywords, cluster_size)

        # Find common phrases
        phrases = self._find_common_phrases(cluster_sentences)

        # Generate insights
        insights = self._generate_template_insights(
            cluster_sentences=cluster_sentences,
            keywords=keywords,
            phrases=phrases,
            sentiment_data=sentiment_data,
            total_sentences=total_sentences
        )

        return {
            'title': title,
            'keywords': keywords[:self.max_keywords],
            'key_insights': insights,
            'common_phrases': phrases[:5],
            'size': cluster_size
        }

    def _extract_keywords(self, sentences: List[str]) -> List[str]:
        """
        Extract top keywords using TF-IDF

        Prioritizes bigrams (more specific) over unigrams.

        Args:
            sentences: List of sentences in cluster

        Returns:
            List of keywords sorted by importance
        """
        if len(sentences) < 1:
            return []

        try:
            # Configure TF-IDF for short customer feedback text
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                max_df=self.max_df,
                min_df=self.min_df,
                max_features=100,
                sublinear_tf=True,  # Log scaling for short text
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic, min 2 chars
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Aggregate TF-IDF scores across all documents
            feature_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            feature_names = vectorizer.get_feature_names_out()

            # Sort by score (descending)
            top_indices = feature_scores.argsort()[::-1]

            # Separate bigrams and unigrams
            bigrams = []
            unigrams = []

            for idx in top_indices:
                term = feature_names[idx]
                if ' ' in term:
                    bigrams.append(term)
                else:
                    unigrams.append(term)

            # Prioritize bigrams (more specific), then unigrams
            keywords = bigrams[:5] + unigrams[:10]

            return keywords[:self.max_keywords]

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # Fallback: simple word frequency
            return self._fallback_keywords(sentences)

    def _fallback_keywords(self, sentences: List[str]) -> List[str]:
        """
        Fallback keyword extraction using word frequency

        Used when TF-IDF fails (e.g., too few sentences).

        Args:
            sentences: List of sentences

        Returns:
            List of top words
        """
        words = []
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from',
            'have', 'has', 'was', 'were', 'been', 'are', 'not'
        }

        for sentence in sentences:
            # Simple tokenization
            tokens = re.findall(r'\b[a-z]{3,}\b', sentence.lower())
            words.extend([w for w in tokens if w not in stop_words])

        # Count and return top words
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(10)]

    def _generate_title(self, keywords: List[str], cluster_size: int) -> str:
        """
        Generate cluster title from keywords

        Strategy:
        1. Prefer 1-2 bigrams (more specific)
        2. Fall back to 2-3 unigrams
        3. Title case formatting

        Args:
            keywords: Extracted keywords
            cluster_size: Number of sentences in cluster

        Returns:
            Cluster title string
        """
        if not keywords:
            return "General Feedback"

        # Separate bigrams and unigrams
        bigrams = [kw for kw in keywords if ' ' in kw]
        unigrams = [kw for kw in keywords if ' ' not in kw]

        # Strategy 1: Single descriptive bigram
        if bigrams:
            title = bigrams[0]
            # Add second bigram if different enough
            if len(bigrams) > 1:
                title = f"{bigrams[0]} & {bigrams[1]}"
            return self._format_title(title)

        # Strategy 2: Combine 2-3 unigrams
        if len(unigrams) >= 2:
            title = ' & '.join(unigrams[:min(3, len(unigrams))])
            return self._format_title(title)

        # Strategy 3: Single keyword
        if keywords:
            return self._format_title(keywords[0])

        return "General Feedback"

    def _format_title(self, title: str) -> str:
        """
        Format title with proper capitalization

        Args:
            title: Raw title string

        Returns:
            Formatted title
        """
        # Capitalize first letter of each word
        words = title.split()
        formatted = ' '.join(word.capitalize() for word in words)

        # Limit length
        if len(formatted) > 50:
            formatted = formatted[:47] + "..."

        return formatted

    def _find_common_phrases(
        self,
        sentences: List[str],
        min_frequency: int = 2,
        top_n: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Find common phrases (bigrams) in cluster

        Args:
            sentences: List of sentences
            min_frequency: Minimum occurrence frequency
            top_n: Number of top phrases to return

        Returns:
            List of (phrase, frequency) tuples
        """
        # Extract all bigrams
        bigrams = []

        for sentence in sentences:
            # Tokenize
            words = re.findall(r'\b[a-z]{3,}\b', sentence.lower())

            # Generate bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)

        # Count frequencies
        phrase_counts = Counter(bigrams)

        # Filter by minimum frequency
        common_phrases = [
            (phrase, count)
            for phrase, count in phrase_counts.most_common()
            if count >= min_frequency
        ]

        return common_phrases[:top_n]

    def _generate_template_insights(
        self,
        cluster_sentences: List[str],
        keywords: List[str],
        phrases: List[Tuple[str, int]],
        sentiment_data: Optional[Dict] = None,
        total_sentences: Optional[int] = None
    ) -> List[str]:
        """
        Generate insights using templates and statistics

        No LLM - uses statistical patterns and templates.

        Args:
            cluster_sentences: Sentences in cluster
            keywords: Extracted keywords
            phrases: Common phrases
            sentiment_data: Sentiment analysis results
            total_sentences: Total sentences in dataset

        Returns:
            List of insight strings (2-5 insights)
        """
        insights = []
        cluster_size = len(cluster_sentences)

        # Insight 1: Prevalence (if total known)
        if total_sentences and total_sentences > 0:
            percentage = (cluster_size / total_sentences) * 100

            if percentage >= 30:
                insights.append(
                    f"{percentage:.0f}% of feedback relates to {keywords[0] if keywords else 'this topic'}"
                )
            elif percentage >= 15:
                insights.append(
                    f"{percentage:.0f}% of feedback mentions {keywords[0] if keywords else 'this theme'}"
                )

        # Insight 2: Sentiment
        if sentiment_data:
            self._add_sentiment_insights(insights, sentiment_data, keywords)

        # Insight 3: Common phrases
        if phrases and len(phrases) > 0:
            top_phrase, count = phrases[0]
            if count >= max(2, cluster_size * 0.3):  # At least 30% or 2 mentions
                insights.append(
                    f"Most common phrase: '{top_phrase}' ({count} mentions)"
                )

        # Insight 4: Keyword frequency
        if keywords:
            keyword_mentions = sum(
                1 for s in cluster_sentences
                if keywords[0] in s.lower()
            )
            if keyword_mentions / cluster_size > 0.5:
                insights.append(
                    f"{keyword_mentions}/{cluster_size} sentences explicitly mention '{keywords[0]}'"
                )

        # Insight 5: Cluster size
        if cluster_size >= 20:
            insights.append(
                f"Significant volume: {cluster_size} feedback items in this category"
            )

        # Return top 3 insights (or all if fewer)
        return insights[:3] if len(insights) > 3 else insights

    def _add_sentiment_insights(
        self,
        insights: List[str],
        sentiment_data: Dict,
        keywords: List[str]
    ) -> None:
        """
        Add sentiment-based insights

        Modifies insights list in place.

        Args:
            insights: Insights list to modify
            sentiment_data: Sentiment analysis results
            keywords: Extracted keywords
        """
        overall = sentiment_data.get('overall', 'neutral')
        percentages = sentiment_data.get('percentages', {})
        neg_pct = percentages.get('negative', 0)
        pos_pct = percentages.get('positive', 0)

        keyword = keywords[0] if keywords else 'this topic'

        # Strongly negative
        if overall == 'negative' and neg_pct >= 70:
            insights.append(
                f"{neg_pct:.0f}% express negative sentiment about {keyword} - requires attention"
            )

        # Strongly positive
        elif overall == 'positive' and pos_pct >= 70:
            insights.append(
                f"Overwhelmingly positive ({pos_pct:.0f}%) - key strength area"
            )

        # Mixed but leaning negative
        elif neg_pct >= 50:
            insights.append(
                f"Majority ({neg_pct:.0f}%) are dissatisfied with {keyword}"
            )

        # Mixed but leaning positive
        elif pos_pct >= 50:
            insights.append(
                f"Generally positive feedback ({pos_pct:.0f}%)"
            )

        # Add intensity insight if available
        stats = sentiment_data.get('statistics', {})
        avg_score = sentiment_data.get('average_score', 0)

        if abs(avg_score) >= 0.5:
            intensity = "strong" if abs(avg_score) >= 0.7 else "moderate"
            sentiment_word = "positive" if avg_score > 0 else "negative"
            insights.append(
                f"{intensity.capitalize()} {sentiment_word} sentiment "
                f"(avg score: {avg_score:.2f})"
            )

    def _empty_insights(self) -> Dict:
        """Fallback for empty clusters"""
        return {
            'title': 'Empty Cluster',
            'keywords': [],
            'key_insights': [],
            'common_phrases': [],
            'size': 0
        }


def generate_cluster_insights(
    cluster_sentences: List[str],
    sentiment_data: Optional[Dict] = None,
    total_sentences: Optional[int] = None,
    max_keywords: int = 10
) -> Dict:
    """
    Quick function to generate insights for a cluster

    Args:
        cluster_sentences: List of sentences in cluster
        sentiment_data: Optional sentiment analysis results
        total_sentences: Total sentences in dataset
        max_keywords: Maximum keywords to extract

    Returns:
        Insights dictionary
    """
    generator = ClusterInsightsGenerator(max_keywords=max_keywords)
    return generator.generate_insights(
        cluster_sentences=cluster_sentences,
        sentiment_data=sentiment_data,
        total_sentences=total_sentences
    )


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Cluster Insights Generation Test")
    print("=" * 60)

    # Test 1: Money/Withdrawal cluster
    print("\nTest 1: Negative Cluster (Money Issues)")
    print("-" * 60)

    money_sentences = [
        "Withholding my money",
        "Have lost so much money",
        "Holding my money hostage",
        "I want my money back",
        "Money lost in that period is your fault",
        "Give me back my money",
        "Cannot withdraw money",
        "Locked up my money",
        "They won't release my money"
    ]

    # Mock sentiment data
    money_sentiment = {
        'overall': 'negative',
        'average_score': -0.72,
        'percentages': {
            'positive': 0.0,
            'neutral': 11.1,
            'negative': 88.9
        },
        'statistics': {
            'median': -0.75,
            'std': 0.15
        }
    }

    generator = ClusterInsightsGenerator()
    insights = generator.generate_insights(
        cluster_sentences=money_sentences,
        sentiment_data=money_sentiment,
        total_sentences=100  # Assume 100 total sentences
    )

    print(f"Title: {insights['title']}")
    print(f"Size: {insights['size']}")
    print(f"\nKeywords: {', '.join(insights['keywords'][:5])}")
    print(f"\nCommon Phrases:")
    for phrase, count in insights['common_phrases']:
        print(f"  - '{phrase}': {count} times")
    print(f"\nKey Insights:")
    for i, insight in enumerate(insights['key_insights'], 1):
        print(f"  {i}. {insight}")

    # Test 2: Positive cluster
    print("\n" + "=" * 60)
    print("Test 2: Positive Cluster (Investment App)")
    print("-" * 60)

    investment_sentences = [
        "Best Investment App",
        "Best investment app",
        "Bar none the best investment APP in the industry",
        "Love Robinhood above all other investment apps",
        "Excellent Investment App, very informative",
        "Great platform to trade and make money",
        "Easy investing",
        "Robinhood makes investing easy",
        "Best UX for any investing app"
    ]

    investment_sentiment = {
        'overall': 'positive',
        'average_score': 0.78,
        'percentages': {
            'positive': 100.0,
            'neutral': 0.0,
            'negative': 0.0
        },
        'statistics': {
            'median': 0.80,
            'std': 0.12
        }
    }

    insights2 = generator.generate_insights(
        cluster_sentences=investment_sentences,
        sentiment_data=investment_sentiment,
        total_sentences=100
    )

    print(f"Title: {insights2['title']}")
    print(f"Size: {insights2['size']}")
    print(f"\nKeywords: {', '.join(insights2['keywords'][:5])}")
    print(f"\nKey Insights:")
    for i, insight in enumerate(insights2['key_insights'], 1):
        print(f"  {i}. {insight}")

    # Test 3: Small cluster (edge case)
    print("\n" + "=" * 60)
    print("Test 3: Small Cluster (2 sentences)")
    print("-" * 60)

    small_cluster = [
        "Food was terrible",
        "The food was not good"
    ]

    insights3 = generator.generate_insights(
        cluster_sentences=small_cluster,
        total_sentences=100
    )

    print(f"Title: {insights3['title']}")
    print(f"Keywords: {insights3['keywords']}")
    print(f"Insights: {insights3['key_insights']}")
