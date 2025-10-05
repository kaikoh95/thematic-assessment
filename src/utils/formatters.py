"""
Response Formatting Module

Transforms raw analysis results into Pydantic-validated API responses.

Responsibilities:
- Combine clustering, sentiment, and insights into structured output
- Map cluster labels to sentence groups
- Generate cluster IDs and metadata
- Create comparison insights when applicable
- Build final AnalysisResponse object

Strategy: Pure data transformation, no ML logic.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from src.utils.validators import (
    AnalysisResponse,
    AnalysisSummary,
    ClusterOutput,
    ClusterSentiment,
    SentenceOutput,
    SentimentData,
    ComparisonInsights,
    SentenceInput
)

logger = logging.getLogger(__name__)


class AnalysisFormatter:
    """
    Format raw analysis results into API response structure

    Pipeline:
    1. Group sentences by cluster label
    2. For each cluster: combine sentiment + insights + sentences
    3. Generate comparison insights if applicable
    4. Create summary statistics
    5. Build final AnalysisResponse

    Usage:
        formatter = AnalysisFormatter()
        response = formatter.format_response(
            sentences=sentences,
            cluster_labels=labels,
            sentiment_results=sentiment,
            insights_results=insights,
            request_data=request
        )
    """

    def __init__(self):
        """Initialize formatter"""
        pass

    def format_response(
        self,
        baseline_sentences: List[SentenceInput],
        cluster_labels: np.ndarray,
        cluster_sentiment_results: Dict[int, Dict],
        cluster_insights_results: Dict[int, Dict],
        sentence_level_sentiments: Dict[str, Dict],
        query: str,
        theme: Optional[str] = None,
        comparison_sentences: Optional[List[SentenceInput]] = None,
        comparison_labels: Optional[np.ndarray] = None
    ) -> AnalysisResponse:
        """
        Format complete analysis results into API response

        Args:
            baseline_sentences: Baseline sentence inputs
            cluster_labels: Cluster assignments for baseline
            cluster_sentiment_results: Sentiment per cluster {cluster_id: sentiment_dict}
            cluster_insights_results: Insights per cluster {cluster_id: insights_dict}
            sentence_level_sentiments: Sentiment per sentence {sentence_id: sentiment_dict}
            query: Analysis query/context
            theme: Optional theme
            comparison_sentences: Optional comparison sentences
            comparison_labels: Optional comparison cluster labels

        Returns:
            Validated AnalysisResponse
        """
        # Build baseline clusters
        baseline_clusters = self._build_clusters(
            sentences=baseline_sentences,
            labels=cluster_labels,
            sentiment_results=cluster_sentiment_results,
            insights_results=cluster_insights_results,
            sentence_sentiments=sentence_level_sentiments,
            source='baseline'
        )

        # Build comparison clusters if provided
        comparison_clusters = []
        if comparison_sentences is not None and comparison_labels is not None:
            comparison_clusters = self._build_clusters(
                sentences=comparison_sentences,
                labels=comparison_labels,
                sentiment_results=cluster_sentiment_results,
                insights_results=cluster_insights_results,
                sentence_sentiments=sentence_level_sentiments,
                source='comparison'
            )

        # Combine all clusters
        all_clusters = baseline_clusters + comparison_clusters

        # Sort by size (largest first)
        all_clusters.sort(key=lambda c: c.size, reverse=True)

        # Calculate overall statistics
        total_sentences = len(baseline_sentences) + (
            len(comparison_sentences) if comparison_sentences else 0
        )
        clusters_found = len(all_clusters)

        # Count unclustered (noise points marked as -1)
        unclustered_baseline = int(list(cluster_labels).count(-1))
        unclustered_comparison = int(list(comparison_labels).count(-1)) if comparison_labels is not None else 0
        unclustered = unclustered_baseline + unclustered_comparison

        # Overall sentiment (weighted by cluster size)
        overall_sentiment = self._calculate_overall_sentiment(all_clusters)

        # Build summary
        summary = AnalysisSummary(
            total_sentences=total_sentences,
            clusters_found=clusters_found,
            unclustered=unclustered,
            overall_sentiment=overall_sentiment,
            query=query,
            theme=theme
        )

        # Build comparison insights if applicable
        comparison_insights = None
        if comparison_clusters:
            comparison_insights = self._build_comparison_insights(
                baseline_clusters=baseline_clusters,
                comparison_clusters=comparison_clusters
            )

        return AnalysisResponse(
            clusters=all_clusters,
            summary=summary,
            comparison_insights=comparison_insights
        )

    def _build_clusters(
        self,
        sentences: List[SentenceInput],
        labels: np.ndarray,
        sentiment_results: Dict[int, Dict],
        insights_results: Dict[int, Dict],
        sentence_sentiments: Dict[str, Dict],
        source: str
    ) -> List[ClusterOutput]:
        """
        Build cluster objects from raw results

        Args:
            sentences: Input sentences
            labels: Cluster assignments
            sentiment_results: Sentiment per cluster
            insights_results: Insights per cluster
            sentence_sentiments: Sentiment per sentence
            source: 'baseline' or 'comparison'

        Returns:
            List of ClusterOutput objects
        """
        # Group sentences by cluster
        cluster_map = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Skip noise points
                cluster_map[int(label)].append(sentences[idx])

        clusters = []

        for cluster_id, cluster_sentences in cluster_map.items():
            # Get sentiment for this cluster
            cluster_sentiment_data = sentiment_results.get(cluster_id, {})

            # Get insights for this cluster
            cluster_insights_data = insights_results.get(cluster_id, {})

            # Build sentence outputs
            sentence_outputs = []
            for sent_input in cluster_sentences:
                sent_sentiment = sentence_sentiments.get(sent_input.id, {})

                sentence_outputs.append(
                    SentenceOutput(
                        sentence=sent_input.sentence,
                        id=sent_input.id,
                        sentiment=SentimentData(
                            label=sent_sentiment.get('label', 'neutral'),
                            score=sent_sentiment.get('score', 0.0)
                        )
                    )
                )

            # Build cluster sentiment
            distribution = cluster_sentiment_data.get('distribution', {
                'positive': 0, 'neutral': 0, 'negative': 0
            })

            cluster_sentiment = ClusterSentiment(
                overall=cluster_sentiment_data.get('overall', 'neutral'),
                distribution=distribution,
                average_score=cluster_sentiment_data.get('average_score', 0.0)
            )

            # Create cluster output
            cluster = ClusterOutput(
                id=f"{source}-cluster-{cluster_id}",
                title=cluster_insights_data.get('title', f'Cluster {cluster_id}'),
                sentences=sentence_outputs,
                size=len(cluster_sentences),
                sentiment=cluster_sentiment,
                key_insights=cluster_insights_data.get('key_insights', []),
                keywords=cluster_insights_data.get('keywords', []),
                source=source
            )

            clusters.append(cluster)

        logger.info(f"Built {len(clusters)} clusters from {source} data")
        return clusters

    def _calculate_overall_sentiment(self, clusters: List[ClusterOutput]) -> str:
        """
        Calculate overall sentiment across all clusters (weighted by size)

        Args:
            clusters: List of cluster outputs

        Returns:
            'positive', 'neutral', or 'negative'
        """
        if not clusters:
            return 'neutral'

        # Weight by cluster size
        weighted_scores = []
        for cluster in clusters:
            score = cluster.sentiment.average_score
            size = cluster.size
            weighted_scores.extend([score] * size)

        if not weighted_scores:
            return 'neutral'

        # Calculate median (robust to outliers)
        import statistics
        median_score = statistics.median(weighted_scores)

        # Classify using standard thresholds
        if median_score >= 0.05:
            return 'positive'
        elif median_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def _build_comparison_insights(
        self,
        baseline_clusters: List[ClusterOutput],
        comparison_clusters: List[ClusterOutput]
    ) -> ComparisonInsights:
        """
        Generate insights comparing baseline vs comparison datasets

        Identifies:
        - Themes only in baseline
        - Themes only in comparison
        - Shared themes

        Args:
            baseline_clusters: Baseline cluster outputs
            comparison_clusters: Comparison cluster outputs

        Returns:
            ComparisonInsights object
        """
        # Extract theme titles (cluster titles represent themes)
        baseline_themes = {c.title for c in baseline_clusters}
        comparison_themes = {c.title for c in comparison_clusters}

        # Find unique and shared themes
        baseline_only = list(baseline_themes - comparison_themes)
        comparison_only = list(comparison_themes - baseline_themes)
        shared = list(baseline_themes & comparison_themes)

        # Sort by frequency (if multiple clusters have similar names, prioritize)
        baseline_only.sort()
        comparison_only.sort()
        shared.sort()

        logger.info(
            f"Comparison: {len(baseline_only)} baseline-only, "
            f"{len(comparison_only)} comparison-only, "
            f"{len(shared)} shared themes"
        )

        return ComparisonInsights(
            baseline_only_themes=baseline_only,
            comparison_only_themes=comparison_only,
            shared_themes=shared
        )


def format_error_response(
    error_message: str,
    status_code: int = 400,
    request_id: Optional[str] = None
) -> Dict:
    """
    Format error response for API Gateway

    Args:
        error_message: Human-readable error description
        status_code: HTTP status code
        request_id: Optional AWS request ID

    Returns:
        API Gateway response dict
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST,OPTIONS'
        },
        'body': json.dumps({
            'error': error_message,
            'request_id': request_id
        })
    }


def format_success_response(
    response: AnalysisResponse,
    request_id: Optional[str] = None
) -> Dict:
    """
    Format success response for API Gateway

    Args:
        response: Validated AnalysisResponse
        request_id: Optional AWS request ID

    Returns:
        API Gateway response dict
    """
    # Convert Pydantic model to dict
    response_dict = response.model_dump()

    # Add request metadata
    if request_id:
        response_dict['request_id'] = request_id

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST,OPTIONS'
        },
        'body': json.dumps(response_dict)
    }


if __name__ == "__main__":
    # Example usage and testing
    import json
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Response Formatting Test")
    print("=" * 60)

    # Mock data
    baseline_sentences = [
        SentenceInput(sentence="I want my money back", id="1"),
        SentenceInput(sentence="Cannot withdraw money", id="2"),
        SentenceInput(sentence="Best investment app", id="3"),
        SentenceInput(sentence="Great platform", id="4"),
    ]

    # Mock cluster labels (cluster 0 and 1)
    cluster_labels = np.array([0, 0, 1, 1])

    # Mock sentiment results
    cluster_sentiment_results = {
        0: {
            'overall': 'negative',
            'average_score': -0.72,
            'distribution': {'positive': 0, 'neutral': 0, 'negative': 2}
        },
        1: {
            'overall': 'positive',
            'average_score': 0.65,
            'distribution': {'positive': 2, 'neutral': 0, 'negative': 0}
        }
    }

    # Mock insights results
    cluster_insights_results = {
        0: {
            'title': 'Money & Withdrawal Issues',
            'keywords': ['money', 'withdraw', 'back'],
            'key_insights': [
                '100% express negative sentiment - requires attention',
                'Most common phrase: "my money" (2 mentions)'
            ]
        },
        1: {
            'title': 'Investment App & Platform',
            'keywords': ['investment', 'app', 'platform', 'best'],
            'key_insights': [
                'Overwhelmingly positive (100%) - key strength area'
            ]
        }
    }

    # Mock sentence-level sentiments
    sentence_sentiments = {
        "1": {'label': 'negative', 'score': -0.75},
        "2": {'label': 'negative', 'score': -0.69},
        "3": {'label': 'positive', 'score': 0.68},
        "4": {'label': 'positive', 'score': 0.62}
    }

    # Test formatting
    formatter = AnalysisFormatter()

    response = formatter.format_response(
        baseline_sentences=baseline_sentences,
        cluster_labels=cluster_labels,
        cluster_sentiment_results=cluster_sentiment_results,
        cluster_insights_results=cluster_insights_results,
        sentence_level_sentiments=sentence_sentiments,
        query="overview",
        theme="product feedback"
    )

    print("\n" + "=" * 60)
    print("Formatted Response")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Total sentences: {response.summary.total_sentences}")
    print(f"  Clusters found: {response.summary.clusters_found}")
    print(f"  Overall sentiment: {response.summary.overall_sentiment}")
    print(f"  Query: {response.summary.query}")
    print(f"  Theme: {response.summary.theme}")

    print(f"\nClusters ({len(response.clusters)}):")
    for cluster in response.clusters:
        print(f"\n  {cluster.id}: {cluster.title}")
        print(f"    Size: {cluster.size}")
        print(f"    Sentiment: {cluster.sentiment.overall} (avg: {cluster.sentiment.average_score:.2f})")
        print(f"    Keywords: {', '.join(cluster.keywords[:5])}")
        print(f"    Insights:")
        for insight in cluster.key_insights:
            print(f"      - {insight}")

    # Test API Gateway response formatting
    print("\n" + "=" * 60)
    print("API Gateway Response Format")
    print("=" * 60)

    api_response = format_success_response(response, request_id="test-123")
    print(f"\nStatus Code: {api_response['statusCode']}")
    print(f"Headers: {json.dumps(api_response['headers'], indent=2)}")
    print(f"Body keys: {list(api_response['body'].keys())}")
    print(f"Body clusters: {len(api_response['body']['clusters'])}")

    # Test error response
    print("\n" + "=" * 60)
    print("Error Response Format")
    print("=" * 60)

    error_response = format_error_response(
        error_message="Invalid request: missing baseline field",
        status_code=400,
        request_id="test-456"
    )

    print(f"\nStatus Code: {error_response['statusCode']}")
    print(f"Error: {error_response['body']['error']}")
    print(f"Request ID: {error_response['body']['request_id']}")

    print("\n✓ Formatter test completed successfully")
