"""
AWS Lambda Handler - Text Analysis Microservice

Main entry point for API Gateway requests.
Orchestrates the complete analysis pipeline:
  Request → Validation → Embeddings → Clustering → Sentiment → Insights → Response

Performance targets:
- Cold start: <5s (with SnapStart on Python 3.12)
- Warm start: <3s for 100 sentences
- <10s for 500 sentences

Key optimizations:
- Global scope model caching (reused across warm invocations)
- Lazy initialization
- Batch processing
- ONNX-accelerated inference
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import traceback

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import only lightweight validation/formatting modules at module level
# Heavy ML imports (PyTorch, transformers) are deferred to get_cached_instances()
from src.utils.validators import (
    AnalysisRequest,
    validate_request,
    format_validation_errors,
    SentenceInput
)
from src.utils.formatters import (
    AnalysisFormatter,
    format_success_response,
    format_error_response
)

# Type hints only (not imported at runtime)
if TYPE_CHECKING:
    from src.clustering.embeddings import SentenceEmbedder
    from src.clustering.clusterer import TextClusterer
    from src.sentiment.analyzer import ClusterSentimentAnalyzer
    from src.clustering.insights import ClusterInsightsGenerator

# Global instances (cached across Lambda invocations)
_embedder: Optional['SentenceEmbedder'] = None
_clusterer: Optional['TextClusterer'] = None
_sentiment_analyzer: Optional['ClusterSentimentAnalyzer'] = None
_insights_generator: Optional['ClusterInsightsGenerator'] = None
_formatter: Optional[AnalysisFormatter] = None


def get_cached_instances():
    """
    Initialize and cache all ML components

    Critical for Lambda performance - models are loaded once
    and reused across warm invocations.

    Returns:
        Tuple of (embedder, clusterer, sentiment_analyzer, insights_generator, formatter)
    """
    global _embedder, _clusterer, _sentiment_analyzer, _insights_generator, _formatter

    if _embedder is None:
        logger.info("Initializing ML components (cold start)...")
        start = time.time()

        # Lazy load ML modules to avoid Lambda init timeout (10s limit)
        # PyTorch + transformers import takes 10-15s, exceeds init phase
        # Invocation phase has 120s timeout - plenty of time for model loading
        from src.clustering.embeddings import SentenceEmbedder
        from src.clustering.clusterer import TextClusterer
        from src.sentiment.analyzer import ClusterSentimentAnalyzer
        from src.clustering.insights import ClusterInsightsGenerator

        _embedder = SentenceEmbedder(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            use_onnx=True
        )

        _clusterer = TextClusterer(
            umap_n_components=5,
            max_clusters=10,
            noise_threshold=0.5
        )

        _sentiment_analyzer = ClusterSentimentAnalyzer(threshold_mode='standard')

        _insights_generator = ClusterInsightsGenerator(max_keywords=10)

        _formatter = AnalysisFormatter()

        duration = time.time() - start
        logger.info(f"ML components initialized in {duration:.2f}s")
    else:
        logger.info("Using cached ML components (warm start)")

    return _embedder, _clusterer, _sentiment_analyzer, _insights_generator, _formatter


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for text analysis API

    Args:
        event: API Gateway event with request body
        context: Lambda context object

    Returns:
        API Gateway response dict with statusCode, headers, body

    Event structure:
        {
            "body": "{\"baseline\": [...], \"comparison\": [...], \"query\": \"overview\"}",
            "headers": {...},
            "requestContext": {...}
        }
    """
    # Start timing
    start_time = time.time()

    # Get request ID for tracing
    # Handle both AWS Lambda context (aws_request_id) and MockContext (request_id)
    request_id = getattr(context, 'aws_request_id', getattr(context, 'request_id', 'local-test')) if context else 'local-test'

    logger.info(f"Processing request {request_id}")

    try:
        # Step 1: Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)  # Direct invocation or test

        logger.info(f"Request body parsed: {len(json.dumps(body))} bytes")

        # Step 2: Validate request
        try:
            request = validate_request(body)
            logger.info(
                f"Request validated: {len(request.baseline)} baseline, "
                f"{len(request.comparison) if request.comparison else 0} comparison sentences"
            )
        except Exception as validation_error:
            logger.warning(f"Validation failed: {validation_error}")
            error_response = format_validation_errors(validation_error)
            return format_error_response(
                error_message=error_response.error,
                status_code=400,
                request_id=request_id
            )

        # Step 3: Initialize ML components (use cached instances)
        embedder, clusterer, sentiment_analyzer, insights_generator, formatter = get_cached_instances()

        # Step 4: Run analysis pipeline
        result = analyze_feedback(
            request=request,
            embedder=embedder,
            clusterer=clusterer,
            sentiment_analyzer=sentiment_analyzer,
            insights_generator=insights_generator,
            formatter=formatter
        )

        # Step 5: Format and return response
        duration = time.time() - start_time
        logger.info(
            f"Request {request_id} completed successfully in {duration:.2f}s "
            f"({result.summary.clusters_found} clusters, "
            f"{result.summary.total_sentences} sentences)"
        )

        return format_success_response(result, request_id=request_id)

    except Exception as e:
        # Catch-all error handler
        duration = time.time() - start_time
        logger.error(
            f"Request {request_id} failed after {duration:.2f}s: {e}\n"
            f"{traceback.format_exc()}"
        )

        return format_error_response(
            error_message=f"Internal server error: {str(e)}",
            status_code=500,
            request_id=request_id
        )


def analyze_feedback(
    request: AnalysisRequest,
    embedder: 'SentenceEmbedder',
    clusterer: 'TextClusterer',
    sentiment_analyzer: 'ClusterSentimentAnalyzer',
    insights_generator: 'ClusterInsightsGenerator',
    formatter: AnalysisFormatter
):
    """
    Complete feedback analysis pipeline

    Pipeline stages:
    1. Generate embeddings for baseline (and comparison if provided)
    2. Cluster embeddings
    3. Analyze sentiment per cluster
    4. Generate insights per cluster
    5. Format response

    Args:
        request: Validated AnalysisRequest
        embedder: SentenceEmbedder instance
        clusterer: TextClusterer instance
        sentiment_analyzer: ClusterSentimentAnalyzer instance
        insights_generator: ClusterInsightsGenerator instance
        formatter: AnalysisFormatter instance

    Returns:
        AnalysisResponse object
    """
    # ========================================================================
    # BASELINE ANALYSIS
    # ========================================================================

    logger.info("=" * 60)
    logger.info("BASELINE ANALYSIS")
    logger.info("=" * 60)

    # Extract baseline sentences
    baseline_sentences = [s.sentence for s in request.baseline]
    baseline_ids = [s.id for s in request.baseline]

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(baseline_sentences)} baseline sentences...")
    baseline_embeddings = embedder.encode(baseline_sentences, batch_size=32)

    # Cluster
    logger.info("Clustering baseline embeddings...")
    baseline_cluster_result = clusterer.cluster(baseline_embeddings, baseline_sentences)

    # Reassign noise points if needed
    baseline_labels = baseline_cluster_result['labels']
    if baseline_cluster_result['noise_count'] > 0:
        logger.info(f"Reassigning {baseline_cluster_result['noise_count']} noise points...")
        baseline_labels = clusterer.assign_noise_to_nearest_cluster(
            baseline_labels,
            baseline_cluster_result['reduced_embeddings']
        )

    # Analyze sentiment and insights per cluster
    baseline_cluster_sentiments = {}
    baseline_cluster_insights = {}
    baseline_sentence_sentiments = {}

    unique_clusters = set(baseline_labels[baseline_labels != -1])
    logger.info(f"Analyzing {len(unique_clusters)} baseline clusters...")

    for cluster_id in unique_clusters:
        cluster_mask = baseline_labels == cluster_id
        cluster_sentences = [baseline_sentences[i] for i, m in enumerate(cluster_mask) if m]
        cluster_ids = [baseline_ids[i] for i, m in enumerate(cluster_mask) if m]

        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_cluster(cluster_sentences, cluster_ids)
        baseline_cluster_sentiments[cluster_id] = sentiment_result

        # Insights generation
        insights_result = insights_generator.generate_insights(
            cluster_sentences=cluster_sentences,
            sentiment_data=sentiment_result,
            total_sentences=len(baseline_sentences)
        )
        baseline_cluster_insights[cluster_id] = insights_result

    # Analyze sentence-level sentiment for all baseline sentences
    for i, sentence in enumerate(baseline_sentences):
        sentiment_scores = sentiment_analyzer.analyze_single(sentence)
        from src.sentiment.analyzer import classify_sentiment
        baseline_sentence_sentiments[baseline_ids[i]] = {
            'label': classify_sentiment(sentiment_scores['compound']),
            'score': sentiment_scores['compound']
        }

    # ========================================================================
    # COMPARISON ANALYSIS (if provided)
    # ========================================================================

    comparison_labels = None
    comparison_cluster_sentiments = {}
    comparison_cluster_insights = {}

    if request.comparison:
        logger.info("=" * 60)
        logger.info("COMPARISON ANALYSIS")
        logger.info("=" * 60)

        comparison_sentences_text = [s.sentence for s in request.comparison]
        comparison_ids = [s.id for s in request.comparison]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(comparison_sentences_text)} comparison sentences...")
        comparison_embeddings = embedder.encode(comparison_sentences_text, batch_size=32)

        # Cluster
        logger.info("Clustering comparison embeddings...")
        comparison_cluster_result = clusterer.cluster(comparison_embeddings, comparison_sentences_text)

        # Reassign noise
        comparison_labels = comparison_cluster_result['labels']
        if comparison_cluster_result['noise_count'] > 0:
            logger.info(f"Reassigning {comparison_cluster_result['noise_count']} noise points...")
            comparison_labels = clusterer.assign_noise_to_nearest_cluster(
                comparison_labels,
                comparison_cluster_result['reduced_embeddings']
            )

        # Analyze each comparison cluster
        unique_comp_clusters = set(comparison_labels[comparison_labels != -1])
        logger.info(f"Analyzing {len(unique_comp_clusters)} comparison clusters...")

        for cluster_id in unique_comp_clusters:
            cluster_mask = comparison_labels == cluster_id
            cluster_sentences = [comparison_sentences_text[i] for i, m in enumerate(cluster_mask) if m]
            cluster_ids_list = [comparison_ids[i] for i, m in enumerate(cluster_mask) if m]

            # Sentiment
            sentiment_result = sentiment_analyzer.analyze_cluster(cluster_sentences, cluster_ids_list)
            comparison_cluster_sentiments[cluster_id] = sentiment_result

            # Insights
            insights_result = insights_generator.generate_insights(
                cluster_sentences=cluster_sentences,
                sentiment_data=sentiment_result,
                total_sentences=len(comparison_sentences_text)
            )
            comparison_cluster_insights[cluster_id] = insights_result

        # Sentence-level sentiment for comparison
        for i, sentence in enumerate(comparison_sentences_text):
            sentiment_scores = sentiment_analyzer.analyze_single(sentence)
            from src.sentiment.analyzer import classify_sentiment
            baseline_sentence_sentiments[comparison_ids[i]] = {
                'label': classify_sentiment(sentiment_scores['compound']),
                'score': sentiment_scores['compound']
            }

    # ========================================================================
    # FORMAT RESPONSE
    # ========================================================================

    logger.info("=" * 60)
    logger.info("FORMATTING RESPONSE")
    logger.info("=" * 60)

    # Merge sentiment and insights dicts for formatter
    all_cluster_sentiments = {**baseline_cluster_sentiments, **comparison_cluster_sentiments}
    all_cluster_insights = {**baseline_cluster_insights, **comparison_cluster_insights}

    response = formatter.format_response(
        baseline_sentences=request.baseline,
        cluster_labels=baseline_labels,
        cluster_sentiment_results=all_cluster_sentiments,
        cluster_insights_results=all_cluster_insights,
        sentence_level_sentiments=baseline_sentence_sentiments,
        query=request.query,
        theme=request.theme,
        comparison_sentences=request.comparison,
        comparison_labels=comparison_labels
    )

    logger.info(f"Response formatted: {response.summary.clusters_found} clusters, {response.summary.total_sentences} sentences")

    return response


# For local testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Lambda Handler Local Test")
    print("=" * 60)

    # Mock event
    test_event = {
        "body": json.dumps({
            "baseline": [
                {"sentence": "I want my money back", "id": "1"},
                {"sentence": "Cannot withdraw money", "id": "2"},
                {"sentence": "Holding my money hostage", "id": "3"},
                {"sentence": "Best investment app", "id": "4"},
                {"sentence": "Great platform to trade", "id": "5"},
                {"sentence": "Love the app", "id": "6"},
                {"sentence": "Easy to use interface", "id": "7"},
                {"sentence": "Terrible customer service", "id": "8"},
                {"sentence": "Support is unhelpful", "id": "9"},
                {"sentence": "No response from support", "id": "10"}
            ],
            "query": "overview",
            "surveyTitle": "Product Feedback Q1",
            "theme": "customer experience"
        })
    }

    # Mock context
    class MockContext:
        request_id = "test-local-123"
        function_name = "text-analysis-dev"
        memory_limit_in_mb = 3008

    # Run handler
    start = time.time()
    result = handler(test_event, MockContext())
    duration = time.time() - start

    print("\n" + "=" * 60)
    print("Handler Result")
    print("=" * 60)

    print(f"\nStatus Code: {result['statusCode']}")
    print(f"Duration: {duration:.2f}s")

    if result['statusCode'] == 200:
        body = result['body']
        print(f"\nSummary:")
        print(f"  Total sentences: {body['summary']['total_sentences']}")
        print(f"  Clusters found: {body['summary']['clusters_found']}")
        print(f"  Overall sentiment: {body['summary']['overall_sentiment']}")

        print(f"\nClusters ({len(body['clusters'])}):")
        for cluster in body['clusters']:
            print(f"\n  {cluster['id']}: {cluster['title']}")
            print(f"    Size: {cluster['size']}")
            print(f"    Sentiment: {cluster['sentiment']['overall']}")
            print(f"    Keywords: {', '.join(cluster['keywords'][:5])}")
    else:
        print(f"\nError: {result['body']}")

    print("\n✓ Local test completed")
