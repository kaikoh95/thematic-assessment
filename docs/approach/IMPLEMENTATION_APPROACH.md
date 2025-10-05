# Implementation Approach

## Core Principles

1. **Simplicity over sophistication** - 4-hour constraint requires focus on working MVP
2. **No external dependencies** - No LLM APIs, use local models/rule-based logic
3. **Deterministic outputs** - Reproducible results for same input
4. **Production-ready patterns** - Even if MVP, follow best practices
5. **Clear trade-offs** - Document what we're NOT doing and why

## Problem Breakdown

### Challenge 1: Semantic Clustering

**Goal**: Group similar sentences into meaningful clusters

**Approach**:
1. **Embedding Generation**
   - Use `sentence-transformers` with `all-MiniLM-L6-v2` model
   - Converts sentences to 384-dimensional vectors
   - Semantic similarity preserved in vector space

   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode(sentences, show_progress_bar=False)
   ```

2. **Clustering Algorithm**
   - **Primary**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)
     - Pros: Handles varying cluster densities, auto-detects cluster count, identifies outliers
     - Cons: Slightly slower, more parameters to tune

   - **Fallback**: KMeans
     - Pros: Fast, predictable
     - Cons: Must specify K, struggles with uneven clusters

   ```python
   import hdbscan

   # Adaptive min_cluster_size based on dataset
   min_size = max(3, len(sentences) // 20)

   clusterer = hdbscan.HDBSCAN(
       min_cluster_size=min_size,
       min_samples=2,
       metric='euclidean',
       cluster_selection_method='eom'
   )
   labels = clusterer.fit_predict(embeddings)
   ```

3. **Post-Processing**
   - Filter out noise (label = -1)
   - Merge very small clusters (<3 items) into "Other"
   - If >10 clusters, merge smallest clusters
   - Goal: 3-10 meaningful clusters

**Why No LLM?**
- ❌ Adds 1-3s latency per API call
- ❌ External dependency (rate limits, API keys, costs)
- ❌ Non-deterministic outputs
- ✅ Sentence transformers + HDBSCAN is proven for this task

### Challenge 2: Sentiment Analysis

**Goal**: Classify sentiment and provide confidence scores

**Approach**: VADER (Valence Aware Dictionary and sEntiment Reasoner)

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    # Returns: {'neg': 0.1, 'neu': 0.5, 'pos': 0.4, 'compound': 0.6}

    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'label': label,
        'score': scores['compound'],
        'confidence': max(scores['pos'], scores['neg'], scores['neu'])
    }
```

**Why VADER?**
- ✅ Designed for social media/short text
- ✅ Handles emoticons, slang, intensifiers
- ✅ Fast (no ML inference)
- ✅ No training required
- ✅ Performs well on app reviews/feedback

**Alternative**: TextBlob
- Simpler API but less accurate for modern text

### Challenge 3: Cluster Title Generation

**Goal**: Generate concise, meaningful titles for each cluster

**Approach**: Keyword Extraction + Template

1. **Extract Keywords** using TF-IDF
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   def get_cluster_keywords(cluster_sentences, n_keywords=5):
       vectorizer = TfidfVectorizer(
           max_features=20,
           stop_words='english',
           ngram_range=(1, 2)  # Include bigrams
       )
       tfidf_matrix = vectorizer.fit_transform(cluster_sentences)

       # Sum TF-IDF scores across all docs in cluster
       feature_scores = tfidf_matrix.sum(axis=0).A1
       feature_names = vectorizer.get_feature_names_out()

       # Sort by score
       top_indices = feature_scores.argsort()[-n_keywords:][::-1]
       keywords = [feature_names[i] for i in top_indices]

       return keywords
   ```

2. **Generate Title**
   ```python
   def generate_title(keywords, cluster_sentences):
       # Option 1: Join top 3 keywords
       title = " + ".join(keywords[:3]).title()

       # Option 2: Template-based
       if len(cluster_sentences) < 5:
           title = f"Minor Theme: {keywords[0].title()}"
       else:
           title = f"{keywords[0].title()} & {keywords[1].title()}"

       # Fallback: Use most common words
       if not keywords:
           title = "Cluster " + str(cluster_id)

       return title
   ```

**Why No LLM?**
- TF-IDF reliably identifies important terms
- Templates ensure consistent formatting
- Much faster than API calls
- Good enough for MVP

### Challenge 4: Key Insights Generation

**Goal**: Provide 2-3 actionable insights per cluster

**Approach**: Statistical Analysis + Templates

```python
def generate_insights(cluster_data):
    sentences = cluster_data['sentences']
    keywords = cluster_data['keywords']
    sentiment = cluster_data['sentiment']

    insights = []

    # Insight 1: Prevalence
    total_sentences = cluster_data['total_in_dataset']
    percentage = (len(sentences) / total_sentences) * 100
    insights.append(
        f"{percentage:.0f}% of feedback relates to {keywords[0]}"
    )

    # Insight 2: Sentiment
    neg_pct = (sentiment['distribution']['negative'] / len(sentences)) * 100
    if neg_pct > 70:
        insights.append(
            f"{neg_pct:.0f}% of mentions are negative - requires attention"
        )
    elif sentiment['distribution']['positive'] / len(sentences) > 0.7:
        insights.append(
            "Overwhelmingly positive sentiment - key strength"
        )

    # Insight 3: Specific patterns
    common_phrases = find_common_phrases(sentences)
    if common_phrases:
        insights.append(
            f"Most common phrase: '{common_phrases[0]}'"
        )

    # Insight 4: Keyword frequency
    keyword_count = sum(1 for s in sentences if keywords[0] in s.lower())
    if keyword_count / len(sentences) > 0.5:
        insights.append(
            f"{keyword_count}/{len(sentences)} mentions explicitly reference '{keywords[0]}'"
        )

    return insights[:3]  # Return top 3
```

**Insight Templates**:
- Prevalence: "X% of feedback mentions [keyword]"
- Sentiment: "Y% express negative sentiment about [topic]"
- Comparison: "This theme appears Z% more in baseline vs comparison"
- Frequency: "Most common issue: [specific phrase]"
- Intensity: "Strong sentiment (avg score: X)"

### Challenge 5: Comparative Analysis

**Goal**: Identify differences between baseline and comparison datasets

**Approach**: Joint Clustering + Source Tracking

```python
def comparative_clustering(baseline, comparison):
    # 1. Combine datasets with source tags
    all_sentences = []
    source_map = {}

    for item in baseline:
        all_sentences.append(item['sentence'])
        source_map[item['id']] = 'baseline'

    for item in comparison:
        all_sentences.append(item['sentence'])
        source_map[item['id']] = 'comparison'

    # 2. Cluster all together
    embeddings = model.encode(all_sentences)
    labels = clusterer.fit_predict(embeddings)

    # 3. Analyze cluster composition
    clusters_analysis = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        cluster_items = [
            all_sentences[i] for i, label in enumerate(labels)
            if label == cluster_id
        ]

        baseline_count = sum(
            1 for item_id, source in source_map.items()
            if source == 'baseline' and item_id in cluster_items
        )
        comparison_count = len(cluster_items) - baseline_count

        clusters_analysis[cluster_id] = {
            'baseline': baseline_count,
            'comparison': comparison_count,
            'ratio': baseline_count / max(comparison_count, 1)
        }

    # 4. Categorize clusters
    baseline_only = [
        cid for cid, data in clusters_analysis.items()
        if data['comparison'] == 0
    ]
    comparison_only = [
        cid for cid, data in clusters_analysis.items()
        if data['baseline'] == 0
    ]
    shared = [
        cid for cid in clusters_analysis
        if cid not in baseline_only and cid not in comparison_only
    ]

    return {
        'baseline_only_themes': [get_cluster_title(cid) for cid in baseline_only],
        'comparison_only_themes': [get_cluster_title(cid) for cid in comparison_only],
        'shared_themes': [get_cluster_title(cid) for cid in shared],
        'cluster_details': clusters_analysis
    }
```

**Comparison Insights**:
- "Theme X appears only in baseline (suggests issue resolved)"
- "Theme Y appears only in comparison (new concern)"
- "Theme Z present in both: baseline 60%, comparison 40% (improving)"

## Algorithm Parameter Tuning

### HDBSCAN Parameters

```python
def get_hdbscan_params(n_sentences):
    """Adaptive parameters based on dataset size"""

    if n_sentences < 50:
        # Small dataset: be conservative
        return {
            'min_cluster_size': 3,
            'min_samples': 2,
            'cluster_selection_epsilon': 0.0
        }
    elif n_sentences < 200:
        # Medium dataset
        return {
            'min_cluster_size': max(5, n_sentences // 30),
            'min_samples': 3,
            'cluster_selection_epsilon': 0.1
        }
    else:
        # Large dataset: allow smaller clusters as % of total
        return {
            'min_cluster_size': max(10, n_sentences // 40),
            'min_samples': 5,
            'cluster_selection_epsilon': 0.2
        }
```

### Fallback Logic

```python
def cluster_with_fallback(embeddings, min_clusters=3, max_clusters=10):
    try:
        # Try HDBSCAN first
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Check if results are reasonable
        if n_clusters < min_clusters or n_clusters > max_clusters * 2:
            raise ValueError("Poor clustering quality")

        return labels, 'hdbscan'

    except Exception as e:
        logger.warning(f"HDBSCAN failed: {e}, falling back to KMeans")

        # Fallback to KMeans
        from sklearn.cluster import KMeans

        # Use elbow method to estimate K
        k = estimate_optimal_k(embeddings, max_clusters)

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        return labels, 'kmeans'
```

## Data Flow Pipeline

```python
class TextAnalysisPipeline:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze(self, baseline, comparison=None, query=None, theme=None):
        # Step 1: Prepare data
        all_data = self._prepare_data(baseline, comparison)

        # Step 2: Generate embeddings
        embeddings = self.model.encode(
            [item['sentence'] for item in all_data],
            show_progress_bar=False,
            batch_size=32
        )

        # Step 3: Cluster
        labels = self._cluster(embeddings, len(all_data))

        # Step 4: Build clusters
        clusters = self._build_clusters(all_data, labels)

        # Step 5: Analyze each cluster
        for cluster in clusters:
            cluster['sentiment'] = self._analyze_cluster_sentiment(cluster)
            cluster['keywords'] = self._extract_keywords(cluster)
            cluster['title'] = self._generate_title(cluster)
            cluster['insights'] = self._generate_insights(cluster)

        # Step 6: Comparative analysis (if comparison exists)
        comparison_insights = None
        if comparison:
            comparison_insights = self._comparative_analysis(clusters, all_data)

        # Step 7: Format response
        return self._format_response(clusters, comparison_insights, query, theme)
```

## Error Handling Strategy

### Input Validation
```python
from pydantic import BaseModel, Field, validator

class SentenceInput(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=1000)
    id: str = Field(..., min_length=1)

class AnalysisRequest(BaseModel):
    baseline: list[SentenceInput] = Field(..., min_items=1, max_items=1000)
    comparison: Optional[list[SentenceInput]] = Field(default=None, max_items=1000)
    query: Optional[str] = Field(default="overview")
    surveyTitle: Optional[str] = None
    theme: Optional[str] = None

    @validator('baseline', 'comparison')
    def check_duplicates(cls, v):
        if v:
            ids = [item.id for item in v]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs found")
        return v
```

### Runtime Error Handling
```python
def safe_cluster(embeddings):
    try:
        return cluster_with_fallback(embeddings)
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        # Return single cluster as fallback
        return np.zeros(len(embeddings)), 'fallback'

def safe_sentiment(text):
    try:
        return analyze_sentiment(text)
    except:
        return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
```

## Performance Optimizations

### 1. Model Caching (Lambda Global Scope)
```python
# Outside handler - persists across warm starts
MODEL = None
SENTIMENT_ANALYZER = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return MODEL
```

### 2. Batch Processing
```python
# Good: Single batch encoding
embeddings = model.encode(all_sentences)

# Bad: Individual encoding
embeddings = [model.encode(s) for s in sentences]  # Much slower!
```

### 3. Lazy Imports
```python
def lambda_handler(event, context):
    # Import heavy libraries only when invoked
    import numpy as np
    import hdbscan
    from sentence_transformers import SentenceTransformer

    # ... rest of handler
```

## Testing Strategy

### Unit Tests
- Embedding generation
- Clustering logic
- Sentiment analysis
- Keyword extraction
- Title generation
- Insight templates

### Integration Tests
- Full pipeline with sample data
- Edge cases (1 sentence, 1000 sentences)
- Comparison mode
- Error scenarios

### Performance Tests
- Measure latency at different input sizes
- Memory usage profiling
- Cold start timing

## Trade-offs & Limitations

### What We're NOT Doing (and Why)

| Feature | Why Not in MVP | Future Consideration |
|---------|----------------|----------------------|
| LLM Integration | Adds latency, cost, complexity | Yes - for richer insights |
| Caching | Adds infrastructure | Yes - Redis/DynamoDB |
| Fine-tuned embeddings | Time-consuming, domain-specific | Maybe - if accuracy insufficient |
| Real-time streaming | Complex for MVP | Maybe - for large datasets |
| Multi-language | Model doesn't support well | Yes - with mBERT/XLM |
| Custom stopwords | Generic is good enough | Yes - domain-specific tuning |
| A/B testing framework | Over-engineering | Maybe - if multiple algorithms |

### Known Limitations

1. **Cluster quality**: Dependent on input quality and diversity
2. **Insight depth**: Templates less sophisticated than LLM
3. **Scalability**: 1000 sentence limit for <10s latency
4. **Cold starts**: 2-4s with ML libraries
5. **No persistence**: Results not stored

## Success Criteria

### Functional
- ✅ Returns 3-10 meaningful clusters
- ✅ Accurate sentiment classification (>80%)
- ✅ Readable cluster titles
- ✅ Actionable insights
- ✅ Handles comparative analysis

### Non-Functional
- ✅ <10s response time for <500 sentences
- ✅ Deterministic outputs (same input = same output)
- ✅ Handles errors gracefully
- ✅ Deployable via single CDK command

### Quality
- ✅ Type-safe (Pydantic validation)
- ✅ Tested (>70% coverage)
- ✅ Documented (inline + external docs)
- ✅ Production patterns (error handling, logging)
