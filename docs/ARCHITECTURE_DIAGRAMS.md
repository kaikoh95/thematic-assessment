# System Architecture Diagrams

## 1. High-Level Architecture

```mermaid
graph TB
    APIClient[API Client]

    subgraph AWS["AWS Cloud"]
        subgraph API["API Layer"]
            APIGW[API Gateway<br/>REST API]
        end

        subgraph Compute["Compute Layer"]
            Lambda[AWS Lambda<br/>Docker Container<br/>Python 3.12<br/>3008 MB]
        end

        subgraph Storage["Storage"]
            ECR[Amazon ECR<br/>Container Registry]
            CW[CloudWatch Logs<br/>7 day retention]
        end

        subgraph IaC["Infrastructure as Code"]
            CDK[AWS CDK<br/>Python Stack]
        end
    end

    subgraph CICD["CI/CD"]
        GH[GitHub Actions<br/>Push to main]
        DockerBuild[Docker Build<br/>x86_64]
    end

    APIClient -->|POST /analyze| APIGW
    APIGW -->|Invoke| Lambda
    Lambda -->|Logs| CW
    Lambda -->|Pull Image| ECR

    GH -->|1. Build| DockerBuild
    DockerBuild -->|2. Push| ECR
    GH -->|3. Deploy| CDK
    CDK -->|4. Update| Lambda

    style Lambda fill:#ff9900
    style APIGW fill:#ff9900
    style ECR fill:#ff9900
    style CW fill:#ff9900
```

## 2. ML Pipeline Architecture

```mermaid
graph LR
    subgraph "Input Processing"
        REQ[API Request<br/>JSON Body]
        VAL[Pydantic Validation<br/>• Check duplicates<br/>• Validate IDs<br/>• Parse fields]
    end

    subgraph "Embedding Generation"
        EMB[SentenceTransformer<br/>all-MiniLM-L6-v2<br/>384 dimensions]
    end

    subgraph "Clustering"
        UMAP[UMAP Reduction<br/>384 → 5 dims]
        HDBSCAN[HDBSCAN Clustering<br/>min_size=2<br/>max_clusters=10]
        NOISE[Noise Reassignment<br/>Distance-based]
    end

    subgraph "Analysis"
        SENT[VADER Sentiment<br/>Cluster-level<br/>Sentence-level]
        KW[TF-IDF Keywords<br/>Top 10 per cluster]
        INS[Insights Generator<br/>Stats + Patterns]
    end

    subgraph "Output"
        FMT[Response Formatter<br/>JSON Structure]
        RESP[API Response<br/>200/400/500]
    end

    REQ --> VAL
    VAL -->|Valid| EMB
    VAL -->|Invalid| RESP
    EMB --> UMAP
    UMAP --> HDBSCAN
    HDBSCAN --> NOISE
    NOISE --> SENT
    NOISE --> KW
    SENT --> INS
    KW --> INS
    INS --> FMT
    FMT --> RESP

    style EMB fill:#4CAF50
    style HDBSCAN fill:#4CAF50
    style SENT fill:#4CAF50
    style RESP fill:#2196F3
```

## 3. Lambda Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant APIGW as API Gateway
    participant Lambda as Lambda Handler
    participant Cache as Global Cache
    participant ML as ML Pipeline
    participant CW as CloudWatch

    Client->>APIGW: POST /analyze<br/>{baseline, query}
    APIGW->>Lambda: Invoke (event, context)

    alt Cold Start
        Lambda->>Cache: Check models loaded?
        Cache-->>Lambda: None
        Lambda->>Lambda: Import ML modules<br/>(10-15s, lazy load)
        Lambda->>Lambda: Download models to /tmp<br/>(2-3s)
        Lambda->>Cache: Cache models globally
        Note over Lambda,Cache: Total: ~4-5s cold start
    else Warm Start
        Lambda->>Cache: Check models loaded?
        Cache-->>Lambda: Already cached
        Note over Lambda,Cache: Total: ~990ms init
    end

    Lambda->>Lambda: Parse & validate request

    alt Validation Fails
        Lambda->>CW: Log validation error
        Lambda->>APIGW: 400 Error Response
        APIGW->>Client: Validation failed
    else Validation Success
        Lambda->>ML: Run analysis pipeline

        ML->>ML: Generate embeddings (batch_size=32)
        ML->>ML: UMAP dimensionality reduction
        ML->>ML: HDBSCAN clustering
        ML->>ML: Reassign noise points
        ML->>ML: Analyze sentiment (VADER)
        ML->>ML: Extract keywords (TF-IDF)
        ML->>ML: Generate insights

        ML-->>Lambda: Analysis results
        Lambda->>Lambda: Format response
        Lambda->>CW: Log success (duration, clusters)
        Lambda->>APIGW: 200 Response
        APIGW->>Client: Analysis results JSON
    end
```

## 4. Deployment Pipeline

```mermaid
graph TB
    subgraph "Developer"
        DEV[Developer<br/>git push main]
    end

    subgraph "GitHub Actions Workflow"
        TRIGGER[Trigger: Push to main]

        subgraph "Test Stage"
            TEST1[Install Python 3.12]
            TEST2[Install dependencies]
            TEST3[Run pytest<br/>Unit + Integration]
        end

        subgraph "Deploy Stage"
            DEP1[Set up Docker Buildx]
            DEP2[Configure AWS credentials]
            DEP3[Build Docker image<br/>platform: linux/amd64]
            DEP4[Push to ECR]
            DEP5[CDK synth]
            DEP6[CDK deploy --require-approval never]
        end

        subgraph "Verify Stage"
            VER1[Get deployment outputs]
            VER2[Test API endpoint<br/>curl POST request]
            VER3[Verify response]
        end
    end

    subgraph "AWS"
        ECR2[ECR Repository]
        CFN[CloudFormation Stack]
        LAMBDA2[Lambda Function<br/>Updated]
        APIGW2[API Gateway<br/>Updated]
    end

    DEV --> TRIGGER
    TRIGGER --> TEST1
    TEST1 --> TEST2
    TEST2 --> TEST3

    TEST3 -->|Tests Pass| DEP1
    TEST3 -->|Tests Fail| FAIL[❌ Workflow Failed]

    DEP1 --> DEP2
    DEP2 --> DEP3
    DEP3 --> DEP4
    DEP4 --> ECR2
    DEP4 --> DEP5
    DEP5 --> DEP6
    DEP6 --> CFN

    CFN --> LAMBDA2
    CFN --> APIGW2

    LAMBDA2 --> VER1
    VER1 --> VER2
    VER2 --> VER3
    VER3 --> SUCCESS[✅ Deployment Complete]

    style TEST3 fill:#4CAF50
    style SUCCESS fill:#4CAF50
    style FAIL fill:#f44336
```

## 5. Data Flow - Request Processing

```mermaid
flowchart TD
    START([API Request]) --> PARSE{Parse JSON Body}

    PARSE -->|Success| VALIDATE[Pydantic Validation]
    PARSE -->|Fail| ERR1[400: Malformed JSON]

    VALIDATE -->|Valid| CHECK{Has Comparison?}
    VALIDATE -->|Invalid| ERR2[400: Validation Error<br/>• Duplicate IDs<br/>• Missing fields<br/>• Invalid types]

    CHECK -->|No| BASE[Baseline Analysis Only]
    CHECK -->|Yes| COMP[Baseline + Comparison]

    BASE --> EMB1[Generate Embeddings<br/>Baseline sentences]
    COMP --> EMB2[Generate Embeddings<br/>Baseline + Comparison]

    EMB1 --> CLUST1[Cluster Baseline]
    EMB2 --> CLUST2[Cluster Baseline<br/>Cluster Comparison<br/>Separately]

    CLUST1 --> SENT1[Sentiment Analysis<br/>Per cluster + sentence]
    CLUST2 --> SENT2[Sentiment Analysis<br/>Both datasets]

    SENT1 --> INS1[Generate Insights<br/>Keywords, Stats]
    SENT2 --> INS2[Generate Insights<br/>+ Comparison insights]

    INS1 --> FMT1[Format Response<br/>Clusters + Summary]
    INS2 --> FMT2[Format Response<br/>+ Comparison section]

    FMT1 --> RESP1[200: Success Response]
    FMT2 --> RESP2[200: Success Response]

    ERR1 --> END1([Error Response])
    ERR2 --> END1
    RESP1 --> END2([Success Response])
    RESP2 --> END2

    style CHECK fill:#FFC107
    style RESP1 fill:#4CAF50
    style RESP2 fill:#4CAF50
    style ERR1 fill:#f44336
    style ERR2 fill:#f44336
```

## 6. Caching & Performance Strategy

```mermaid
graph TB
    subgraph "Lambda Container Lifecycle"
        INIT[Container Init<br/>~990ms]

        subgraph "Global Scope Cached"
            MODEL[SentenceTransformer<br/>all-MiniLM-L6-v2]
            CLUST[TextClusterer<br/>UMAP + HDBSCAN]
            SENTIMENT[VADER Analyzer]
            FORMAT[Formatters]
        end

        subgraph "Ephemeral Storage /tmp"
            TRANS[Transformers cache<br/>/tmp/transformers]
            HF[HuggingFace cache<br/>/tmp/hf]
            SENT_CACHE[Sentence models<br/>/tmp/sentence_transformers]
            NUMBA[Numba JIT cache<br/>/tmp]
        end

        INVOKE1[Invocation 1<br/>Cold Start<br/>4-5s]
        INVOKE2[Invocation 2<br/>Warm Start<br/><3s]
        INVOKE3[Invocation 3<br/>Warm Start<br/><3s]
    end

    INIT --> INVOKE1
    INVOKE1 -->|Load & Cache| MODEL
    INVOKE1 -->|Load & Cache| CLUST
    INVOKE1 -->|Load & Cache| SENTIMENT
    INVOKE1 -->|Load & Cache| FORMAT

    INVOKE1 -->|Download to| TRANS
    INVOKE1 -->|Download to| HF
    INVOKE1 -->|Download to| SENT_CACHE

    MODEL -.Reuse.-> INVOKE2
    CLUST -.Reuse.-> INVOKE2
    SENTIMENT -.Reuse.-> INVOKE2
    FORMAT -.Reuse.-> INVOKE2

    TRANS -.Cached.-> INVOKE2
    HF -.Cached.-> INVOKE2
    SENT_CACHE -.Cached.-> INVOKE2

    MODEL -.Reuse.-> INVOKE3
    CLUST -.Reuse.-> INVOKE3
    SENTIMENT -.Reuse.-> INVOKE3
    FORMAT -.Reuse.-> INVOKE3

    style INVOKE1 fill:#ff9900
    style INVOKE2 fill:#4CAF50
    style INVOKE3 fill:#4CAF50
```

## 7. Component Dependencies

```mermaid
graph TD
    subgraph "External Dependencies"
        ST[sentence-transformers<br/>3.1.1]
        SK[scikit-learn<br/>1.5.2]
        HDB[hdbscan<br/>0.8.38.post2]
        UMAP_LIB[umap-learn<br/>0.5.6]
        VADER[vaderSentiment<br/>3.3.2]
        PYD[pydantic<br/>2.9.2]
    end

    subgraph "Custom Modules"
        subgraph "src/"
            HANDLER[lambda_function.py<br/>Main handler]

            subgraph "clustering/"
                EMB[embeddings.py]
                CLUSTR[clusterer.py]
                INSGHT[insights.py]
            end

            subgraph "sentiment/"
                ANLYZ[analyzer.py]
            end

            subgraph "utils/"
                VALID[validators.py]
                FRMTR[formatters.py]
            end
        end
    end

    subgraph "Infrastructure"
        CDK_STACK[infrastructure/stacks/<br/>lambda_stack.py]
        DOCKER[Dockerfile]
    end

    HANDLER --> EMB
    HANDLER --> CLUSTR
    HANDLER --> ANLYZ
    HANDLER --> INSGHT
    HANDLER --> VALID
    HANDLER --> FRMTR

    EMB --> ST
    CLUSTR --> SK
    CLUSTR --> HDB
    CLUSTR --> UMAP_LIB
    ANLYZ --> VADER
    VALID --> PYD
    FRMTR --> PYD

    DOCKER --> HANDLER
    CDK_STACK --> DOCKER

    style HANDLER fill:#ff9900
    style CDK_STACK fill:#2196F3
    style DOCKER fill:#2196F3
```

## 8. Error Handling Flow

```mermaid
flowchart TD
    REQ[Incoming Request] --> TRY{Try Block}

    TRY --> PARSE[Parse JSON]
    PARSE -->|Success| VAL[Validate with Pydantic]
    PARSE -->|JSONDecodeError| CATCH1[Log Error]

    VAL -->|Success| INIT[Initialize ML Components]
    VAL -->|ValidationError| CATCH2[Format Validation Error]

    INIT -->|Success| ANALYZE[Run Analysis Pipeline]
    INIT -->|ImportError<br/>ModelError| CATCH3[Log Init Error]

    ANALYZE -->|Success| FORMAT[Format Response]
    ANALYZE -->|ClusteringError<br/>SentimentError| CATCH4[Log Analysis Error]

    FORMAT --> SUCCESS[200 Response<br/>+ Request ID]

    CATCH1 --> ERR1[500: Internal Error<br/>+ Request ID<br/>+ Traceback]
    CATCH2 --> ERR2[400: Validation Error<br/>+ Request ID<br/>+ Details]
    CATCH3 --> ERR3[500: Init Error<br/>+ Request ID<br/>+ Traceback]
    CATCH4 --> ERR4[500: Processing Error<br/>+ Request ID<br/>+ Traceback]

    SUCCESS --> LOG_SUCCESS[CloudWatch:<br/>INFO level<br/>Duration, clusters, etc]
    ERR1 --> LOG_ERROR1[CloudWatch:<br/>ERROR level<br/>Full traceback]
    ERR2 --> LOG_WARN[CloudWatch:<br/>WARNING level<br/>Validation details]
    ERR3 --> LOG_ERROR2[CloudWatch:<br/>ERROR level<br/>Full traceback]
    ERR4 --> LOG_ERROR3[CloudWatch:<br/>ERROR level<br/>Full traceback]

    style SUCCESS fill:#4CAF50
    style ERR1 fill:#f44336
    style ERR2 fill:#ff9800
    style ERR3 fill:#f44336
    style ERR4 fill:#f44336
```

## 9. Testing Strategy

```mermaid
graph TB
    subgraph "Unit Tests"
        TEST_VAL[test_validators.py<br/>• Pydantic validation<br/>• Duplicate ID checks<br/>• Field validation]
        TEST_FMT[test_formatters.py<br/>• Response structure<br/>• JSON serialization<br/>• Field mapping]
        TEST_SENT[test_sentiment.py<br/>• VADER scoring<br/>• Classification logic<br/>• Edge cases]
    end

    subgraph "Integration Tests"
        TEST_INT[test_integration.py<br/>• Full handler invocation<br/>• End-to-end pipeline<br/>• Error scenarios<br/>• Performance tests]
    end

    subgraph "Data Example Tests"
        TEST_DATA[test_data_examples.py<br/>• Validate example files<br/>• Test with real data<br/>• Verify duplicate detection]
    end

    subgraph "CI Pipeline"
        PYTEST[pytest<br/>--cov=src<br/>--cov-fail-under=70]
    end

    TEST_VAL --> PYTEST
    TEST_FMT --> PYTEST
    TEST_SENT --> PYTEST
    TEST_INT --> PYTEST
    TEST_DATA --> PYTEST

    PYTEST -->|Pass| DEPLOY[Deploy to AWS]
    PYTEST -->|Fail| BLOCK[❌ Block Deployment]

    style PYTEST fill:#4CAF50
    style DEPLOY fill:#2196F3
    style BLOCK fill:#f44336
```

## 10. AWS Resource Relationships

```mermaid
graph TB
    subgraph "IAM"
        ROLE[Lambda Execution Role<br/>• CloudWatch Logs permissions]
    end

    subgraph "Networking"
        APIGW_ENDPOINT[API Gateway<br/>Public Endpoint<br/>CORS enabled]
    end

    subgraph "Compute"
        LAMBDA_FUNC[Lambda Function<br/>• Container Image<br/>• 3008 MB memory<br/>• 900s timeout<br/>• 512 MB ephemeral]
    end

    subgraph "Storage & Registry"
        ECR_REPO[ECR Repository<br/>Docker images]
        CW_LOGS[CloudWatch Log Group<br/>7 day retention]
    end

    subgraph "Infrastructure"
        CFN_STACK[CloudFormation Stack<br/>text-analysis-prod]
    end

    APIGW_ENDPOINT -->|Invoke| LAMBDA_FUNC
    LAMBDA_FUNC -->|Assumes| ROLE
    LAMBDA_FUNC -->|Pull| ECR_REPO
    LAMBDA_FUNC -->|Write| CW_LOGS

    CFN_STACK -->|Creates| APIGW_ENDPOINT
    CFN_STACK -->|Creates| LAMBDA_FUNC
    CFN_STACK -->|Creates| ROLE
    CFN_STACK -->|Creates| CW_LOGS

    style LAMBDA_FUNC fill:#ff9900
    style CFN_STACK fill:#2196F3
```

---

## Diagram Explanations

### 1. High-Level Architecture
Shows the overall AWS infrastructure, deployment pipeline, and main components.

### 2. ML Pipeline Architecture
Details the complete machine learning processing pipeline from input to output.

### 3. Lambda Execution Flow
Sequence diagram showing cold vs warm starts and request processing flow.

### 4. Deployment Pipeline
GitHub Actions workflow stages from code push to production deployment.

### 5. Data Flow - Request Processing
Decision tree showing how different request types are processed.

### 6. Caching & Performance Strategy
Illustrates global scope caching and ephemeral storage usage for performance.

### 7. Component Dependencies
Shows all module dependencies and their relationships.

### 8. Error Handling Flow
Complete error handling strategy with logging and response codes.

### 9. Testing Strategy
Test organization and CI/CD integration.

### 10. AWS Resource Relationships
CloudFormation-managed resources and their interactions.

---

## Key Performance Metrics

| Metric | Cold Start | Warm Start |
|--------|-----------|------------|
| **Init Duration** | ~4-5s | ~990ms |
| **100 sentences** | ~7-8s total | <3s |
| **500 sentences** | ~14-15s total | <10s |
| **Memory Used** | ~2.8 GB | ~2.5 GB |

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| **Runtime** | Python | 3.12 |
| **Compute** | AWS Lambda | Docker Container |
| **API** | API Gateway | REST API |
| **IaC** | AWS CDK | Python |
| **CI/CD** | GitHub Actions | - |
| **Embeddings** | sentence-transformers | 3.1.1 |
| **Clustering** | HDBSCAN + UMAP | 0.8.38.post2 / 0.5.6 |
| **Sentiment** | VADER | 3.3.2 |
| **Validation** | Pydantic | 2.9.2 |
