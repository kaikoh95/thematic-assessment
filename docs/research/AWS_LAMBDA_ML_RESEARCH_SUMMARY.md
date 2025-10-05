# AWS Lambda ML/NLP Workload Deployment Guide (2024-2025)

## 1. Python Runtime Versions

### Current Support (as of 2024-2025)
- **Python 3.11**: Fully supported (released July 2023, continues through 2025)
  - Uses Amazon Linux 2 (AL2) base
  - Deployment footprint: 100MB+
  - Package manager: yum
- **Python 3.12**: Available since January 2024
- **Python 3.13**: Latest, available since November 2024

**Recommendation**: Python 3.11 is fully supported for ML workloads. Consider Python 3.12+ for SnapStart compatibility.

**Documentation**: https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html

---

## 2. Lambda Layers: Packaging ML Dependencies

### Size Limits
- **Direct upload (zip)**: 50 MB per layer
- **Total unzipped size**: 250 MB (function + all layers combined)
- **Maximum layers per function**: 5
- **S3 upload**: Use for files >50 MB

### Creating a Layer for ML Dependencies

#### Step 1: Create Directory Structure
```bash
mkdir -p python
```

#### Step 2: Install Dependencies
For pure Python packages:
```bash
pip install sentence-transformers -t python/
```

For packages with compiled components (NumPy, scikit-learn, HDBSCAN):
```bash
pip install scikit-learn hdbscan \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  -t python/
```

Using requirements.txt:
```bash
# Create requirements.txt
cat > requirements.txt << EOF
sentence-transformers==2.2.2
scikit-learn==1.3.0
hdbscan==0.8.33
EOF

# Install dependencies
pip install -r requirements.txt \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  -t python/
```

#### Step 3: Create ZIP Archive
```bash
zip -r layer.zip python/
```

#### Step 4: Publish Layer via AWS CLI
```bash
aws lambda publish-layer-version \
  --layer-name ml-dependencies \
  --description "ML/NLP dependencies: sentence-transformers, scikit-learn, hdbscan" \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.11 python3.12
```

Or upload from S3 (for large layers):
```bash
aws s3 cp layer.zip s3://my-bucket/layers/ml-dependencies.zip

aws lambda publish-layer-version \
  --layer-name ml-dependencies \
  --description "ML/NLP dependencies" \
  --content S3Bucket=my-bucket,S3Key=layers/ml-dependencies.zip \
  --compatible-runtimes python3.11 python3.12
```

#### Step 5: Attach Layer to Function
```bash
aws lambda update-function-configuration \
  --function-name my-ml-function \
  --layers arn:aws:lambda:us-east-1:123456789012:layer:ml-dependencies:1
```

**Documentation**: https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html

---

## 3. Size Limits Summary

| Resource | Limit |
|----------|-------|
| Zipped deployment package (direct upload) | 50 MB |
| Unzipped deployment package + layers | 250 MB |
| Container image (uncompressed) | 10 GB |
| Layers per function | 5 |
| /tmp directory storage | 512 MB - 10 GB (configurable) |
| Function/layer storage per region | 75 GB |

**Note**: For large ML models (>250 MB), consider:
1. Container images (up to 10 GB)
2. Download models from S3 to /tmp at runtime
3. Amazon EFS for multi-model scenarios

**Documentation**: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html

---

## 4. Best Practices for Reducing Cold Starts

### Strategy 1: SnapStart (Recommended for Python 3.12+)
**Available since November 2024** for Python 3.12+

Enable SnapStart:
```bash
aws lambda update-function-configuration \
  --function-name my-ml-function \
  --snap-start ApplyOn=PublishedVersions
```

**Benefits**:
- Reduces cold starts from several seconds to sub-second
- Particularly effective for ML dependencies (NumPy, Pandas, LangChain, etc.)
- Works with published versions/aliases (not $LATEST)

**Note**: Not available for Python 3.11. Upgrade to Python 3.12+ to use SnapStart.

**Documentation**: https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html

### Strategy 2: Global Scope Model Caching
Load models and SDK clients outside the handler function:

```python
import json
import boto3
from sentence_transformers import SentenceTransformer

# Initialize in global scope (runs once per container)
s3_client = boto3.client('s3')
model = None

def get_model():
    """Lazy load model to optimize cold start"""
    global model
    if model is None:
        print("Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def lambda_handler(event, context):
    # Model is cached and reused across warm invocations
    model = get_model()

    # Process text
    text = event.get('text', '')
    embeddings = model.encode(text)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'embeddings': embeddings.tolist()
        })
    }
```

### Strategy 3: Provisioned Concurrency
For predictable, low-latency performance:

```bash
aws lambda put-provisioned-concurrency-auto-scaling \
  --function-name my-ml-function \
  --provisioned-concurrent-executions 5
```

**Benefits**:
- Keeps execution environments warm
- Response times in double-digit milliseconds
- Best for production ML inference APIs

### Strategy 4: Package Size Optimization
- Remove unused dependencies
- Use lightweight model variants (e.g., "all-MiniLM-L6-v2" instead of larger models)
- Exclude test files, documentation, and unnecessary artifacts
- Use tree shaking for JavaScript/TypeScript dependencies

**Documentation**:
- https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/
- https://docs.aws.amazon.com/lambda/latest/operatorguide/global-scope.html

---

## 5. Memory and Timeout Configuration for ML Workloads

### Memory Configuration
- **Range**: 128 MB to 10,240 MB (10 GB) in 1 MB increments
- **vCPU allocation**: At 1,769 MB, function gets 1 vCPU equivalent
- **AVX2 support**: Available for ML inferencing optimization

### Recommendations for ML Inference

| Model Size | Recommended Memory | Notes |
|------------|-------------------|-------|
| Small models (<100 MB) | 512 MB - 1,024 MB | Basic NLP tasks, small sentence transformers |
| Medium models (100-500 MB) | 1,024 MB - 3,008 MB | Standard sentence-transformers, scikit-learn models |
| Large models (500 MB - 2 GB) | 3,008 MB - 10,240 MB | Large transformers, ensemble models |

**Memory Tuning**:
Use AWS Lambda Power Tuning to find optimal configuration:
```bash
# Install Power Tuning tool
git clone https://github.com/alexcasalboni/aws-lambda-power-tuning.git
```

### Timeout Configuration
- **Maximum**: 900 seconds (15 minutes)
- **Recommendation for ML**:
  - First invocation (cold start + model loading): 30-60 seconds
  - Warm invocations (inference only): 5-30 seconds

Example configuration:
```bash
aws lambda update-function-configuration \
  --function-name my-ml-function \
  --memory-size 3008 \
  --timeout 60
```

### Ephemeral Storage (/tmp)
For large models downloaded from S3:
```bash
aws lambda update-function-configuration \
  --function-name my-ml-function \
  --ephemeral-storage Size=5120  # 5 GB
```

**Documentation**:
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html
- https://aws.amazon.com/blogs/compute/choosing-between-storage-mechanisms-for-ml-inferencing-with-aws-lambda/

---

## 6. Caching Models in Global Scope for Warm Starts

### Pattern 1: Direct Global Initialization (Simple)
```python
from sentence_transformers import SentenceTransformer

# Load model once during cold start
model = SentenceTransformer('all-MiniLM-L6-v2')

def lambda_handler(event, context):
    # Model is already loaded and cached
    text = event['text']
    embeddings = model.encode(text)
    return {'embeddings': embeddings.tolist()}
```

### Pattern 2: Lazy Loading (Recommended)
```python
import boto3
from sentence_transformers import SentenceTransformer
import os

# Global variables
model = None
s3_client = None

def get_s3_client():
    """Initialize S3 client once"""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client('s3')
    return s3_client

def get_model():
    """Lazy load model on first use"""
    global model
    if model is None:
        print("Initializing model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")
    return model

def lambda_handler(event, context):
    # Only loads model on first invocation
    model = get_model()

    # Use model for inference
    text = event.get('text', '')
    embeddings = model.encode(text)

    return {
        'statusCode': 200,
        'body': {
            'embeddings': embeddings.tolist(),
            'model': 'all-MiniLM-L6-v2'
        }
    }
```

### Pattern 3: S3 Model Download with Caching
```python
import os
import boto3
from sentence_transformers import SentenceTransformer

# Global variables
model = None
s3_client = boto3.client('s3')
MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']
MODEL_PATH = '/tmp/model'

def download_model_from_s3():
    """Download model from S3 to /tmp"""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        os.makedirs(MODEL_PATH, exist_ok=True)

        # Download model files
        s3_client.download_file(MODEL_BUCKET, MODEL_KEY, f'{MODEL_PATH}/model.tar.gz')

        # Extract if needed
        import tarfile
        with tarfile.open(f'{MODEL_PATH}/model.tar.gz', 'r:gz') as tar:
            tar.extractall(MODEL_PATH)
    else:
        print("Model already cached in /tmp")

def get_model():
    """Load model with S3 caching"""
    global model
    if model is None:
        download_model_from_s3()
        print("Loading model from /tmp...")
        model = SentenceTransformer(MODEL_PATH)
        print("Model loaded successfully")
    return model

def lambda_handler(event, context):
    model = get_model()

    texts = event.get('texts', [])
    embeddings = model.encode(texts)

    return {
        'statusCode': 200,
        'body': {
            'embeddings': embeddings.tolist(),
            'count': len(embeddings)
        }
    }
```

### Pattern 4: Connection Pooling (for database connections)
```python
import psycopg2
import os

# Database connection pool
db_connection = None

def get_db_connection():
    """Reuse database connection across invocations"""
    global db_connection
    if db_connection is None or db_connection.closed:
        print("Creating new database connection...")
        db_connection = psycopg2.connect(
            host=os.environ['DB_HOST'],
            database=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD']
        )
    return db_connection

def lambda_handler(event, context):
    conn = get_db_connection()
    # Use connection for queries
    # Connection persists across warm invocations
```

### Best Practices for Global Scope
1. **DO**: Initialize SDK clients, models, and connections in global scope
2. **DO**: Use lazy loading for large objects to reduce initialization time
3. **DO**: Cache static assets in /tmp directory
4. **DON'T**: Store user data or request-specific data in global variables
5. **DON'T**: Assume global state persists indefinitely (containers are recycled)

**Documentation**:
- https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- https://docs.aws.amazon.com/lambda/latest/operatorguide/global-scope.html

---

## 7. Complete Example: Deploying Sentence-Transformers on Lambda

### Directory Structure
```
ml-lambda/
├── lambda_function.py
├── requirements.txt
└── layers/
    └── ml-dependencies/
        └── python/
            ├── sentence_transformers/
            ├── sklearn/
            └── hdbscan/
```

### requirements.txt (for layer)
```
sentence-transformers==2.2.2
scikit-learn==1.3.0
hdbscan==0.8.33
numpy<2.0.0
torch==2.0.1
```

### lambda_function.py
```python
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Global model cache
model = None

def get_model():
    """Lazy load sentence transformer model"""
    global model
    if model is None:
        print("Loading sentence-transformer model...")
        # Use a smaller model to fit in Lambda limits
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")
    return model

def lambda_handler(event, context):
    """
    Lambda handler for sentence embeddings

    Expected event format:
    {
        "texts": ["sentence 1", "sentence 2", ...]
    }
    """
    try:
        # Get cached model
        model = get_model()

        # Extract texts from event
        texts = event.get('texts', [])
        if not texts:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No texts provided'})
            }

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts)

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'embeddings': embeddings_list,
                'count': len(embeddings_list),
                'dimension': len(embeddings_list[0])
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Deployment Steps

#### 1. Create Layer
```bash
# Create layer directory
mkdir -p layers/ml-dependencies/python

# Install dependencies (use Linux-compatible wheels)
pip install sentence-transformers scikit-learn hdbscan \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  -t layers/ml-dependencies/python/

# Create ZIP
cd layers/ml-dependencies
zip -r ../../ml-dependencies.zip python/
cd ../..

# Publish layer
aws lambda publish-layer-version \
  --layer-name ml-nlp-dependencies \
  --description "Sentence-transformers, scikit-learn, hdbscan" \
  --zip-file fileb://ml-dependencies.zip \
  --compatible-runtimes python3.11 python3.12
```

#### 2. Create Function
```bash
# Create function ZIP (without dependencies)
zip function.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
  --function-name sentence-embeddings \
  --runtime python3.11 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --memory-size 3008 \
  --timeout 60 \
  --layers arn:aws:lambda:us-east-1:123456789012:layer:ml-nlp-dependencies:1
```

#### 3. Test Function
```bash
# Test invocation
aws lambda invoke \
  --function-name sentence-embeddings \
  --payload '{"texts": ["Hello world", "Machine learning is awesome"]}' \
  response.json

# View response
cat response.json
```

---

## 8. Alternative: Container Images for Large ML Models

For models >250 MB (after compression), use container images:

### Dockerfile Example
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Download model during build (optional)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD [ "lambda_function.lambda_handler" ]
```

### Build and Deploy
```bash
# Build image
docker build -t ml-inference .

# Tag for ECR
docker tag ml-inference:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-inference:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-inference:latest

# Create function from container image
aws lambda create-function \
  --function-name ml-inference-container \
  --package-type Image \
  --code ImageUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-inference:latest \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --memory-size 3008 \
  --timeout 60
```

**Documentation**: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html

---

## 9. Monitoring and Optimization

### CloudWatch Metrics to Monitor
- **Duration**: Total execution time
- **Init Duration**: Cold start initialization time
- **Memory Used**: Actual memory consumption
- **Throttles**: Rate limiting indicators

### Optimization Tools
1. **AWS Lambda Power Tuning**: Find optimal memory configuration
   - GitHub: https://github.com/alexcasalboni/aws-lambda-power-tuning

2. **AWS Compute Optimizer**: Automated recommendations
   - https://docs.aws.amazon.com/compute-optimizer/latest/ug/view-lambda-recommendations.html

### Example CloudWatch Insights Query
```sql
fields @timestamp, @duration, @initDuration, @memorySize, @maxMemoryUsed
| filter @type = "REPORT"
| stats avg(@duration), max(@duration), avg(@maxMemoryUsed) by bin(5m)
```

---

## 10. Key Takeaways

### For sentence-transformers, scikit-learn, hdbscan deployment:

1. **Runtime**: Use Python 3.11 (or 3.12+ for SnapStart)
2. **Packaging**: Use Lambda layers for dependencies, keep function code separate
3. **Model Size**: Choose compact models (all-MiniLM-L6-v2 is ~90MB)
4. **Memory**: Start with 3,008 MB (2 vCPUs) for ML inference
5. **Timeout**: Set 60 seconds for cold starts, 30 seconds for warm
6. **Caching**: Load models in global scope with lazy initialization
7. **Cold Starts**: Enable SnapStart (Python 3.12+) or use Provisioned Concurrency
8. **Large Models**: Use container images if total size exceeds 250 MB

### Quick Reference Links
- **Lambda Runtimes**: https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html
- **Lambda Layers**: https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html
- **Size Limits**: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- **Best Practices**: https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- **SnapStart**: https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html
- **Global Scope**: https://docs.aws.amazon.com/lambda/latest/operatorguide/global-scope.html
- **Memory Config**: https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html
- **Container Images**: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html

---

*Research compiled from official AWS documentation (2024-2025)*
