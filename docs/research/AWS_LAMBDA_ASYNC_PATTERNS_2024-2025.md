# AWS Lambda Asynchronous Invocation Patterns with API Gateway
## Comprehensive Research Summary (2024-2025)

**Research Date:** October 6, 2025
**Focus:** Async job processing patterns for text analysis microservice

---

## Executive Summary

This research covers the latest AWS Lambda asynchronous invocation patterns with API Gateway for implementing async job processing systems. The recommended architecture uses:

- **Storage Solution:** DynamoDB (primary choice)
- **Invocation Pattern:** Synchronous proxy Lambda that invokes async worker Lambda
- **Error Handling:** Lambda Destinations (preferred over DLQ)
- **Cleanup:** DynamoDB TTL for automatic job expiration
- **Orchestration:** Direct Lambda async invocation (Step Functions for complex workflows only)

---

## Table of Contents

1. [Storage Solution: DynamoDB vs S3](#storage-solution-dynamodb-vs-s3)
2. [Lambda Asynchronous Invocation Methods](#lambda-asynchronous-invocation-methods)
3. [API Gateway Integration Patterns](#api-gateway-integration-patterns)
4. [Response and Status Endpoint Structure](#response-and-status-endpoint-structure)
5. [Error Handling: Destinations vs DLQ](#error-handling-destinations-vs-dlq)
6. [Step Functions vs Direct Lambda Async](#step-functions-vs-direct-lambda-async)
7. [DynamoDB Schema Design](#dynamodb-schema-design)
8. [TTL Patterns for Automatic Cleanup](#ttl-patterns-for-automatic-cleanup)
9. [CDK Implementation Patterns](#cdk-implementation-patterns)
10. [Best Practices and Recommendations](#best-practices-and-recommendations)

---

## 1. Storage Solution: DynamoDB vs S3

### Recommendation: **DynamoDB** (Primary Choice)

### Why DynamoDB for Async Job Results?

**Performance:**
- Consistent millisecond latency vs S3's variable latency
- Ideal for job status tracking requiring frequent reads
- Fast query capabilities by job ID, status, or timestamp

**Functionality:**
- Native support for querying data based on attributes
- Process groups of related items together
- Update individual records without rewriting entire objects
- Global Secondary Indexes (GSI) enable flexible querying patterns

**Cost (2024 Updates):**
- November 2024: DynamoDB on-demand throughput costs reduced by 50%
- Standard-Infrequent Access storage reduces costs by 60% for long-tail data
- TTL-based deletion attracts no additional costs

**Operational:**
- Built-in TTL for automatic cleanup
- DynamoDB Streams for event-driven processing
- No initialization handshakes required
- Scales on demand without being overloaded by Lambda spikes

### When to Use S3 Instead

S3 is better when you:
- Never need ad-hoc queries on objects
- Always access data by primary key, one at a time
- Store larger objects or files (>400KB)
- Have large, infrequently accessed data

**S3 Limitations for Job Status:**
- No querying capabilities
- Cannot update files (must rewrite entire object)
- Higher and less consistent latency
- No atomic updates

### Hybrid Pattern: DynamoDB + S3

For large result payloads:
```
1. Store job metadata and status in DynamoDB
2. Store large results (>400KB) in S3
3. DynamoDB record includes S3 object key reference
4. Use DynamoDB Streams to trigger S3 archival for completed jobs
```

**Pattern:**
```
DynamoDB (fast status tracking) → Lambda enrichment → S3 (durable storage)
```

---

## 2. Lambda Asynchronous Invocation Methods

### How Async Invocation Works

When you invoke a Lambda function asynchronously:
1. Lambda places the event in an internal queue
2. Returns HTTP 202 immediately (no content)
3. Internal poller invokes function synchronously from queue
4. Retries up to 2 additional times on failure
5. Events retained for up to 6 hours

### Key Characteristics

**No Throttling:**
- Async invocations NEVER experience throttling at the API level
- When internal poller is throttled, request returns to queue
- Retried for up to 6 hours
- Many developers unnecessarily use SNS → Lambda to "protect" against throttling

**Automatic Retries:**
- 2 additional attempts with exponential backoff
- Configurable retry behavior
- Failed events sent to Destinations or DLQ

**Payload Limits:**
- Max payload: 256 KB for async invocations
- If you need larger payloads, store in S3 and pass reference

### Invocation Type Header

```bash
# Synchronous (default)
InvocationType: RequestResponse

# Asynchronous
InvocationType: Event
```

**For API Gateway:**
```
X-Amz-Invocation-Type: 'Event'  # Note: quotes are important!
```

---

## 3. API Gateway Integration Patterns

### Pattern Comparison

| Pattern | API Type | Complexity | Use Case |
|---------|----------|------------|----------|
| Non-Proxy Integration | REST API (V1) | Medium | Direct async invocation |
| Proxy Lambda Pattern | REST/HTTP API | Low | Recommended for HTTP APIs |
| Direct Service Integration | REST API (V1) | High | No Lambda for simple operations |

### Pattern 1: Non-Proxy Integration (REST API V1 Only)

**Configuration:**
```typescript
// In Integration Request, add header:
"integration.request.header.X-Amz-Invocation-Type": "'Event'"
```

**Pros:**
- Direct async invocation
- No additional Lambda required
- Returns HTTP 202 immediately

**Cons:**
- Only works with REST API (V1)
- Cannot use Lambda proxy integration
- More complex API Gateway configuration

**Important:** HTTP APIs (API Gateway V2) do NOT support setting X-Amz-Invocation-Type header.

### Pattern 2: Synchronous Proxy Lambda (Recommended)

**Architecture:**
```
Client → API Gateway → Proxy Lambda (sync) → Worker Lambda (async)
                            ↓
                       Generate Job ID
                       Write to DynamoDB
                       Return 202 + Job ID
```

**Why This Pattern?**

1. **Works with both REST and HTTP APIs**
2. **Uses familiar Lambda proxy integration**
3. **Full control over job ID generation and initial state**
4. **Proper error handling and validation before queueing**

**Proxy Lambda Responsibilities:**
```python
def lambda_handler(event, context):
    # 1. Validate input
    # 2. Generate unique job ID (UUID)
    # 3. Write initial job record to DynamoDB (status=PENDING)
    # 4. Invoke worker Lambda asynchronously
    # 5. Return 202 with job ID immediately

    job_id = str(uuid.uuid4())

    # Write to DynamoDB
    table.put_item(Item={
        'PK': f'JOB#{job_id}',
        'SK': 'METADATA',
        'status': 'PENDING',
        'createdAt': int(time.time()),
        'ttl': int(time.time()) + 86400  # 24 hours
    })

    # Invoke worker async
    lambda_client.invoke(
        FunctionName='worker-lambda',
        InvocationType='Event',  # Async
        Payload=json.dumps({
            'jobId': job_id,
            'data': event['body']
        })
    )

    return {
        'statusCode': 202,
        'body': json.dumps({'jobId': job_id})
    }
```

**Worker Lambda Responsibilities:**
```python
def lambda_handler(event, context):
    job_id = event['jobId']

    try:
        # Update status to PROCESSING
        table.update_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #status = :status',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':status': 'PROCESSING'}
        )

        # Do expensive processing
        result = process_text_analysis(event['data'])

        # Update with results
        table.update_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #status = :status, #result = :result, #completedAt = :completedAt',
            ExpressionAttributeNames={
                '#status': 'status',
                '#result': 'result',
                '#completedAt': 'completedAt'
            },
            ExpressionAttributeValues={
                ':status': 'COMPLETED',
                ':result': result,
                ':completedAt': int(time.time())
            }
        )

    except Exception as e:
        # Update status to FAILED
        table.update_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #status = :status, #error = :error',
            ExpressionAttributeNames={'#status': 'status', '#error': 'error'},
            ExpressionAttributeValues={
                ':status': 'FAILED',
                ':error': str(e)
            }
        )
        raise  # Re-raise to trigger Destinations
```

### Pattern 3: Client-Controlled Invocation Type

Allow clients to choose sync vs async:

```typescript
// Method Request: Add header parameter
InvocationType

// Integration Request: Map to Lambda header
"integration.request.header.X-Amz-Invocation-Type": "method.request.header.InvocationType"
```

Clients can then specify:
- `InvocationType: Event` for async
- `InvocationType: RequestResponse` for sync

---

## 4. Response and Status Endpoint Structure

### Endpoint Design

```
POST /jobs          → Create new async job (returns 202 + jobId)
GET  /jobs/{id}     → Get job status and results
GET  /jobs          → List jobs (optional, requires GSI)
DELETE /jobs/{id}   → Cancel/delete job (optional)
```

### POST /jobs Response

**Success (202 Accepted):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "createdAt": "2025-10-06T12:34:56Z",
  "statusUrl": "/jobs/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error (400 Bad Request):**
```json
{
  "error": "ValidationError",
  "message": "Invalid input format",
  "details": {
    "field": "baseline",
    "issue": "Array cannot be empty"
  }
}
```

### GET /jobs/{id} Response States

**1. PENDING (Job Queued):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "createdAt": "2025-10-06T12:34:56Z",
  "estimatedCompletion": "2025-10-06T12:35:06Z"
}
```

**2. PROCESSING (Job Running):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "createdAt": "2025-10-06T12:34:56Z",
  "startedAt": "2025-10-06T12:34:58Z",
  "progress": 45  // Optional
}
```

**3. COMPLETED (Success):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "createdAt": "2025-10-06T12:34:56Z",
  "startedAt": "2025-10-06T12:34:58Z",
  "completedAt": "2025-10-06T12:35:03Z",
  "duration": 5,
  "result": {
    "clusters": [...]
  }
}
```

**4. FAILED (Error):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "FAILED",
  "createdAt": "2025-10-06T12:34:56Z",
  "startedAt": "2025-10-06T12:34:58Z",
  "failedAt": "2025-10-06T12:35:00Z",
  "error": {
    "type": "ProcessingError",
    "message": "Insufficient sentences for clustering",
    "retriable": false
  }
}
```

**5. NOT FOUND (404):**
```json
{
  "error": "NotFound",
  "message": "Job not found or expired"
}
```

### Status Endpoint Implementation

```python
def get_job_status(job_id):
    try:
        response = table.get_item(
            Key={
                'PK': f'JOB#{job_id}',
                'SK': 'METADATA'
            }
        )

        if 'Item' not in response:
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'error': 'NotFound',
                    'message': 'Job not found or expired'
                })
            }

        item = response['Item']

        # Build response based on status
        result = {
            'jobId': job_id,
            'status': item['status'],
            'createdAt': item['createdAt']
        }

        if item['status'] == 'PROCESSING':
            result['startedAt'] = item.get('startedAt')

        elif item['status'] == 'COMPLETED':
            result['completedAt'] = item.get('completedAt')
            result['duration'] = item.get('completedAt', 0) - item.get('startedAt', 0)
            result['result'] = item.get('result')

        elif item['status'] == 'FAILED':
            result['failedAt'] = item.get('failedAt')
            result['error'] = item.get('error')

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'InternalError',
                'message': str(e)
            })
        }
```

### Client Polling Pattern

```javascript
async function pollJobStatus(jobId, maxAttempts = 60, interval = 2000) {
    for (let i = 0; i < maxAttempts; i++) {
        const response = await fetch(`/jobs/${jobId}`);
        const data = await response.json();

        if (data.status === 'COMPLETED') {
            return data.result;
        }

        if (data.status === 'FAILED') {
            throw new Error(data.error.message);
        }

        // Exponential backoff
        const delay = Math.min(interval * Math.pow(1.5, i), 30000);
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    throw new Error('Job timeout');
}
```

---

## 5. Error Handling: Destinations vs DLQ

### Recommendation: **Lambda Destinations** (Preferred)

### Lambda Destinations (Modern Approach)

**Advantages:**
- More informative than DLQ (includes request/response details)
- Support multiple targets (SQS, SNS, Lambda, EventBridge)
- Separate destinations for success and failure
- JSON payload includes full invocation record

**Destination Payload:**
```json
{
  "version": "1.0",
  "timestamp": "2025-10-06T12:35:00Z",
  "requestContext": {
    "requestId": "c6af9ac6-7b61-11e6-9a41-93e8deadbeef",
    "functionArn": "arn:aws:lambda:us-east-1:123456789012:function:worker",
    "condition": "RetriesExhausted",
    "approximateInvokeCount": 3
  },
  "requestPayload": {
    "jobId": "550e8400-e29b-41d4-a716-446655440000",
    "data": "..."
  },
  "responseContext": {
    "statusCode": 200,
    "executedVersion": "$LATEST",
    "functionError": "Unhandled"
  },
  "responsePayload": {
    "errorType": "ProcessingError",
    "errorMessage": "Failed to cluster sentences",
    "trace": [...]
  }
}
```

**Configuration Options:**
- OnSuccess: SQS, SNS, Lambda, EventBridge
- OnFailure: SQS, SNS, Lambda, EventBridge
- Multiple destinations per function

### Dead Letter Queue (Legacy Approach)

**Disadvantages:**
- Only sends event content (no response details)
- Only SQS or SNS as targets
- Function-level configuration only (affects all versions)
- Less information for debugging

**DLQ Payload:**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "data": "..."
}
```

### Recommended Error Handling Architecture

```
Worker Lambda
    ↓ (on success)
    Success Destination (optional) → Metrics/Logging
    ↓ (on failure)
    Failure Destination (SQS) → Error Handler Lambda → Update DynamoDB + Alerts
```

**Why SQS for Failure Destination:**
- Provides buffer for error processing
- Allows batch processing of failures
- Built-in retry with configurable delays
- Can be monitored for queue depth alerts

**Error Handler Lambda:**
```python
def handle_failed_job(event, context):
    """Process failed async job invocations"""

    for record in event['Records']:
        # Parse Destination payload
        payload = json.loads(record['body'])

        job_id = payload['requestPayload']['jobId']
        error_info = payload['responsePayload']

        # Update job status in DynamoDB
        table.update_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #status = :status, #error = :error, #failedAt = :failedAt',
            ExpressionAttributeNames={
                '#status': 'status',
                '#error': 'error',
                '#failedAt': 'failedAt'
            },
            ExpressionAttributeValues={
                ':status': 'FAILED',
                ':error': {
                    'type': error_info['errorType'],
                    'message': error_info['errorMessage'],
                    'retriable': is_retriable(error_info)
                },
                ':failedAt': int(time.time())
            }
        )

        # Send alert for critical errors
        if is_critical_error(error_info):
            sns.publish(
                TopicArn=ALERT_TOPIC_ARN,
                Subject=f'Critical Job Failure: {job_id}',
                Message=json.dumps(error_info, indent=2)
            )
```

### Reprocessing Failed Jobs

**For Destinations:**
```python
# Extract original payload from Destination event
original_payload = destination_event['requestPayload']

# Re-invoke with same payload
lambda_client.invoke(
    FunctionName='worker-lambda',
    InvocationType='Event',
    Payload=json.dumps(original_payload)
)
```

**For DLQ:**
```python
# Setup subscription to DLQ (disabled by default)
# Enable manually when reprocessing needed

def reprocess_dlq_messages(event, context):
    for record in event['Records']:
        # DLQ only has original event
        original_payload = json.loads(record['body'])

        lambda_client.invoke(
            FunctionName='worker-lambda',
            InvocationType='Event',
            Payload=json.dumps(original_payload)
        )
```

---

## 6. Step Functions vs Direct Lambda Async

### When to Choose Direct Lambda Async Invocation

**Use Direct Lambda Async When:**
- Simple, single-step async processing
- No complex orchestration needed
- Cost optimization is priority
- Processing SQS, EventBridge, or S3 events
- Task completes within 15 minutes
- Simple retry logic is sufficient

**Advantages:**
- Lower cost ($0.20 per 1M requests vs $25 per 1M state transitions)
- Simpler architecture
- Built-in retry (2 additional attempts)
- No cold starts for the orchestrator
- Direct event-driven processing

**Limitations:**
- 15-minute execution limit
- Limited visibility into execution state
- Logs mixed together in CloudWatch
- Manual correlation of related invocations
- No built-in waiting/polling capabilities

### When to Choose Step Functions

**Use Step Functions When:**
- Complex multi-step workflows
- Coordination between multiple services
- Long-running processes (>15 minutes, up to 1 year)
- Visual workflow monitoring required
- Sophisticated error handling needed
- Audit trail and compliance requirements
- Waiting/polling required in workflow
- Human approval steps needed

**Advantages:**
- Visual workflow representation
- Built-in audit history
- Execution tracking and debugging
- Direct service integrations (no Lambda needed)
- Sophisticated retry with exponential backoff
- Different retry policies per step
- Parallel execution support
- No code for service orchestration

**Disadvantages:**
- Higher cost ($25 per 1M state transitions)
- Additional service complexity
- Constrained by both Step Functions AND Lambda limits
- Learning curve for ASL (Amazon States Language)

### Cost Comparison Example

**Scenario:** 1M async jobs per month

**Direct Lambda Async:**
```
Lambda invocations: 1M requests = $0.20
Lambda duration: 1M * 3s * 128MB = ~$0.60
DynamoDB: 1M writes + 5M reads = ~$1.50
Total: ~$2.30/month
```

**Step Functions + Lambda:**
```
Step Functions: 3M state transitions (start/invoke/end) = $75.00
Lambda invocations: 1M requests = $0.20
Lambda duration: 1M * 3s * 128MB = ~$0.60
DynamoDB: 1M writes + 5M reads = ~$1.50
Total: ~$77.30/month
```

**Cost Difference:** ~33x more expensive

### Recommendation for Text Analysis Microservice

**Use Direct Lambda Async Invocation Because:**

1. **Single-step processing:** Analyze text → Generate clusters → Store results
2. **No complex orchestration:** No waiting, no human approval, no branching logic
3. **Cost-effective:** 33x cheaper for simple async pattern
4. **Sufficient error handling:** Destinations provide adequate failure tracking
5. **Performance:** No cold starts from orchestrator
6. **Within time limits:** Text analysis completes in <15 minutes

**Consider Step Functions Later If:**
- Adding multi-stage pipeline (preprocess → analyze → post-process)
- Implementing comparison analysis requiring coordination
- Need to wait for external API responses
- Compliance requires detailed audit trails
- Workflow becomes complex with conditional logic

---

## 7. DynamoDB Schema Design

### Single-Table Design for Job Tracking

**Table Design:**
```
Table: AsyncJobs
- PK (Partition Key): String
- SK (Sort Key): String
- GSI1PK (GSI Partition Key): String
- GSI1SK (GSI Sort Key): String
- Other attributes...
```

### Access Patterns

1. Get job by ID → Query by PK
2. List jobs by status → Query GSI1
3. List jobs by user → Query with user-based PK
4. Get jobs created in time range → Query with SK range
5. Automatic cleanup after N days → TTL

### Schema Patterns

#### Pattern 1: Simple Job-Only Schema

```python
# Main table item
{
    'PK': 'JOB#550e8400-e29b-41d4-a716-446655440000',
    'SK': 'METADATA',
    'jobId': '550e8400-e29b-41d4-a716-446655440000',
    'status': 'COMPLETED',  # PENDING | PROCESSING | COMPLETED | FAILED
    'surveyTitle': 'Robinhood App Store',
    'theme': 'account',
    'createdAt': 1696598400,  # Unix timestamp
    'startedAt': 1696598402,
    'completedAt': 1696598407,
    'duration': 5,
    'result': {...},  # Clusters and analysis
    'error': None,
    'ttl': 1696684800,  # createdAt + 24 hours

    # GSI attributes
    'GSI1PK': 'STATUS#COMPLETED',
    'GSI1SK': '1696598400'  # createdAt for sorting
}
```

**Access Patterns:**
```python
# Get job by ID
response = table.get_item(
    Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'}
)

# Query jobs by status
response = table.query(
    IndexName='GSI1',
    KeyConditionExpression='GSI1PK = :pk',
    ExpressionAttributeValues={':pk': 'STATUS#COMPLETED'}
)

# Query jobs by status in time range
response = table.query(
    IndexName='GSI1',
    KeyConditionExpression='GSI1PK = :pk AND GSI1SK BETWEEN :start AND :end',
    ExpressionAttributeValues={
        ':pk': 'STATUS#COMPLETED',
        ':start': '1696500000',
        ':end': '1696600000'
    }
)
```

#### Pattern 2: Multi-Tenant with User Isolation

```python
# Main table item
{
    'PK': 'USER#user123#JOB#550e8400',  # User-scoped
    'SK': 'METADATA',
    'jobId': '550e8400-e29b-41d4-a716-446655440000',
    'userId': 'user123',
    'status': 'COMPLETED',
    'createdAt': 1696598400,
    # ... other fields

    # GSI for querying by status across all users
    'GSI1PK': 'STATUS#COMPLETED',
    'GSI1SK': 'USER#user123#1696598400'
}
```

**Access Patterns:**
```python
# Get user's job
response = table.get_item(
    Key={
        'PK': f'USER#{user_id}#JOB#{job_id}',
        'SK': 'METADATA'
    }
)

# List all jobs for user
response = table.query(
    KeyConditionExpression='PK BEGINS_WITH :pk',
    ExpressionAttributeValues={':pk': f'USER#{user_id}#JOB#'}
)

# List user's jobs by status
response = table.query(
    IndexName='GSI1',
    KeyConditionExpression='GSI1PK = :status AND begins_with(GSI1SK, :user)',
    ExpressionAttributeValues={
        ':status': 'STATUS#COMPLETED',
        ':user': f'USER#{user_id}#'
    }
)
```

#### Pattern 3: Hierarchical Data with Result Separation

For large results (>400KB), separate metadata from results:

```python
# Metadata item
{
    'PK': 'JOB#550e8400-e29b-41d4-a716-446655440000',
    'SK': 'METADATA',
    'status': 'COMPLETED',
    'createdAt': 1696598400,
    # ... other metadata
    's3ResultKey': 's3://bucket/results/550e8400.json',  # For large results
    'GSI1PK': 'STATUS#COMPLETED',
    'GSI1SK': '1696598400'
}

# Optional: Result item (if <400KB)
{
    'PK': 'JOB#550e8400-e29b-41d4-a716-446655440000',
    'SK': 'RESULT',
    'clusters': [...]
}
```

### Global Secondary Index (GSI) Design

```typescript
// CDK Configuration
const jobsTable = new dynamodb.Table(this, 'AsyncJobsTable', {
    tableName: 'AsyncJobs',
    partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    timeToLiveAttribute: 'ttl',
    removalPolicy: cdk.RemovalPolicy.RETAIN,
    pointInTimeRecovery: true
});

// GSI for querying by status
jobsTable.addGlobalSecondaryIndex({
    indexName: 'GSI1',
    partitionKey: { name: 'GSI1PK', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'GSI1SK', type: dynamodb.AttributeType.STRING },
    projectionType: dynamodb.ProjectionType.ALL
});
```

### Status Value Design

**Composite Status Keys for Better Querying:**

```python
# Instead of just 'COMPLETED', use composite keys:
'GSI1PK': 'STATUS#COMPLETED#2025-10'  # Monthly partition
'GSI1PK': 'STATUS#FAILED#RETRY_EXHAUSTED'  # Failure type
'GSI1PK': 'STATUS#PROCESSING#HIGH_PRIORITY'  # Priority
```

**Benefits:**
- Avoid hot partitions (all COMPLETED in one partition)
- Enable more specific queries
- Better performance at scale

### Best Practices

1. **Use High-Cardinality Partition Keys:** Job ID is excellent (unique per item)
2. **Composite Sort Keys:** Enable hierarchical queries (`STATUS#TIMESTAMP`)
3. **Overload GSI Attributes:** Reuse GSI1PK/GSI1SK for multiple query patterns
4. **Sparse Indexes:** Only items with GSI1PK are indexed (saves cost)
5. **Avoid Hot Partitions:** Don't put all jobs with same status in one partition

---

## 8. TTL Patterns for Automatic Cleanup

### DynamoDB TTL Overview

**Key Features:**
- Automatically deletes expired items
- No additional cost
- No write capacity consumed
- Asynchronous deletion (within 48 hours of expiration)
- Based on Unix epoch timestamp (seconds)

### How TTL Works

1. Enable TTL on a table attribute (e.g., `ttl`)
2. Set attribute to Unix timestamp when item should expire
3. DynamoDB scans and deletes expired items in background
4. Deletion may take up to 48 hours after expiration time
5. Deleted items can be captured via DynamoDB Streams

### TTL Implementation

```python
import time

# Calculate expiration timestamp
def create_job_record(job_id, ttl_days=1):
    current_time = int(time.time())
    expiration_time = current_time + (ttl_days * 86400)  # 86400 = 1 day

    table.put_item(Item={
        'PK': f'JOB#{job_id}',
        'SK': 'METADATA',
        'status': 'PENDING',
        'createdAt': current_time,
        'ttl': expiration_time  # TTL attribute
    })
```

### TTL Strategies by Job Status

#### Strategy 1: Fixed TTL from Creation

**All jobs expire after N days regardless of status:**

```python
TTL_DAYS = 7

table.put_item(Item={
    'PK': f'JOB#{job_id}',
    'SK': 'METADATA',
    'status': 'PENDING',
    'createdAt': int(time.time()),
    'ttl': int(time.time()) + (TTL_DAYS * 86400)
})
```

**Pros:** Simple, predictable
**Cons:** Completed jobs deleted same as failed jobs

#### Strategy 2: Status-Based TTL

**Different retention periods based on status:**

```python
TTL_CONFIG = {
    'PENDING': 1,      # 1 day
    'PROCESSING': 1,   # 1 day
    'COMPLETED': 7,    # 7 days
    'FAILED': 14       # 14 days (longer for debugging)
}

def update_job_status(job_id, new_status):
    ttl_days = TTL_CONFIG[new_status]
    new_ttl = int(time.time()) + (ttl_days * 86400)

    table.update_item(
        Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
        UpdateExpression='SET #status = :status, #ttl = :ttl',
        ExpressionAttributeNames={'#status': 'status', '#ttl': 'ttl'},
        ExpressionAttributeValues={':status': new_status, ':ttl': new_ttl}
    )
```

**Pros:** Flexible retention by status
**Cons:** More complex, TTL updates on status change

#### Strategy 3: Extend TTL on Access

**Jobs actively accessed get extended retention:**

```python
def get_job_with_ttl_extension(job_id, extend_days=7):
    response = table.get_item(
        Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'}
    )

    if 'Item' in response:
        # Extend TTL when job is accessed
        new_ttl = int(time.time()) + (extend_days * 86400)

        table.update_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #ttl = :ttl',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={':ttl': new_ttl}
        )

    return response.get('Item')
```

**Pros:** Active jobs retained longer
**Cons:** Additional write costs, complex logic

### TTL + DynamoDB Streams Pattern

**Capture deleted items for archival or metrics:**

```python
def handle_ttl_deletion(event, context):
    """Triggered by DynamoDB Stream when TTL deletes items"""

    for record in event['Records']:
        if record['eventName'] == 'REMOVE':
            # Check if deletion was from TTL
            user_identity = record.get('userIdentity', {})
            if user_identity.get('type') == 'Service' and \
               user_identity.get('principalId') == 'dynamodb.amazonaws.com':

                # This was a TTL deletion
                old_image = record['dynamodb']['OldImage']
                job_id = old_image['jobId']['S']
                status = old_image['status']['S']

                # Archive to S3 for compliance
                s3.put_object(
                    Bucket='job-archive-bucket',
                    Key=f'archived-jobs/{job_id}.json',
                    Body=json.dumps(old_image)
                )

                # Update metrics
                cloudwatch.put_metric_data(
                    Namespace='AsyncJobs',
                    MetricData=[{
                        'MetricName': 'JobsExpired',
                        'Value': 1,
                        'Dimensions': [{'Name': 'Status', 'Value': status}]
                    }]
                )
```

### Recommended TTL Configuration

**For Text Analysis Microservice:**

```python
# Recommended TTL settings
TTL_CONFIG = {
    'PENDING': {
        'days': 1,
        'reason': 'Stuck jobs should be cleaned up quickly'
    },
    'PROCESSING': {
        'days': 1,
        'reason': 'Should complete or fail within hours'
    },
    'COMPLETED': {
        'days': 7,
        'reason': 'Allow reasonable time for result retrieval'
    },
    'FAILED': {
        'days': 14,
        'reason': 'Keep longer for debugging'
    }
}
```

### CDK TTL Configuration

```typescript
const jobsTable = new dynamodb.Table(this, 'AsyncJobsTable', {
    tableName: 'AsyncJobs',
    partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    timeToLiveAttribute: 'ttl',  // Enable TTL on 'ttl' attribute
    stream: dynamodb.StreamViewType.OLD_IMAGE,  // Capture deleted items
    removalPolicy: cdk.RemovalPolicy.RETAIN
});

// Lambda to handle TTL deletions (optional)
const ttlHandlerLambda = new lambda.Function(this, 'TTLHandler', {
    runtime: lambda.Runtime.PYTHON_3_12,
    handler: 'index.handler',
    code: lambda.Code.fromAsset('lambda/ttl-handler'),
    environment: {
        ARCHIVE_BUCKET: archiveBucket.bucketName
    }
});

// Grant S3 permissions
archiveBucket.grantWrite(ttlHandlerLambda);

// Trigger on DynamoDB Stream
ttlHandlerLambda.addEventSource(
    new lambdaEventSources.DynamoEventSource(jobsTable, {
        startingPosition: lambda.StartingPosition.LATEST,
        batchSize: 100,
        bisectBatchOnError: true,
        retryAttempts: 3
    })
);
```

### Monitoring TTL Deletions

```python
# CloudWatch metric filter for TTL monitoring
import boto3

cloudwatch = boto3.client('cloudwatch')

def create_ttl_metrics():
    cloudwatch.put_metric_alarm(
        AlarmName='HighTTLDeletionRate',
        MetricName='JobsExpired',
        Namespace='AsyncJobs',
        Statistic='Sum',
        Period=3600,  # 1 hour
        EvaluationPeriods=1,
        Threshold=1000,
        ComparisonOperator='GreaterThanThreshold',
        AlarmActions=['arn:aws:sns:us-east-1:123456789012:alerts']
    )
```

---

## 9. CDK Implementation Patterns

### Complete CDK Stack for Async Job Processing

```typescript
// lib/async-job-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class AsyncJobStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // ==========================================
        // DynamoDB Table
        // ==========================================
        const jobsTable = new dynamodb.Table(this, 'AsyncJobsTable', {
            tableName: 'TextAnalysisJobs',
            partitionKey: {
                name: 'PK',
                type: dynamodb.AttributeType.STRING
            },
            sortKey: {
                name: 'SK',
                type: dynamodb.AttributeType.STRING
            },
            billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
            timeToLiveAttribute: 'ttl',
            stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
            pointInTimeRecovery: true,
            removalPolicy: cdk.RemovalPolicy.RETAIN
        });

        // GSI for querying by status
        jobsTable.addGlobalSecondaryIndex({
            indexName: 'StatusIndex',
            partitionKey: {
                name: 'GSI1PK',
                type: dynamodb.AttributeType.STRING
            },
            sortKey: {
                name: 'GSI1SK',
                type: dynamodb.AttributeType.STRING
            },
            projectionType: dynamodb.ProjectionType.ALL
        });

        // ==========================================
        // SQS Queue for Failed Jobs
        // ==========================================
        const failedJobsDLQ = new sqs.Queue(this, 'FailedJobsDLQ', {
            queueName: 'text-analysis-failed-jobs-dlq',
            retentionPeriod: cdk.Duration.days(14)
        });

        const failedJobsQueue = new sqs.Queue(this, 'FailedJobsQueue', {
            queueName: 'text-analysis-failed-jobs',
            visibilityTimeout: cdk.Duration.seconds(300),
            deadLetterQueue: {
                queue: failedJobsDLQ,
                maxReceiveCount: 3
            }
        });

        // ==========================================
        // Lambda Layer (shared dependencies)
        // ==========================================
        const sharedLayer = new lambda.LayerVersion(this, 'SharedLayer', {
            code: lambda.Code.fromAsset('layers/shared'),
            compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
            description: 'Shared dependencies for text analysis'
        });

        // ==========================================
        // Worker Lambda (Async Processing)
        // ==========================================
        const workerLambda = new lambda.Function(this, 'WorkerLambda', {
            functionName: 'text-analysis-worker',
            runtime: lambda.Runtime.PYTHON_3_12,
            handler: 'index.handler',
            code: lambda.Code.fromAsset('src/lambdas/worker'),
            timeout: cdk.Duration.minutes(15),
            memorySize: 3008,  // Max memory for CPU-intensive tasks
            layers: [sharedLayer],
            environment: {
                TABLE_NAME: jobsTable.tableName,
                POWERTOOLS_SERVICE_NAME: 'text-analysis-worker',
                LOG_LEVEL: 'INFO'
            },
            reservedConcurrentExecutions: 100  // Prevent runaway scaling
        });

        // Grant DynamoDB permissions
        jobsTable.grantReadWriteData(workerLambda);

        // Configure Lambda Destination for failures
        workerLambda.addDestination(
            new lambda.EventInvokeDestination(
                lambda.DestinationType.ON_FAILURE,
                new lambda.SqsDestination(failedJobsQueue)
            )
        );

        // ==========================================
        // Proxy Lambda (API Handler)
        // ==========================================
        const proxyLambda = new lambda.Function(this, 'ProxyLambda', {
            functionName: 'text-analysis-proxy',
            runtime: lambda.Runtime.PYTHON_3_12,
            handler: 'index.handler',
            code: lambda.Code.fromAsset('src/lambdas/proxy'),
            timeout: cdk.Duration.seconds(30),
            memorySize: 512,
            environment: {
                TABLE_NAME: jobsTable.tableName,
                WORKER_FUNCTION_NAME: workerLambda.functionName,
                POWERTOOLS_SERVICE_NAME: 'text-analysis-proxy',
                LOG_LEVEL: 'INFO'
            }
        });

        // Grant permissions
        jobsTable.grantWriteData(proxyLambda);
        workerLambda.grantInvoke(proxyLambda);

        // ==========================================
        // Status Lambda (Get Job Status)
        // ==========================================
        const statusLambda = new lambda.Function(this, 'StatusLambda', {
            functionName: 'text-analysis-status',
            runtime: lambda.Runtime.PYTHON_3_12,
            handler: 'index.handler',
            code: lambda.Code.fromAsset('src/lambdas/status'),
            timeout: cdk.Duration.seconds(10),
            memorySize: 256,
            environment: {
                TABLE_NAME: jobsTable.tableName,
                POWERTOOLS_SERVICE_NAME: 'text-analysis-status',
                LOG_LEVEL: 'INFO'
            }
        });

        jobsTable.grantReadData(statusLambda);

        // ==========================================
        // Error Handler Lambda (Process Failed Jobs)
        // ==========================================
        const errorHandlerLambda = new lambda.Function(this, 'ErrorHandlerLambda', {
            functionName: 'text-analysis-error-handler',
            runtime: lambda.Runtime.PYTHON_3_12,
            handler: 'index.handler',
            code: lambda.Code.fromAsset('src/lambdas/error-handler'),
            timeout: cdk.Duration.seconds(60),
            memorySize: 256,
            environment: {
                TABLE_NAME: jobsTable.tableName,
                POWERTOOLS_SERVICE_NAME: 'text-analysis-error-handler',
                LOG_LEVEL: 'INFO'
            }
        });

        jobsTable.grantWriteData(errorHandlerLambda);

        // Subscribe to failed jobs queue
        errorHandlerLambda.addEventSource(
            new lambdaEventSources.SqsEventSource(failedJobsQueue, {
                batchSize: 10,
                reportBatchItemFailures: true
            })
        );

        // ==========================================
        // API Gateway
        // ==========================================
        const api = new apigateway.RestApi(this, 'TextAnalysisApi', {
            restApiName: 'Text Analysis Service',
            description: 'Async text analysis microservice',
            deployOptions: {
                stageName: 'prod',
                throttlingRateLimit: 100,
                throttlingBurstLimit: 200,
                metricsEnabled: true,
                loggingLevel: apigateway.MethodLoggingLevel.INFO,
                dataTraceEnabled: true
            },
            defaultCorsPreflightOptions: {
                allowOrigins: apigateway.Cors.ALL_ORIGINS,
                allowMethods: ['GET', 'POST', 'OPTIONS'],
                allowHeaders: ['Content-Type', 'Authorization']
            }
        });

        // /jobs resource
        const jobsResource = api.root.addResource('jobs');

        // POST /jobs - Create new job
        jobsResource.addMethod(
            'POST',
            new apigateway.LambdaIntegration(proxyLambda, {
                proxy: true,
                integrationResponses: [
                    {
                        statusCode: '202',
                        responseParameters: {
                            'method.response.header.Access-Control-Allow-Origin': "'*'"
                        }
                    }
                ]
            }),
            {
                methodResponses: [
                    {
                        statusCode: '202',
                        responseParameters: {
                            'method.response.header.Access-Control-Allow-Origin': true
                        }
                    },
                    { statusCode: '400' },
                    { statusCode: '500' }
                ]
            }
        );

        // /jobs/{jobId} resource
        const jobResource = jobsResource.addResource('{jobId}');

        // GET /jobs/{jobId} - Get job status
        jobResource.addMethod(
            'GET',
            new apigateway.LambdaIntegration(statusLambda, {
                proxy: true
            }),
            {
                methodResponses: [
                    { statusCode: '200' },
                    { statusCode: '404' },
                    { statusCode: '500' }
                ]
            }
        );

        // ==========================================
        // CloudWatch Alarms
        // ==========================================
        workerLambda.metricErrors().createAlarm(this, 'WorkerErrorAlarm', {
            threshold: 10,
            evaluationPeriods: 1,
            alarmDescription: 'Alert on worker Lambda errors',
            alarmName: 'text-analysis-worker-errors'
        });

        failedJobsQueue.metricApproximateNumberOfMessagesVisible().createAlarm(
            this, 'FailedJobsAlarm',
            {
                threshold: 50,
                evaluationPeriods: 1,
                alarmDescription: 'Alert on high failed job count',
                alarmName: 'text-analysis-failed-jobs-high'
            }
        );

        // ==========================================
        // Outputs
        // ==========================================
        new cdk.CfnOutput(this, 'ApiUrl', {
            value: api.url,
            description: 'API Gateway URL'
        });

        new cdk.CfnOutput(this, 'TableName', {
            value: jobsTable.tableName,
            description: 'DynamoDB table name'
        });

        new cdk.CfnOutput(this, 'WorkerFunctionName', {
            value: workerLambda.functionName,
            description: 'Worker Lambda function name'
        });
    }
}
```

### Lambda Function Implementations

#### Proxy Lambda (src/lambdas/proxy/index.py)

```python
import json
import os
import time
import uuid
import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths

logger = Logger()
tracer = Tracer()
app = APIGatewayRestResolver()

dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

TABLE_NAME = os.environ['TABLE_NAME']
WORKER_FUNCTION_NAME = os.environ['WORKER_FUNCTION_NAME']

table = dynamodb.Table(TABLE_NAME)

@tracer.capture_method
def validate_input(data):
    """Validate input data"""
    if 'surveyTitle' not in data:
        raise ValueError('surveyTitle is required')
    if 'theme' not in data:
        raise ValueError('theme is required')
    if 'baseline' not in data or not isinstance(data['baseline'], list):
        raise ValueError('baseline must be a non-empty array')
    if len(data['baseline']) == 0:
        raise ValueError('baseline cannot be empty')

    # Validate sentence structure
    for sentence in data['baseline']:
        if 'sentence' not in sentence or 'id' not in sentence:
            raise ValueError('Each baseline item must have sentence and id')

@tracer.capture_method
def create_job_record(job_id, data):
    """Create initial job record in DynamoDB"""
    current_time = int(time.time())
    ttl = current_time + (86400 * 7)  # 7 days

    table.put_item(Item={
        'PK': f'JOB#{job_id}',
        'SK': 'METADATA',
        'jobId': job_id,
        'status': 'PENDING',
        'surveyTitle': data['surveyTitle'],
        'theme': data['theme'],
        'sentenceCount': len(data['baseline']),
        'hasComparison': 'comparison' in data,
        'createdAt': current_time,
        'ttl': ttl,
        'GSI1PK': 'STATUS#PENDING',
        'GSI1SK': str(current_time)
    })

    logger.info(f"Created job record for {job_id}")

@tracer.capture_method
def invoke_worker_async(job_id, data):
    """Invoke worker Lambda asynchronously"""
    payload = {
        'jobId': job_id,
        'data': data
    }

    response = lambda_client.invoke(
        FunctionName=WORKER_FUNCTION_NAME,
        InvocationType='Event',  # Async invocation
        Payload=json.dumps(payload)
    )

    logger.info(f"Invoked worker Lambda for job {job_id}",
                extra={'statusCode': response['StatusCode']})

@app.post("/jobs")
@tracer.capture_method
def create_job():
    """Handle POST /jobs - Create new async job"""
    try:
        # Parse and validate input
        data = app.current_event.json_body
        validate_input(data)

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job record
        create_job_record(job_id, data)

        # Invoke worker asynchronously
        invoke_worker_async(job_id, data)

        # Return 202 Accepted with job ID
        return {
            'statusCode': 202,
            'body': {
                'jobId': job_id,
                'status': 'PENDING',
                'createdAt': int(time.time()),
                'message': 'Job accepted for processing',
                'statusUrl': f'/jobs/{job_id}'
            }
        }

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return {
            'statusCode': 400,
            'body': {
                'error': 'ValidationError',
                'message': str(e)
            }
        }
    except Exception as e:
        logger.exception("Unexpected error creating job")
        return {
            'statusCode': 500,
            'body': {
                'error': 'InternalError',
                'message': 'Failed to create job'
            }
        }

@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
def handler(event, context):
    return app.resolve(event, context)
```

#### Worker Lambda (src/lambdas/worker/index.py)

```python
import json
import os
import time
import boto3
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()

dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

@tracer.capture_method
def update_job_status(job_id, status, **kwargs):
    """Update job status in DynamoDB"""
    update_expr = 'SET #status = :status'
    expr_attr_names = {'#status': 'status'}
    expr_attr_values = {':status': status}

    # Add GSI attributes for status-based queries
    current_time = int(time.time())
    update_expr += ', GSI1PK = :gsi1pk, GSI1SK = :gsi1sk'
    expr_attr_values[':gsi1pk'] = f'STATUS#{status}'
    expr_attr_values[':gsi1sk'] = str(current_time)

    # Update TTL based on status
    ttl_days = {'PENDING': 1, 'PROCESSING': 1, 'COMPLETED': 7, 'FAILED': 14}
    new_ttl = current_time + (ttl_days[status] * 86400)
    update_expr += ', #ttl = :ttl'
    expr_attr_names['#ttl'] = 'ttl'
    expr_attr_values[':ttl'] = new_ttl

    # Add optional attributes
    for key, value in kwargs.items():
        update_expr += f', #{key} = :{key}'
        expr_attr_names[f'#{key}'] = key
        expr_attr_values[f':{key}'] = value

    table.update_item(
        Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_attr_names,
        ExpressionAttributeValues=expr_attr_values
    )

    logger.info(f"Updated job {job_id} to status {status}")

@tracer.capture_method
def process_text_analysis(data):
    """
    Perform text analysis
    This is a placeholder - implement actual clustering logic
    """
    # Import your text analysis modules here
    # from text_analyzer import cluster_sentences, analyze_sentiment

    baseline = data['baseline']

    # Simulate processing time
    time.sleep(2)

    # Placeholder result
    result = {
        'clusters': [
            {
                'title': 'Example Cluster',
                'sentiment': 'positive',
                'sentences': [s['id'] for s in baseline[:3]],
                'keyInsights': [
                    'This is a placeholder insight',
                    'Actual implementation would use ML models'
                ]
            }
        ]
    }

    return result

@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    """Process async text analysis job"""
    job_id = event['jobId']
    data = event['data']

    logger.info(f"Processing job {job_id}")

    try:
        # Update status to PROCESSING
        update_job_status(
            job_id,
            'PROCESSING',
            startedAt=int(time.time())
        )

        # Perform text analysis
        result = process_text_analysis(data)

        # Update status to COMPLETED with results
        update_job_status(
            job_id,
            'COMPLETED',
            result=result,
            completedAt=int(time.time()),
            duration=int(time.time()) - event.get('startedAt', int(time.time()))
        )

        logger.info(f"Successfully completed job {job_id}")

        return {
            'statusCode': 200,
            'jobId': job_id,
            'status': 'COMPLETED'
        }

    except Exception as e:
        logger.exception(f"Error processing job {job_id}")

        # Update status to FAILED
        update_job_status(
            job_id,
            'FAILED',
            error={
                'type': type(e).__name__,
                'message': str(e),
                'retriable': False
            },
            failedAt=int(time.time())
        )

        # Re-raise to trigger Lambda Destination
        raise
```

#### Status Lambda (src/lambdas/status/index.py)

```python
import json
import os
import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver

logger = Logger()
tracer = Tracer()
app = APIGatewayRestResolver()

dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

@app.get("/jobs/<job_id>")
@tracer.capture_method
def get_job_status(job_id: str):
    """Get job status and results"""
    try:
        response = table.get_item(
            Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'}
        )

        if 'Item' not in response:
            return {
                'statusCode': 404,
                'body': {
                    'error': 'NotFound',
                    'message': 'Job not found or expired'
                }
            }

        item = response['Item']

        # Build response based on status
        result = {
            'jobId': job_id,
            'status': item['status'],
            'createdAt': item['createdAt']
        }

        # Add status-specific fields
        if item['status'] == 'PROCESSING':
            result['startedAt'] = item.get('startedAt')

        elif item['status'] == 'COMPLETED':
            result['completedAt'] = item.get('completedAt')
            result['duration'] = item.get('duration')
            result['result'] = item.get('result')

        elif item['status'] == 'FAILED':
            result['failedAt'] = item.get('failedAt')
            result['error'] = item.get('error')

        return {
            'statusCode': 200,
            'body': result
        }

    except Exception as e:
        logger.exception(f"Error retrieving job {job_id}")
        return {
            'statusCode': 500,
            'body': {
                'error': 'InternalError',
                'message': 'Failed to retrieve job status'
            }
        }

@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    return app.resolve(event, context)
```

#### Error Handler Lambda (src/lambdas/error-handler/index.py)

```python
import json
import os
import time
import boto3
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()

dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

@tracer.capture_method
def is_retriable_error(error_type):
    """Determine if error is retriable"""
    retriable_errors = [
        'TimeoutError',
        'ThrottlingException',
        'ServiceUnavailable'
    ]
    return error_type in retriable_errors

@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    """Handle failed async job invocations from Lambda Destination"""

    for record in event['Records']:
        try:
            # Parse Destination payload
            body = json.loads(record['body'])

            # Extract job info from request payload
            request_payload = body.get('requestPayload', {})
            job_id = request_payload.get('jobId')

            if not job_id:
                logger.warning("No jobId in failed invocation record")
                continue

            # Extract error info from response payload
            response_payload = body.get('responsePayload', {})
            error_type = response_payload.get('errorType', 'Unknown')
            error_message = response_payload.get('errorMessage', 'Unknown error')

            logger.info(f"Processing failed job {job_id}: {error_type}")

            # Update job status in DynamoDB
            current_time = int(time.time())
            table.update_item(
                Key={'PK': f'JOB#{job_id}', 'SK': 'METADATA'},
                UpdateExpression='SET #status = :status, #error = :error, '
                                '#failedAt = :failedAt, GSI1PK = :gsi1pk, GSI1SK = :gsi1sk, '
                                '#ttl = :ttl',
                ExpressionAttributeNames={
                    '#status': 'status',
                    '#error': 'error',
                    '#failedAt': 'failedAt',
                    '#ttl': 'ttl'
                },
                ExpressionAttributeValues={
                    ':status': 'FAILED',
                    ':error': {
                        'type': error_type,
                        'message': error_message,
                        'retriable': is_retriable_error(error_type)
                    },
                    ':failedAt': current_time,
                    ':gsi1pk': 'STATUS#FAILED',
                    ':gsi1sk': str(current_time),
                    ':ttl': current_time + (14 * 86400)  # 14 days for failed jobs
                }
            )

            logger.info(f"Updated job {job_id} status to FAILED")

        except Exception as e:
            logger.exception("Error processing failed job record")
            # Re-raise to move to DLQ
            raise

    return {
        'statusCode': 200,
        'body': json.dumps('Processed failed jobs')
    }
```

---

## 10. Best Practices and Recommendations

### Architecture Best Practices

1. **Use Proxy Lambda Pattern for HTTP APIs**
   - HTTP APIs only support proxy integration
   - Gives full control over job ID generation and validation
   - Clean separation of concerns (proxy vs worker)

2. **Implement Idempotency**
   - Accept client-provided idempotency keys
   - Check DynamoDB before creating duplicate jobs
   - Return existing job ID if duplicate request

3. **Instrument for Observability**
   - Use AWS Lambda Powertools for structured logging
   - Enable X-Ray tracing for distributed tracing
   - Track key metrics (job duration, failure rate, queue depth)
   - Set CloudWatch alarms for critical errors

4. **Design for Failure**
   - All jobs should have timeout limits
   - Implement circuit breakers for external dependencies
   - Use Lambda Destinations for failure handling
   - Store failed job details for debugging

### DynamoDB Best Practices

1. **Use On-Demand Billing**
   - Simpler than provisioned capacity
   - Auto-scales with traffic
   - 50% cheaper since November 2024

2. **Enable Point-in-Time Recovery**
   - Protects against accidental deletions
   - Enables restore to any point in last 35 days

3. **Use Composite Keys**
   - PK: `JOB#{job_id}` or `USER#{user_id}#JOB#{job_id}`
   - SK: `METADATA`, `RESULT`, `STATUS#{timestamp}`
   - Enables flexible querying and data organization

4. **Leverage GSI for Queries**
   - Query by status, time range, user
   - Use sparse indexes (only items with GSI attributes)
   - Composite GSI keys prevent hot partitions

5. **Implement TTL for Cleanup**
   - Different TTL based on job status
   - Use DynamoDB Streams to capture deletions
   - Archive to S3 before deletion if needed

### Lambda Best Practices

1. **Right-Size Memory**
   - Proxy Lambda: 512 MB (lightweight validation)
   - Worker Lambda: 3008 MB (CPU-intensive ML processing)
   - Status Lambda: 256 MB (simple reads)

2. **Set Appropriate Timeouts**
   - Proxy Lambda: 30 seconds (quick validation + invoke)
   - Worker Lambda: 15 minutes (max for complex processing)
   - Status Lambda: 10 seconds (simple DynamoDB read)

3. **Use Reserved Concurrency**
   - Prevent runaway scaling costs
   - Protect downstream services from overload
   - Set based on expected peak load

4. **Implement Structured Logging**
   ```python
   from aws_lambda_powertools import Logger

   logger = Logger()

   logger.info("Processing job", extra={
       "job_id": job_id,
       "status": "processing",
       "sentence_count": len(data['baseline'])
   })
   ```

5. **Use Lambda Layers**
   - Share common dependencies across functions
   - Faster deployments (don't re-upload dependencies)
   - Separate code from dependencies

### Security Best Practices

1. **Apply Least Privilege IAM**
   - Grant only required permissions
   - Use resource-level permissions where possible
   - Separate read and write permissions

2. **Enable API Key or Cognito**
   - Prevent unauthorized access
   - Track usage by API key
   - Rate limit by key

3. **Validate All Inputs**
   - Schema validation in proxy Lambda
   - Sanitize user inputs
   - Reject oversized payloads early

4. **Encrypt Sensitive Data**
   - DynamoDB encryption at rest (enabled by default)
   - Use AWS KMS for additional encryption
   - Don't log sensitive data

### Cost Optimization

1. **Monitor and Alert on Costs**
   - Set budget alerts
   - Track Lambda duration and invocations
   - Monitor DynamoDB capacity usage

2. **Optimize Lambda Performance**
   - Faster execution = lower costs
   - Use compiled languages for CPU-intensive tasks
   - Cache static data in /tmp

3. **Use DynamoDB Efficiently**
   - Batch reads/writes where possible
   - Project only needed attributes in queries
   - Use DynamoDB Accelerator (DAX) for hot data

4. **Implement Exponential Backoff**
   - Reduce polling frequency over time
   - Clients shouldn't poll every second
   - Start at 2s, increase to 30s max

### Testing Strategy

1. **Unit Tests**
   - Test validation logic
   - Test status transitions
   - Mock DynamoDB and Lambda clients

2. **Integration Tests**
   - Test full flow (create → process → status)
   - Use LocalStack or DynamoDB Local
   - Test error scenarios

3. **Load Tests**
   - Simulate concurrent job submissions
   - Test Lambda concurrency limits
   - Monitor DynamoDB throttling

4. **Chaos Engineering**
   - Test Lambda timeout scenarios
   - Simulate DynamoDB unavailability
   - Test retry and failure paths

### Monitoring and Alerting

```python
# Key Metrics to Track

# Lambda Metrics
- Duration (p50, p95, p99)
- Errors and throttles
- Concurrent executions
- Cold starts

# DynamoDB Metrics
- Consumed read/write capacity
- Throttled requests
- System errors
- Conditional check failures

# Business Metrics
- Jobs created per minute
- Jobs completed per minute
- Average job duration
- Job failure rate by error type
- Queue depth (PENDING jobs)

# Alarms to Set
- Worker Lambda errors > 10 in 5 minutes
- Failed jobs queue depth > 50
- Average job duration > 5 minutes
- DynamoDB throttled requests > 0
- API Gateway 5xx errors > 1%
```

### Recommendations for Text Analysis Microservice

1. **Storage:** DynamoDB (fast queries, TTL, streams)
2. **Invocation:** Proxy Lambda pattern (works with HTTP API)
3. **Error Handling:** Lambda Destinations to SQS
4. **Cleanup:** TTL with status-based retention (1-14 days)
5. **Orchestration:** Direct Lambda async (no Step Functions needed)
6. **Monitoring:** CloudWatch metrics + alarms + X-Ray tracing

### Future Enhancements

1. **WebSocket for Real-Time Updates**
   - Replace polling with push notifications
   - Use API Gateway WebSocket API
   - Lambda publishes status updates to connected clients

2. **Batch Job Processing**
   - Accept multiple jobs in single request
   - Process in parallel with fan-out pattern
   - Aggregate results

3. **Job Prioritization**
   - Add priority field to jobs
   - Use SQS FIFO queues for ordering
   - Separate Lambda pools for priority levels

4. **Result Caching**
   - Cache results in ElastiCache/DAX
   - Return cached results for duplicate requests
   - Reduce DynamoDB costs for hot data

5. **Multi-Region Deployment**
   - DynamoDB Global Tables for replication
   - Route 53 for geographic routing
   - Cross-region Lambda invocations

---

## Conclusion

The recommended architecture for async job processing with API Gateway and Lambda uses:

**Core Pattern:**
```
Client → API Gateway → Proxy Lambda → Worker Lambda (async)
                            ↓              ↓
                       DynamoDB     Lambda Destination (SQS)
                                            ↓
                                    Error Handler Lambda
```

**Key Decisions:**
- ✅ DynamoDB for storage (not S3)
- ✅ Proxy Lambda pattern (not X-Amz-Invocation-Type)
- ✅ Lambda Destinations (not DLQ)
- ✅ Direct async invocation (not Step Functions)
- ✅ Status-based TTL for cleanup

This architecture provides:
- Low latency job submission (<100ms)
- Reliable async processing (up to 15 minutes)
- Automatic retries and error handling
- Cost-effective at scale
- Simple to implement and maintain
- Production-ready monitoring and observability

**Total Cost Estimate (1M jobs/month):**
- API Gateway: ~$3.50
- Lambda (proxy + worker + error): ~$2.00
- DynamoDB (on-demand): ~$2.50
- SQS (failed jobs): ~$0.10
- CloudWatch Logs: ~$1.00
- **Total: ~$9.00/month for 1M jobs**

This research summary provides comprehensive guidance for implementing production-grade async job processing on AWS Lambda with API Gateway.

---

## References

- [AWS Lambda Async Invocation Documentation](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html)
- [API Gateway Lambda Integration](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-integration-async.html)
- [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)
- [Lambda Destinations](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-retain-records.html)
- [DynamoDB TTL](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html)
- [AWS Prescriptive Guidance - Async Processing](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/process-events-asynchronously-with-amazon-api-gateway-and-aws-lambda.html)

**Research Completed:** October 6, 2025
**Last Updated:** October 6, 2025
