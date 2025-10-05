# CloudWatch Logs Monitoring Guide

This guide provides sample commands to check AWS CloudWatch logs for the Lambda function.

## Quick Reference

### Lambda Function Name

```bash
text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug
```

### Log Group Name

```bash
/aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug
```

## Basic Log Commands

### 1. Tail Recent Logs (Live)

```bash
# Follow logs in real-time (like tail -f)
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug --follow
```

### 2. View Last 5 Minutes

```bash
# Show logs from last 5 minutes
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug --since 5m --format short
```

### 3. View Last Hour

```bash
# Show logs from last hour
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug --since 1h --format short
```

### 4. View Specific Time Range

```bash
# Logs from specific start time
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since '2025-10-05T10:00:00' \
  --until '2025-10-05T11:00:00' \
  --format short
```

## Filtered Log Searches

### 1. Find Errors Only

```bash
# Show only ERROR level logs
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep ERROR
```

### 2. Find Successful Completions

```bash
# Show successful request completions
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "completed successfully"
```

### 3. Find Cold Starts

```bash
# Show cold start initializations
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "(cold start|Initializing ML)"
```

### 4. Find Validation Errors

```bash
# Show validation failures
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "Validation failed"
```

### 5. Find Performance Metrics

```bash
# Show request durations and performance
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "(Duration:|REPORT)"
```

## Advanced Filtering

### 1. Filter by Request ID

```bash
# Track specific request through logs
REQUEST_ID="abc-123-def"
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "$REQUEST_ID"
```

### 2. Extract Cold Start Times

```bash
# Get all cold start initialization times
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 6h --format short | grep "ML components initialized in"
```

### 3. Find Cluster Analysis Results

```bash
# Show clustering results
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "clusters found"
```

### 4. Monitor Memory Usage

```bash
# Extract memory usage reports
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "Max Memory Used"
```

## Performance Analysis

### 1. Request Duration Summary

```bash
# Get duration for all recent requests
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "REPORT RequestId" | \
  awk '{print $8}' | sort -n
```

### 2. Cold vs Warm Start Comparison

```bash
# Show initialization times
echo "=== Cold Starts ==="
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 6h --format short | grep "Initializing ML components (cold start)"

echo -e "\n=== Warm Starts ==="
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep "Using cached ML components"
```

### 3. Average Processing Time

```bash
# Calculate average duration (requires jq)
aws logs filter-log-events \
  --log-group-name /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --start-time $(date -u -d '1 hour ago' +%s)000 \
  --filter-pattern "REPORT" \
  --query 'events[*].message' \
  --output text | \
  grep -oP 'Duration: \K[0-9.]+' | \
  awk '{sum+=$1; count++} END {print "Average Duration:", sum/count, "ms"}'
```

## Error Investigation

### 1. Recent Errors with Context

```bash
# Show errors with 5 lines of context
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 30m --format short | grep -A 5 -B 5 ERROR
```

### 2. Stack Traces

```bash
# Find Python stack traces
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -A 10 "Traceback"
```

### 3. Validation Errors Detail

```bash
# Get full validation error messages
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -A 3 "Validation failed"
```

### 4. Import Errors

```bash
# Check for module import issues
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -i "import\|ModuleNotFoundError"
```

## CloudWatch Insights Queries

### 1. Query Builder - Request Count

```bash
# Use CloudWatch Insights to count requests
aws logs start-query \
  --log-group-name /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --start-time $(date -u -d '1 hour ago' +%s) \
  --end-time $(date -u +%s) \
  --query-string 'fields @timestamp, @message | filter @message like /Processing request/ | count'
```

### 2. Query Error Rate

```bash
# Calculate error rate
aws logs start-query \
  --log-group-name /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --start-time $(date -u -d '1 hour ago' +%s) \
  --end-time $(date -u +%s) \
  --query-string 'fields @timestamp | stats count(*) as total, count(@message like /ERROR/) as errors | fields errors * 100 / total as error_rate'
```

## Log Streaming

### 1. Stream Logs to File

```bash
# Save logs to file while following
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --follow --format short | tee lambda_logs_$(date +%Y%m%d_%H%M%S).log
```

### 2. Stream Only Errors to File

```bash
# Save only errors
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --follow --format short | grep ERROR | tee errors_$(date +%Y%m%d_%H%M%S).log
```

## Debugging Specific Issues

### 1. Timeout Investigation

```bash
# Find requests that timed out
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "Task timed out|timeout"
```

### 2. Memory Issues

```bash
# Check if function is running out of memory
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "out of memory|MemoryError"
```

### 3. Model Loading Issues

```bash
# Check model loading process
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "Loading embedding model|Model loaded"
```

### 4. Clustering Failures

```bash
# Find HDBSCAN/clustering errors
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --since 1h --format short | grep -E "HDBSCAN failed|Clustering|KMeans"
```

## Quick Diagnostic Script

Save as `check_logs.sh`:

```bash
#!/bin/bash
LOG_GROUP="/aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug"

echo "=== Lambda CloudWatch Logs Diagnostic ==="
echo "Time: $(date)"
echo ""

echo "1. Recent Errors (last 30 minutes):"
aws logs tail $LOG_GROUP --since 30m --format short | grep ERROR | tail -10
echo ""

echo "2. Recent Successes (last 30 minutes):"
aws logs tail $LOG_GROUP --since 30m --format short | grep "completed successfully" | tail -5
echo ""

echo "3. Performance (last hour):"
aws logs tail $LOG_GROUP --since 1h --format short | grep "REPORT" | tail -5
echo ""

echo "4. Cold Starts (last 6 hours):"
aws logs tail $LOG_GROUP --since 6h --format short | grep "cold start" | wc -l
echo ""

echo "5. Warm Starts (last hour):"
aws logs tail $LOG_GROUP --since 1h --format short | grep "Using cached" | wc -l
echo ""

echo "=== End Diagnostic ==="
```

Run with: `chmod +x check_logs.sh && ./check_logs.sh`

## Real-Time Monitoring

### Live Error Monitor

```bash
# Watch for errors in real-time
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --follow --format short --filter-pattern "?ERROR ?WARN ?Exception"
```

### Live Request Monitor

```bash
# Watch all incoming requests
aws logs tail /aws/lambda/text-analysis-prod-TextAnalysisFunctionD080EBE8-oRT1IeZUOYug \
  --follow --format short --filter-pattern "Processing request"
```

## Log Retention

Current retention: **7 days** (configured in CDK stack)

To view retention settings:

```bash
aws logs describe-log-groups \
  --log-group-name-prefix /aws/lambda/text-analysis-prod \
  --query 'logGroups[*].[logGroupName,retentionInDays]' \
  --output table
```

## Useful Log Patterns

Common log messages to look for:

- **Cold Start**: `Initializing ML components (cold start)...`
- **Warm Start**: `Using cached ML components (warm start)`
- **Success**: `Request {id} completed successfully in {time}s`
- **Error**: `Request {id} failed after {time}s: {error}`
- **Validation**: `Validation failed: {details}`
- **Duration**: `REPORT RequestId: {id} Duration: {ms} ms`
- **Memory**: `Max Memory Used: {mb} MB`
- **Init**: `Init Duration: {ms} ms` (for cold starts)
