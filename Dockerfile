FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install build dependencies for packages that need compilation
RUN microdnf install -y gcc gcc-c++ python3-devel && microdnf clean all

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.handler"]
