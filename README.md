# Backend Take-Home Task: Text Analysis Microservice

## Overview

You will build a serverless microservice that analyzes text data and groups similar sentences into thematic clusters. This task mimics a real production system we recently deployed - a text analysis service that processes customer feedback and generates actionable insights.

Your task is to spend no more than 4 hours approaching a solution to the task below. **It is not expected (or possible?) to complete this to a high standard in this time**. The intention is to see how you approach the problem and what you value. You should keep this in mind and make sure you are able to talk about various elements of the task, why you chose certain approaches and what would be necessary to progress past the point you are able to achieve in the time.

**Note:** while this task includes an element of using AI in the solution, this isn't the primary thing we will be talking about. Please don't focus too heavily on getting great quality results from a model, we are more interested in how you would access and maintain use of a model.

While we are unable to pay you for your time on this task, there will be a small recompense.

## Submission Guidelines

Please submit a git repository with your solution.

## Questions?

If you have questions about requirements or need clarification, please reach out. We prefer candidates who ask thoughtful questions rather than make assumptions.

Good luck! We're excited to see your approach to this real-world problem.

## Service description

The service should:

- Accept structured text input via API
- Perform intelligent clustering of sentences by theme/topic
- Return organized clusters with sentiment analysis and key insights
- Handle both standalone analysis and comparative analysis scenarios

## Input Format

There are example files in the data directory.

Your service will receive JSON input in one of two formats:

### Standalone Analysis

```json
{
  "surveyTitle": "Robinhood App Store",
  "theme": "account",
  "baseline": [
    {
      "sentence": "Example sentence text",
      "id": "unique-sentence-id"
    }
  ]
}
```

Fields

- surveyTitle: the title of the dataset the sentences came from
- theme: the theme that all provided sentences are about
- baseline: an array of sentences and ids

## Expected Output

### Standalone Analysis Output

```json
{
  "clusters": [
    {
      "title": "Specific Sub theme Title",
      "theme": "specific nuanced aspect",
      "sentiment": "positive|negative|neutral",
      "sentences": ["unique-sentence-id", "unique-sentence-id-2"],
      "keyInsights": [
        "specific **insight1**",
        "specific insight2 **bolded here**",
        "**specific insight3** here"
      ]
    }
  ]
}
```

Fields:

- clusters: an array of interesting clusters
  - title: a good, short name for the cluster
  - sentiment: whether this cluster represents positive, negative or neutral sentiment
  - sentences: an array of sentence ids (taken from the input)
  - keyInsights: 2-3 sentences (markdown allowed) to display in bullet points

## Technical Requirements

### Core Requirements

- **Language**: We use Python so this should be considered
- **AWS Lambda**: Deploy as a serverless function
- **API Gateway**: RESTful endpoint for receiving requests
- **Text Processing**: Intelligent sentence clustering and theme identification
- **Sentiment Analysis**: Classify sentiment for each cluster
- **Error Handling**: Robust error responses and input validation

### Architecture Considerations

- **Scalability**: Handle varying input sizes efficiently
- **Performance**: Optimize for sub-10 second response times
- **Cost**: Consider AWS Lambda pricing and execution time
- **Security**: Implement proper authentication/authorization

## Sample Data

Example input files are provided in the `data/` directory:

- `input_example.json` - Standalone analysis format
- `input_comparison_example.json` - Comparative analysis format

These contain real-world text samples you can use for testing and development.

## What We're Evaluating

Below are some of the things that we are intending to look at or discuss

### Architecture & Infrastructure

- AWS service selection and configuration
- Infrastructure as Code (CloudFormation, CDK, Terraform, etc.)
- Security best practices
- Scalability considerations

### Testing Strategy

- Unit test coverage and quality
- Integration testing approach
- Performance testing methodology
- Test data management

### Deployment & DevOps

- CI/CD pipeline setup
- Environment management
- Monitoring and logging
- Documentation of deployment process

### Code Quality

- Clean, readable, maintainable code
- Proper error handling
- Code organization and structure
- Documentation and comments

### ML/AI Implementation

- Text clustering approach and effectiveness
- Sentiment analysis accuracy
- Handling of edge cases
- Performance optimization

## Getting Started

1. **Analyze the Problem**: Review the input/output formats and understand the clustering requirements

2. **Choose Your Tech Stack**:

   - Programming language (Python, Node.js, Java, etc.)
   - Text processing libraries (spaCy, NLTK, transformers, etc.)
   - AWS services (Lambda, API Gateway, S3, etc.)

3. **Design Your Architecture**:

   - Sketch out your AWS infrastructure
   - Plan your data flow and processing pipeline
   - Consider how you'll handle different input sizes

4. **Implement Core Logic**:

   - Text preprocessing and cleaning
   - Sentence clustering algorithm
   - Sentiment analysis
   - Output formatting

5. **Add Infrastructure**:

   - Set up AWS resources
   - Configure API Gateway
   - Implement deployment automation

6. **Testing & Validation**:
   - Test with provided sample data
   - Add comprehensive test coverage
   - Performance testing

## AI Tool Usage

**We encourage and expect you to use AI tools** (ChatGPT, Claude, Copilot, etc.) to accelerate development. However, we want to see:

- **Your architectural decisions** and reasoning
- **Your testing strategy** and implementation
- **Your infrastructure choices** and configuration
- **Your problem-solving approach** when AI suggestions don't work

Please document where and how you used AI tools in your submission.

---

**Note**: This task mirrors actual production systems we maintain. Your solution should demonstrate production-ready thinking, even if not every feature is fully implemented.

# Extra

In reality this system is used for comparing data as well as analysis of a single query. When designing this is something that would be good to take into account.

### Comparative Analysis Input

```json
{
  "surveyTitle": "Robinhood App Store",
  "theme": "account",
  "baseline": [
    {
      "sentence": "Example baseline sentence",
      "id": "unique-sentence-id"
    }
  ],
  "comparison": [
    {
      "sentence": "Example comparison sentence",
      "id": "unique-sentence-id"
    }
  ]
}
```

Fields

- surveyTitle: the title of the dataset the sentences came from
- theme: the theme that all provided sentences are about
- baseline: an array of sentences and ids

### Comparative Analysis Output

```json
{
  "clusters": [
    {
      "title": "Specific Sub theme Title",
      "sentiment": "positive|negative|neutral",
      "baselineSentences": ["unique-sentence-id"],
      "comparisonSentences": ["unique-sentence-id-2"],
      "keySimilarities": [
        "specific **insight1**",
        "specific insight2 **bolded here**",
        "**specific insight3** here"
      ],
      "keyDifferences": [
        "specific **insight1**",
        "specific insight2 **bolded here**",
        "**specific insight3** here"
      ]
    }
  ]
}
```

- clusters: an array of interesting clusters
  - title: a good, short name for the cluster
  - sentiment: whether this cluster represents positive, negative or neutral sentiment
  - baselineSentences: an array of sentence ids (taken from the baseline input) for sentences related to this cluster
  - comparisonSentences: an array of sentence ids (taken from the comparison input) for sentences related to this cluster
  - keySimilarities: 2-3 sentences (markdown allowed) to display in bullet points that describe similarities between the two sources
  - keyDifferences: 2-3 sentences (markdown allowed) to display in bullet points that describe the differences between the two sources
