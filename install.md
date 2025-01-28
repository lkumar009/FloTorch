# FloTorch CloudFormation Templates

This directory contains AWS CloudFormation templates for deploying the FloTorch infrastructure. The templates create a comprehensive architecture for managing and deploying machine learning models, combining serverless components with managed services like OpenSearch.

## Subscription Steps

Before deploying the FloTorch infrastructure, you need to subscribe to the FloTorch product on AWS Marketplace:

1. Visit the [FloTorch AWS Marketplace page](https://aws.amazon.com/marketplace/pp/prodview-z5zcvloh7l3ky?ref_=aws-mp-console-subscription-detail-payg).
2. Click on the "View Purchase options" button and once its shown that you've subscribed, then click on 'Continue to Configuration' button.
3. The select the fulfilment option and the Software Version you want to use

The CloudFormation templates in this directory will automatically use your FloTorch subscription to set up the required resources and services.

## Architecture Overview

The infrastructure consists of several key components:

- **VPC Stack**: Private network infrastructure
- **VPC Endpoint Stack**: Secure access to AWS services within the VPC
- **DynamoDB Stack**: Database tables and S3 bucket for data storage
- **OpenSearch Stack**: Search functionality
- **ECR Repository Stack**: Container image repositories
- **ECS Stack**: Elastic Container Service for running containerized applications
- **Lambda Stack**: Serverless functions for various operations
- **State Machine Stack**: Step Functions for orchestration
- **AppRunner Stack**: Application hosting and management

### Key Features

1. Automated deployment and management
2. Serverless architecture for cost optimization
3. Secure VPC configuration
4. Integrated monitoring and logging
5. Automated container image management

## Prerequisites

1. AWS Account with administrative access
2. AWS CLI installed and configured
3. Sufficient AWS service quotas for:
   - VPC and related networking resources
   - Lambda functions
   - EventBridge rules
   - SageMaker endpoints
   - OpenSearch domains
   - DynamoDB tables
   - ECR repositories
   - AppRunner services
4. Bedrock access to the following models:
   - Embedding models:
     - Amazon/amazon.titan-embed-text-v2:0
     - Amazon/amazon.titan-embed-image-v1
     - Cohere/cohere.embed-english-v3
     - Cohere/cohere.embed-multilingual-v3
   - Retrieval models:
     - Amazon/amazon.titan-text-lite-v1
     - Amazon/amazon.titan-text-express-v1
     - Amazon/amazon.nova-lite-v1:0
     - Amazon/amazon.nova-micro-v1:0
     - Amazon/amazon.nova-pro-v1:0
     - Anthropic/anthropic.claude-3-5-sonnet-20241022-v2:0
     - Anthropic/anthropic.claude-3-5-sonnet-20240620-v1:0
     - Cohere/cohere.command-r-plus-v1:0
     - Cohere/cohere.command-r-v1:0
     - Meta/meta.llama3-2-1b-instruct-v1:0
     - Meta/meta.llama3-2-3b-instruct-v1:0
     - Meta/meta.llama3-2-11b-instruct-v1:0
     - Meta/meta.llama3-2-90b-instruct-v1:0
     - Mistral AI/mistral.mistral-7b-instruct-v0:2
     - Mistral AI/mistral.mistral-large-2402-v1:0

## Required Permissions

The user doing the deployment requires administrative access(recommended) or the following AWS permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudformation:*",
                "s3:*",
                "ec2:*",
                "iam:*",
                "lambda:*",
                "dynamodb:*",
                "events:*",
                "sagemaker:*",
                "opensearch:*",
                "ecr:*",
                "apprunner:*",
                "cloudwatch:*",
                "logs:*",
                "ssm:*",
                "es:*"
            ],
            "Resource": "*"
        }
    ]
}
```

## Getting Started

1. Click on the [master template (us-east-1)](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create?stackName=flotorch-stack&templateURL=https://flotorch-public.s3.us-east-1.amazonaws.com/2.0.1/templates/master-template.yaml) to open the CloudFormation console with the template pre-loaded.

2. In the CloudFormation console, fill in the parameters:
   - ProjectName: your-project-name
     Description: Name of the project (e.g., "flotorch")
   - TableSuffix: unique-suffix
     Description: A unique 6-character lowercase suffix to append to resource names (e.g., "abcdef")
   - ClientName: your-client-name
     Description: Lowercase name of the client (e.g., "acmecorp")
   - OpenSearchAdminUser: admin
     Description: Username for OpenSearch admin access
   - OpenSearchAdminPassword: YourSecurePassword123!
     Description: Secure password for OpenSearch admin (8-41 characters, including letters, numbers, and symbols)
   - NginxAuthPassword: YourNginxPassword123!
     Description: Password for NGINX basic authentication (8-41 characters, including letters, numbers, and symbols)

3. Review and create the stack, acknowledging that it may create IAM resources.

Alternatively, you can deploy using the AWS CLI:

```bash
aws cloudformation create-stack \
    --stack-name flotorch-stack \
    --template-url https://flotorch-public.s3.us-east-1.amazonaws.com/templates/master-template.yaml \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
    --parameters \
        ParameterKey=ProjectName,ParameterValue=your-project-name \
        ParameterKey=TableSuffix,ParameterValue=unique-suffix \
        ParameterKey=ClientName,ParameterValue=your-client-name \
        ParameterKey=OpenSearchAdminUser,ParameterValue=admin \
        ParameterKey=OpenSearchAdminPassword,ParameterValue=YourSecurePassword123! \
        ParameterKey=NginxAuthPassword,ParameterValue=YourNginxPassword123!
```

## Template Details

### master-template.yaml
- Main template that orchestrates all other stacks
- Manages dependencies between stacks
- Handles global parameters

### lambda-template.yaml
- Contains Lambda functions for:
  - SageMaker endpoint cleanup
  - Docker image management
  - Runtime operations
  - Cost computation

### vpc-template.yaml
- Creates VPC and associated networking resources
- Sets up public and private subnets
- Configures route tables and internet gateway

### dynamodb-template.yaml
- Sets up DynamoDB tables for data storage
- Configures auto-scaling for tables
- Implements backup and recovery options

### opensearch-template.yaml
- Deploys OpenSearch domain
- Configures access policies and encryption
- Sets up index templates and mappings

### ecr-template.yaml
- Creates Elastic Container Registry repositories
- Sets up lifecycle policies for images
- Configures cross-account access if needed

### apprunner-template.yaml
- Deploys AppRunner services
- Configures auto-scaling and health checks
- Sets up custom domains and SSL certificates

## Cost Estimation (per day)

Note: These are approximate costs based on AWS pricing in the us-east-1 region. Actual costs may vary.

1. **Lambda Functions**:
   - Free tier: 1M requests/month and 400,000 GB-seconds of compute time
   - Estimated cost: $0.50-$2.00/day (depending on usage and memory allocation)

2. **SageMaker Endpoints**:
   - ml.t3.medium instance: $0.0582/hour
   - Estimated cost: $1.40-$2.80/day (assuming 1-2 endpoints running)

3. **OpenSearch**:
   - r7g.2xlarge instance: $0.5184/hour
   - Provisioned IOPS (16000): $1.152/hour
   - Provisioned Throughput (1000 MiB/s): $0.072/hour
   - Estimated cost: $41.81/day

4. **DynamoDB**:
   - On-demand pricing: $1.25 per million write request units, $0.25 per million read request units
   - Storage: $0.25 per GB per month
   - Estimated cost: $2-$5/day (depending on read/write operations and data stored)

5. **AppRunner**:
   - Compute: $0.064/hour for 1 vCPU, 2GB memory
   - Estimated cost: $1.54-$3.08/day (assuming 1-2 instances)

6. **ECR**:
   - Storage: $0.10/GB/month
   - Data transfer: $0.09/GB (out to internet)
   - Estimated cost: $0.20-$0.50/day

7. **VPC and Networking**:
   - NAT Gateway: $0.045/hour, $0.045/GB data processed
   - Estimated cost: $1.08-$2.00/day

8. **CloudWatch**:
   - $0.30 per GB ingested for logs
   - $0.01 per 1,000 metrics requested
   - Estimated cost: $0.50-$1.50/day

9. **Step Functions**:
   - $0.025 per 1,000 state transitions
   - Estimated cost: $0.50-$2.00/day (depending on workflow complexity and frequency)

10. **ECS (Fargate)**:
    - $0.04048 per vCPU-hour
    - $0.004445 per GB-hour
    - Estimated cost: $2.00-$5.00/day (depending on task size and runtime)

11. **Bedrock**:
    - Input tokens: $0.0001 per 1K tokens
    - Output tokens: $0.0002 per 1K tokens
    - Estimated cost: $5.00-$10.00/day (assuming 25M-50M input tokens and 12.5M-25M output tokens)

**Total Estimated Daily Cost**: $56.53-$75.69/day
(Costs can vary significantly based on usage patterns and data volumes)

## Monitoring and Maintenance

1. **CloudWatch Metrics**:
   - Monitor Lambda execution
   - Track SageMaker endpoint usage
   - AppRunner service metrics

2. **Logs**:
   - Lambda function logs
   - AppRunner application logs
   - OpenSearch logs

3. **Alerts**:
   - Cost thresholds
   - Error rates
   - Service health

## Security Features

1. VPC isolation
2. IAM role-based access
3. Encryption at rest
4. Network security groups
5. NGINX basic authentication
6. OpenSearch access control

## Troubleshooting

1. **Stack Creation Failures**:
   - Check CloudFormation events
   - Verify IAM permissions
   - Validate parameter values

2. **Runtime Issues**:
   - Check CloudWatch logs
   - Verify network connectivity
   - Check service quotas

## Support

For issues and support:
1. Check CloudWatch logs
2. Review stack events
3. Contact DevOps team
4. Reach out to us at info@flotorch.ai

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
