# ============================================================================
# AWS INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
# For 8+ Years SDE / AWS Engineer Experience
# Focus: Microservices Architecture on AWS
# Services: VPC, EFS, EBS, ELB, Route53, ECS, S3, DynamoDB, Lambda,
#           SQS, Step Functions, SNS, ECR, CloudWatch, X-Ray, CloudFormation
# ============================================================================

# ============================================================================
# SECTION 1: VPC — Virtual Private Cloud
# ============================================================================
# KEY CONCEPTS TO KNOW:
#   - CIDR blocks, subnets (public vs private), route tables
#   - Internet Gateway vs NAT Gateway
#   - Security Groups (stateful) vs NACLs (stateless)
#   - VPC Peering vs Transit Gateway vs PrivateLink
#   - VPC Endpoints (Gateway vs Interface)
#   - Flow Logs for network monitoring
#   - Bastion Hosts / Jump Servers
#   - Multi-AZ design for HA microservices
# ============================================================================

import boto3
import json

# --- 1.1 Create a VPC with public and private subnets ---
def create_vpc_with_subnets():
    ec2 = boto3.client('ec2', region_name='us-east-1')

    # Create VPC
    vpc = ec2.create_vpc(CidrBlock='10.0.0.0/16')
    vpc_id = vpc['Vpc']['VpcId']
    ec2.create_tags(Resources=[vpc_id], Tags=[{'Key': 'Name', 'Value': 'microservice-vpc'}])

    # Enable DNS hostnames — required for ECS/EFS service discovery
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})

    # Public subnet (hosts ALB, NAT Gateway, Bastion)
    pub_subnet = ec2.create_subnet(VpcId=vpc_id, CidrBlock='10.0.1.0/24', AvailabilityZone='us-east-1a')
    pub_subnet_id = pub_subnet['Subnet']['SubnetId']

    # Private subnet (hosts ECS tasks, Lambda, RDS)
    priv_subnet = ec2.create_subnet(VpcId=vpc_id, CidrBlock='10.0.2.0/24', AvailabilityZone='us-east-1a')
    priv_subnet_id = priv_subnet['Subnet']['SubnetId']

    # Internet Gateway — allows public subnet traffic to/from internet
    igw = ec2.create_internet_gateway()
    igw_id = igw['InternetGateway']['InternetGatewayId']
    ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)

    # Route table for public subnet
    pub_rt = ec2.create_route_table(VpcId=vpc_id)
    pub_rt_id = pub_rt['RouteTable']['RouteTableId']
    ec2.create_route(RouteTableId=pub_rt_id, DestinationCidrBlock='0.0.0.0/0', GatewayId=igw_id)
    ec2.associate_route_table(RouteTableId=pub_rt_id, SubnetId=pub_subnet_id)

    # Elastic IP + NAT Gateway in public subnet (private subnet internet egress)
    eip = ec2.allocate_address(Domain='vpc')
    nat_gw = ec2.create_nat_gateway(SubnetId=pub_subnet_id, AllocationId=eip['AllocationId'])
    nat_gw_id = nat_gw['NatGateway']['NatGatewayId']

    # Route table for private subnet — routes internet traffic through NAT
    priv_rt = ec2.create_route_table(VpcId=vpc_id)
    priv_rt_id = priv_rt['RouteTable']['RouteTableId']
    # NOTE: NAT GW needs to be in AVAILABLE state before adding route; in practice use waiter
    ec2.create_route(RouteTableId=priv_rt_id, DestinationCidrBlock='0.0.0.0/0', NatGatewayId=nat_gw_id)
    ec2.associate_route_table(RouteTableId=priv_rt_id, SubnetId=priv_subnet_id)

    return vpc_id, pub_subnet_id, priv_subnet_id

# --- 1.2 Security Group vs NACL ---
# Security Group: stateful, instance-level, ALLOW rules only
# NACL: stateless (inbound + outbound rules), subnet-level, ALLOW + DENY rules
def create_security_groups(vpc_id):
    ec2 = boto3.client('ec2')

    # ALB Security Group — allows HTTP/HTTPS from internet
    alb_sg = ec2.create_security_group(
        GroupName='alb-sg', Description='ALB inbound', VpcId=vpc_id)
    alb_sg_id = alb_sg['GroupId']
    ec2.authorize_security_group_ingress(GroupId=alb_sg_id, IpPermissions=[
        {'IpProtocol': 'tcp', 'FromPort': 443, 'ToPort': 443, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
        {'IpProtocol': 'tcp', 'FromPort': 80, 'ToPort': 80, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
    ])

    # ECS Task Security Group — allows traffic ONLY from ALB sg
    ecs_sg = ec2.create_security_group(
        GroupName='ecs-task-sg', Description='ECS tasks', VpcId=vpc_id)
    ecs_sg_id = ecs_sg['GroupId']
    ec2.authorize_security_group_ingress(GroupId=ecs_sg_id, IpPermissions=[
        {'IpProtocol': 'tcp', 'FromPort': 8080, 'ToPort': 8080,
         'UserIdGroupPairs': [{'GroupId': alb_sg_id}]},  # only from ALB
    ])
    return alb_sg_id, ecs_sg_id

# --- 1.3 VPC Endpoint — S3 Gateway endpoint (private traffic, no NAT costs) ---
def create_s3_vpc_endpoint(vpc_id, route_table_id):
    ec2 = boto3.client('ec2', region_name='us-east-1')
    ec2.create_vpc_endpoint(
        VpcId=vpc_id,
        ServiceName='com.amazonaws.us-east-1.s3',
        RouteTableIds=[route_table_id],
        VpcEndpointType='Gateway'  # S3 and DynamoDB support Gateway; others use Interface
    )

# --- 1.4 VPC Flow Logs — capture network traffic metadata ---
def enable_vpc_flow_logs(vpc_id, log_group_name, iam_role_arn):
    ec2 = boto3.client('ec2')
    ec2.create_flow_logs(
        ResourceIds=[vpc_id],
        ResourceType='VPC',
        TrafficType='ALL',         # ALL | ACCEPT | REJECT
        LogDestinationType='cloud-watch-logs',
        LogGroupName=log_group_name,
        DeliverLogsPermissionArn=iam_role_arn
    )

# INTERVIEW Q: VPC Peering vs Transit Gateway?
# VPC Peering: 1-to-1, non-transitive (A-B + B-C doesn't mean A-C)
# Transit Gateway: hub-and-spoke, supports transitive routing, cross-account, VPN/Direct Connect


# ============================================================================
# SECTION 2: EBS & EFS — Storage for EC2 and ECS
# ============================================================================
# KEY CONCEPTS:
#   - EBS: block storage, single AZ, attached to one EC2 at a time (except Multi-Attach io2)
#   - EFS: NFS-based, multi-AZ, shared across many EC2/ECS tasks simultaneously
#   - EBS Types: gp3 (general), io2 (high IOPS), st1 (throughput), sc1 (cold)
#   - EFS Performance modes: General Purpose vs Max I/O
#   - EFS Throughput modes: Bursting vs Provisioned vs Elastic
#   - EBS snapshots → S3 (incremental), can copy cross-region
#   - EFS used for shared config, ML models, CMS content in microservices
# ============================================================================

# --- 2.1 Create and attach EBS volume ---
def create_ebs_volume():
    ec2 = boto3.client('ec2', region_name='us-east-1')

    volume = ec2.create_volume(
        AvailabilityZone='us-east-1a',
        Size=100,          # GB
        VolumeType='gp3',  # gp3: cheaper than gp2, configurable IOPS/throughput
        Iops=3000,         # gp3: 3000 baseline, up to 16000
        Throughput=125,    # MB/s, gp3 specific
        Encrypted=True,
        TagSpecifications=[{'ResourceType': 'volume', 'Tags': [{'Key': 'Name', 'Value': 'app-data'}]}]
    )
    volume_id = volume['VolumeId']

    # Attach to EC2 instance (same AZ required)
    ec2.attach_volume(VolumeId=volume_id, InstanceId='i-1234567890abcdef0', Device='/dev/xvdf')

    # Snapshot for backup
    ec2.create_snapshot(VolumeId=volume_id, Description='daily-backup')
    return volume_id

# --- 2.2 Create EFS for shared microservice storage ---
def create_efs_filesystem():
    efs = boto3.client('efs', region_name='us-east-1')

    fs = efs.create_file_system(
        PerformanceMode='generalPurpose',  # or 'maxIO' for massive parallelism
        ThroughputMode='elastic',           # scales automatically with workload
        Encrypted=True,
        Tags=[{'Key': 'Name', 'Value': 'shared-microservice-storage'}]
    )
    fs_id = fs['FileSystemId']

    # Create mount targets in each AZ subnet (needed for ECS Fargate access)
    efs.create_mount_target(
        FileSystemId=fs_id,
        SubnetId='subnet-xxxxxxxx',
        SecurityGroups=['sg-xxxxxxxx']  # SG must allow NFS port 2049
    )

    # Lifecycle policy — move files to IA storage class after 30 days
    efs.put_lifecycle_configuration(
        FileSystemId=fs_id,
        LifecyclePolicies=[
            {'TransitionToIA': 'AFTER_30_DAYS'},
            {'TransitionToPrimaryStorageClass': 'AFTER_1_ACCESS'}  # move back on access
        ]
    )
    return fs_id

# INTERVIEW Q: When EBS vs EFS?
# EBS: single-instance, low latency databases (MySQL on EC2), boot volumes
# EFS: shared file system, multiple ECS tasks reading same ML model or config


# ============================================================================
# SECTION 3: ELB — Elastic Load Balancing (ALB focus for microservices)
# ============================================================================
# KEY CONCEPTS:
#   - ALB (Application LB): L7, path/host/header routing, best for microservices
#   - NLB (Network LB): L4, ultra-low latency, static IP, TCP/UDP
#   - GLB (Gateway LB): for inline security appliances (firewalls)
#   - Target Groups: EC2, ECS tasks, Lambda, IP addresses
#   - ALB Listener Rules: priority-based, path-based routing
#   - Sticky Sessions (session affinity) via cookies
#   - Connection draining (deregistration delay)
#   - ALB access logs → S3
#   - Health checks — critical for zero-downtime ECS deployments
# ============================================================================

# --- 3.1 ALB with path-based routing for microservices ---
def create_alb_with_routing():
    elbv2 = boto3.client('elbv2', region_name='us-east-1')

    # Create ALB
    alb = elbv2.create_load_balancer(
        Name='microservice-alb',
        Subnets=['subnet-pub-1a', 'subnet-pub-1b'],  # Must be public subnets
        SecurityGroups=['sg-alb-id'],
        Scheme='internet-facing',
        Type='application',
        IpAddressType='ipv4'
    )
    alb_arn = alb['LoadBalancers'][0]['LoadBalancerArn']

    # Target group for Orders service
    orders_tg = elbv2.create_target_group(
        Name='orders-service-tg',
        Protocol='HTTP',
        Port=8080,
        VpcId='vpc-xxxxxxxx',
        TargetType='ip',                     # 'ip' for ECS Fargate tasks
        HealthCheckPath='/health',
        HealthCheckIntervalSeconds=30,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=3
    )
    orders_tg_arn = orders_tg['TargetGroups'][0]['TargetGroupArn']

    # Target group for Products service
    products_tg = elbv2.create_target_group(
        Name='products-service-tg', Protocol='HTTP', Port=8080,
        VpcId='vpc-xxxxxxxx', TargetType='ip', HealthCheckPath='/health'
    )
    products_tg_arn = products_tg['TargetGroups'][0]['TargetGroupArn']

    # HTTPS Listener
    listener = elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol='HTTPS',
        Port=443,
        Certificates=[{'CertificateArn': 'arn:aws:acm:us-east-1:123:certificate/xxx'}],
        DefaultActions=[{'Type': 'forward', 'TargetGroupArn': orders_tg_arn}]
    )
    listener_arn = listener['Listeners'][0]['ListenerArn']

    # Path-based routing rules — the core of microservice routing via single ALB
    elbv2.create_rule(
        ListenerArn=listener_arn,
        Priority=10,
        Conditions=[{'Field': 'path-pattern', 'Values': ['/api/orders/*']}],
        Actions=[{'Type': 'forward', 'TargetGroupArn': orders_tg_arn}]
    )
    elbv2.create_rule(
        ListenerArn=listener_arn,
        Priority=20,
        Conditions=[{'Field': 'path-pattern', 'Values': ['/api/products/*']}],
        Actions=[{'Type': 'forward', 'TargetGroupArn': products_tg_arn}]
    )

    # Header-based routing (e.g., canary deployments)
    # elbv2.create_rule(Conditions=[{'Field': 'http-header', 'HttpHeaderConfig': {'HttpHeaderName': 'X-Canary', 'Values': ['true']}}], ...)

    return alb_arn


# ============================================================================
# SECTION 4: Route 53 — DNS and Service Discovery
# ============================================================================
# KEY CONCEPTS:
#   - Routing policies: Simple, Weighted, Latency-based, Failover, Geolocation,
#     Geoproximity, Multivalue Answer
#   - Health checks integration with failover routing
#   - Private hosted zones for internal microservice DNS
#   - Alias records (free, for AWS resources) vs CNAME
#   - Route 53 Resolver for hybrid DNS (on-prem ↔ AWS)
#   - Service discovery via Cloud Map + Route 53 auto naming
# ============================================================================

# --- 4.1 Weighted routing for blue/green or canary deployments ---
def create_weighted_routing():
    r53 = boto3.client('route53')
    hosted_zone_id = 'ZXXXXXXXXXXXXX'

    r53.change_resource_record_sets(
        HostedZoneId=hosted_zone_id,
        ChangeBatch={
            'Changes': [
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': 'api.myapp.com',
                        'Type': 'A',
                        'SetIdentifier': 'blue-env',  # required for weighted
                        'Weight': 90,
                        'AliasTarget': {
                            'HostedZoneId': 'Z35SXDOTRQ7X7K',  # ALB hosted zone
                            'DNSName': 'blue-alb.us-east-1.elb.amazonaws.com',
                            'EvaluateTargetHealth': True
                        }
                    }
                },
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': 'api.myapp.com',
                        'Type': 'A',
                        'SetIdentifier': 'green-env',
                        'Weight': 10,  # 10% traffic to new version
                        'AliasTarget': {
                            'HostedZoneId': 'Z35SXDOTRQ7X7K',
                            'DNSName': 'green-alb.us-east-1.elb.amazonaws.com',
                            'EvaluateTargetHealth': True
                        }
                    }
                }
            ]
        }
    )

# --- 4.2 Failover routing with health checks ---
def create_failover_routing():
    r53 = boto3.client('route53')

    # Create health check for primary region
    hc = r53.create_health_check(
        CallerReference='primary-hc-001',
        HealthCheckConfig={
            'IPAddress': '1.2.3.4',
            'Port': 443,
            'Type': 'HTTPS',
            'ResourcePath': '/health',
            'RequestInterval': 30,
            'FailureThreshold': 3
        }
    )
    hc_id = hc['HealthCheck']['Id']

    r53.change_resource_record_sets(
        HostedZoneId='ZXXXXXXXXXXXXX',
        ChangeBatch={'Changes': [
            {'Action': 'UPSERT', 'ResourceRecordSet': {
                'Name': 'api.myapp.com', 'Type': 'A',
                'SetIdentifier': 'primary', 'Failover': 'PRIMARY',
                'HealthCheckId': hc_id,
                'AliasTarget': {'HostedZoneId': 'Z35SXDOTRQ7X7K',
                                'DNSName': 'primary-alb.elb.amazonaws.com',
                                'EvaluateTargetHealth': True}
            }},
            {'Action': 'UPSERT', 'ResourceRecordSet': {
                'Name': 'api.myapp.com', 'Type': 'A',
                'SetIdentifier': 'secondary', 'Failover': 'SECONDARY',
                'AliasTarget': {'HostedZoneId': 'Z35SXDOTRQ7X7K',
                                'DNSName': 'secondary-alb.elb.amazonaws.com',
                                'EvaluateTargetHealth': True}
            }}
        ]}
    )

# --- 4.3 Private Hosted Zone for internal service-to-service DNS ---
def create_private_hosted_zone(vpc_id):
    r53 = boto3.client('route53')
    zone = r53.create_hosted_zone(
        Name='internal.myapp.local',
        CallerReference='private-zone-001',
        HostedZoneConfig={'PrivateZone': True},
        VPC={'VPCRegion': 'us-east-1', 'VPCId': vpc_id}
    )
    # Now ECS services can reach orders.internal.myapp.local internally


# ============================================================================
# SECTION 5: ECS — Elastic Container Service (Fargate focus)
# ============================================================================
# KEY CONCEPTS:
#   - Launch types: EC2 (you manage instances) vs Fargate (serverless containers)
#   - Task Definition: blueprint (CPU, memory, image, env vars, volumes, logging)
#   - Service: runs N tasks, integrates with ALB, handles rolling deployments
#   - Cluster: logical grouping of tasks/services
#   - Task IAM Role vs Execution Role (pulls ECR image, writes logs)
#   - Service Auto Scaling: Target Tracking, Step Scaling
#   - Blue/Green deployment via CodeDeploy + ECS
#   - Service Connect / Cloud Map for service discovery
#   - ECS Exec for container debugging (like SSH)
#   - Capacity Providers for cost optimization (Fargate Spot)
# ============================================================================

# --- 5.1 Register Task Definition ---
def register_task_definition():
    ecs = boto3.client('ecs', region_name='us-east-1')

    response = ecs.register_task_definition(
        family='orders-service',
        networkMode='awsvpc',          # Required for Fargate; each task gets its own ENI
        requiresCompatibilities=['FARGATE'],
        cpu='512',                     # 0.5 vCPU
        memory='1024',                 # 1 GB
        executionRoleArn='arn:aws:iam::123:role/ecsTaskExecutionRole',  # pull image, push logs
        taskRoleArn='arn:aws:iam::123:role/ordersServiceTaskRole',       # app-level AWS access

        containerDefinitions=[{
            'name': 'orders-service',
            'image': '123456789.dkr.ecr.us-east-1.amazonaws.com/orders-service:latest',
            'portMappings': [{'containerPort': 8080, 'protocol': 'tcp'}],
            'environment': [
                {'name': 'ENV', 'value': 'production'},
                {'name': 'DB_HOST', 'value': 'orders-db.internal.myapp.local'}
            ],
            'secrets': [  # Fetched from SSM/Secrets Manager at task start
                {'name': 'DB_PASSWORD', 'valueFrom': 'arn:aws:secretsmanager:us-east-1:123:secret:orders-db-pass'}
            ],
            'logConfiguration': {
                'logDriver': 'awslogs',
                'options': {
                    'awslogs-group': '/ecs/orders-service',
                    'awslogs-region': 'us-east-1',
                    'awslogs-stream-prefix': 'ecs'
                }
            },
            'healthCheck': {
                'command': ['CMD-SHELL', 'curl -f http://localhost:8080/health || exit 1'],
                'interval': 30,
                'timeout': 5,
                'retries': 3
            }
        }],

        volumes=[{  # Attach EFS volume for shared storage
            'name': 'shared-config',
            'efsVolumeConfiguration': {
                'fileSystemId': 'fs-xxxxxxxx',
                'rootDirectory': '/config',
                'transitEncryption': 'ENABLED'
            }
        }]
    )
    return response['taskDefinition']['taskDefinitionArn']

# --- 5.2 Create ECS Service with ALB and Auto Scaling ---
def create_ecs_service(cluster_name, task_def_arn, tg_arn, subnet_ids, sg_ids):
    ecs = boto3.client('ecs')
    appas = boto3.client('application-autoscaling')

    service = ecs.create_service(
        cluster=cluster_name,
        serviceName='orders-service',
        taskDefinition=task_def_arn,
        desiredCount=3,
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': subnet_ids,
                'securityGroups': sg_ids,
                'assignPublicIp': 'DISABLED'  # Private subnet; NAT handles egress
            }
        },
        loadBalancers=[{
            'targetGroupArn': tg_arn,
            'containerName': 'orders-service',
            'containerPort': 8080
        }],
        deploymentConfiguration={
            'maximumPercent': 200,       # Allow double tasks during rolling deploy
            'minimumHealthyPercent': 100 # Never go below desired count
        },
        deploymentController={'type': 'ECS'},  # or 'CODE_DEPLOY' for blue/green
        enableExecuteCommand=True  # Enables ECS Exec (container shell access)
    )

    # Register scalable target
    appas.register_scalable_target(
        ServiceNamespace='ecs',
        ResourceId=f'service/{cluster_name}/orders-service',
        ScalableDimension='ecs:service:DesiredCount',
        MinCapacity=2,
        MaxCapacity=20
    )

    # Scale on CPU — Target Tracking
    appas.put_scaling_policy(
        PolicyName='orders-cpu-scaling',
        ServiceNamespace='ecs',
        ResourceId=f'service/{cluster_name}/orders-service',
        ScalableDimension='ecs:service:DesiredCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'PredefinedMetricSpecification': {'PredefinedMetricType': 'ECSServiceAverageCPUUtilization'},
            'TargetValue': 60.0,   # Scale to keep CPU at 60%
            'ScaleInCooldown': 300,
            'ScaleOutCooldown': 60
        }
    )

# --- 5.3 ECS Exec — debug a running container ---
import subprocess
def ecs_exec_into_container(cluster, task_id, container_name):
    # Requires: enableExecuteCommand=True on service, SSM Agent in container
    subprocess.run([
        'aws', 'ecs', 'execute-command',
        '--cluster', cluster,
        '--task', task_id,
        '--container', container_name,
        '--interactive',
        '--command', '/bin/sh'
    ])

# INTERVIEW Q: ECS rolling deploy vs blue/green?
# Rolling: replace tasks gradually; risk of mixed versions serving traffic
# Blue/Green (CodeDeploy): full new env, instant cutover or canary shift; easy rollback


# ============================================================================
# SECTION 6: ECR — Elastic Container Registry
# ============================================================================
# KEY CONCEPTS:
#   - Private and public registries
#   - Image scanning: Basic (on push) vs Enhanced (via Inspector, continuous)
#   - Image tag immutability — prevent overwriting tags (prod safety)
#   - Lifecycle policies — auto-delete old untagged images (cost)
#   - Cross-account access via resource-based policy
#   - ECR pull-through cache — cache Docker Hub images in ECR
# ============================================================================

# --- 6.1 Create repo, push image, set lifecycle policy ---
def setup_ecr_repository():
    ecr = boto3.client('ecr', region_name='us-east-1')

    repo = ecr.create_repository(
        repositoryName='orders-service',
        imageTagMutability='IMMUTABLE',  # Prevents accidental tag overwrite
        imageScanningConfiguration={'scanOnPush': True},
        encryptionConfiguration={'encryptionType': 'AES256'}
    )
    repo_uri = repo['repository']['repositoryUri']

    # Lifecycle policy: keep last 10 tagged images, delete untagged after 1 day
    ecr.put_lifecycle_policy(
        repositoryName='orders-service',
        lifecyclePolicyText=json.dumps({
            'rules': [
                {
                    'rulePriority': 1,
                    'description': 'Delete untagged images after 1 day',
                    'selection': {'tagStatus': 'untagged', 'countType': 'sinceImagePushed',
                                  'countUnit': 'days', 'countNumber': 1},
                    'action': {'type': 'expire'}
                },
                {
                    'rulePriority': 2,
                    'description': 'Keep last 10 tagged images',
                    'selection': {'tagStatus': 'tagged', 'tagPrefixList': ['v'],
                                  'countType': 'imageCountMoreThan', 'countNumber': 10},
                    'action': {'type': 'expire'}
                }
            ]
        })
    )
    return repo_uri

# --- 6.2 Cross-account ECR access policy ---
def set_ecr_cross_account_policy(repo_name, consumer_account_id):
    ecr = boto3.client('ecr')
    ecr.set_repository_policy(
        repositoryName=repo_name,
        policy=json.dumps({
            'Version': '2012-10-17',
            'Statement': [{
                'Effect': 'Allow',
                'Principal': {'AWS': f'arn:aws:iam::{consumer_account_id}:root'},
                'Action': ['ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage',
                           'ecr:BatchCheckLayerAvailability']
            }]
        })
    )


# ============================================================================
# SECTION 7: S3 — Simple Storage Service
# ============================================================================
# KEY CONCEPTS:
#   - Storage classes: S3 Standard, Intelligent-Tiering, Standard-IA, Glacier, Glacier Deep Archive
#   - Versioning, MFA Delete, Object Lock (WORM compliance)
#   - Lifecycle rules — transition/expire objects automatically
#   - Replication: CRR (cross-region), SRR (same-region); requires versioning
#   - Event notifications → Lambda / SQS / SNS
#   - Presigned URLs — temporary access without credentials
#   - S3 Transfer Acceleration (CloudFront edge for uploads)
#   - Multipart upload — for files > 100MB
#   - S3 Select — SQL queries on individual objects (CSV, JSON, Parquet)
#   - Block Public Access settings (bucket + account level)
#   - VPC Gateway Endpoint for private S3 access
# ============================================================================

# --- 7.1 Bucket with versioning, encryption, and lifecycle ---
def create_s3_bucket_production():
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'myapp-orders-artifacts-prod'

    s3.create_bucket(Bucket=bucket_name)

    # Block all public access
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True, 'IgnorePublicAcls': True,
            'BlockPublicPolicy': True, 'RestrictPublicBuckets': True
        }
    )

    # Enable versioning
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={'Status': 'Enabled'}
    )

    # Server-side encryption with KMS
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': 'arn:aws:kms:us-east-1:123:key/xxx'
            },
            'BucketKeyEnabled': True  # Reduces KMS call costs significantly
        }]}
    )

    # Lifecycle policy
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration={'Rules': [{
            'ID': 'archive-and-expire',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'logs/'},
            'Transitions': [
                {'Days': 30, 'StorageClass': 'STANDARD_IA'},
                {'Days': 90, 'StorageClass': 'GLACIER'}
            ],
            'Expiration': {'Days': 365},
            'NoncurrentVersionExpiration': {'NoncurrentDays': 30}
        }]}
    )

# --- 7.2 S3 Event Notification → SQS ---
def configure_s3_event_notification(bucket_name, sqs_arn):
    s3 = boto3.client('s3')
    s3.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration={
            'QueueConfigurations': [{
                'QueueArn': sqs_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {'Key': {'FilterRules': [
                    {'Name': 'prefix', 'Value': 'uploads/'},
                    {'Name': 'suffix', 'Value': '.json'}
                ]}}
            }]
        }
    )

# --- 7.3 Presigned URL for secure temporary upload ---
def generate_presigned_upload_url(bucket_name, object_key, expiry_seconds=3600):
    s3 = boto3.client('s3', region_name='us-east-1')
    url = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={'Bucket': bucket_name, 'Key': object_key,
                'ContentType': 'application/json'},
        ExpiresIn=expiry_seconds
    )
    return url  # Client uploads directly to S3; backend never handles the file stream

# --- 7.4 Multipart upload for large files ---
def multipart_upload(bucket, key, file_path):
    s3 = boto3.client('s3')
    mpu = s3.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id = mpu['UploadId']
    parts = []

    with open(file_path, 'rb') as f:
        part_number = 1
        while True:
            data = f.read(10 * 1024 * 1024)  # 10MB chunks
            if not data:
                break
            part = s3.upload_part(Bucket=bucket, Key=key, PartNumber=part_number,
                                   UploadId=upload_id, Body=data)
            parts.append({'PartNumber': part_number, 'ETag': part['ETag']})
            part_number += 1

    s3.complete_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id,
        MultipartUpload={'Parts': parts}
    )


# ============================================================================
# SECTION 8: DynamoDB — NoSQL Key-Value / Document Store
# ============================================================================
# KEY CONCEPTS:
#   - Partition key (+ optional sort key) design is critical for even distribution
#   - Hot partition problem — avoid high-cardinality writes to single key
#   - Read/Write Capacity: Provisioned vs On-Demand
#   - GSI (Global Secondary Index): different PK/SK, eventually consistent reads
#   - LSI (Local Secondary Index): same PK, different SK, strongly consistent reads
#   - DynamoDB Streams — change data capture (triggers Lambda)
#   - TTL — auto-expire items (sessions, cache, leases)
#   - Transactions — ACID for up to 100 items across tables
#   - DAX (DynamoDB Accelerator) — in-memory cache, microsecond reads
#   - Single Table Design — embed multiple entity types in one table
#   - Condition expressions — optimistic locking
#   - Pagination with LastEvaluatedKey
# ============================================================================

# --- 8.1 Create DynamoDB table with GSI ---
def create_dynamodb_table():
    dynamodb = boto3.client('dynamodb', region_name='us-east-1')

    dynamodb.create_table(
        TableName='Orders',
        BillingMode='PAY_PER_REQUEST',  # On-demand; no capacity planning
        TableClass='STANDARD',
        AttributeDefinitions=[
            {'AttributeName': 'PK', 'AttributeType': 'S'},       # e.g., ORDER#<order_id>
            {'AttributeName': 'SK', 'AttributeType': 'S'},       # e.g., ITEM#<item_id>
            {'AttributeName': 'customerId', 'AttributeType': 'S'},
            {'AttributeName': 'createdAt', 'AttributeType': 'S'}
        ],
        KeySchema=[
            {'AttributeName': 'PK', 'KeyType': 'HASH'},
            {'AttributeName': 'SK', 'KeyType': 'RANGE'}
        ],
        GlobalSecondaryIndexes=[{
            'IndexName': 'CustomerOrdersIndex',
            'KeySchema': [
                {'AttributeName': 'customerId', 'KeyType': 'HASH'},
                {'AttributeName': 'createdAt', 'KeyType': 'RANGE'}  # enables time-range queries
            ],
            'Projection': {'ProjectionType': 'ALL'}
        }],
        StreamSpecification={
            'StreamEnabled': True,
            'StreamViewType': 'NEW_AND_OLD_IMAGES'  # for Lambda triggers
        },
        SSESpecification={'Enabled': True, 'SSEType': 'KMS'}
    )

# --- 8.2 DynamoDB operations with resource interface ---
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

dynamodb_resource = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb_resource.Table('Orders')

def put_order(order_id, customer_id, items, total):
    table.put_item(
        Item={
            'PK': f'ORDER#{order_id}',
            'SK': 'METADATA',
            'customerId': customer_id,
            'total': Decimal(str(total)),
            'status': 'PENDING',
            'createdAt': '2024-01-15T10:00:00Z',
            'ttl': 1735689600  # epoch; DynamoDB auto-deletes after this time
        },
        ConditionExpression='attribute_not_exists(PK)'  # Idempotent — fail if exists
    )

def get_order(order_id):
    response = table.get_item(
        Key={'PK': f'ORDER#{order_id}', 'SK': 'METADATA'},
        ConsistentRead=True  # Strongly consistent; more expensive but always latest
    )
    return response.get('Item')

def query_customer_orders(customer_id, from_date):
    # Query GSI — get all orders for a customer after a date
    response = table.query(
        IndexName='CustomerOrdersIndex',
        KeyConditionExpression=Key('customerId').eq(customer_id) & Key('createdAt').gt(from_date),
        FilterExpression=Attr('status').ne('CANCELLED'),
        ScanIndexForward=False,  # Descending order (newest first)
        Limit=20
    )
    return response['Items']

def update_order_status_optimistic(order_id, new_status, expected_version):
    # Optimistic locking with condition expression
    try:
        table.update_item(
            Key={'PK': f'ORDER#{order_id}', 'SK': 'METADATA'},
            UpdateExpression='SET #s = :new_status, version = :new_version',
            ConditionExpression='version = :expected_version',
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={
                ':new_status': new_status,
                ':new_version': expected_version + 1,
                ':expected_version': expected_version
            }
        )
    except dynamodb_resource.meta.client.exceptions.ConditionalCheckFailedException:
        raise Exception('Concurrent modification detected — retry with latest version')

def transactional_place_order(order_id, customer_id, product_id, quantity):
    # Atomic: create order + decrement inventory
    dynamodb_resource.meta.client.transact_write_items(
        TransactItems=[
            {
                'Put': {
                    'TableName': 'Orders',
                    'Item': {
                        'PK': {'S': f'ORDER#{order_id}'},
                        'SK': {'S': 'METADATA'},
                        'customerId': {'S': customer_id},
                        'status': {'S': 'PENDING'}
                    },
                    'ConditionExpression': 'attribute_not_exists(PK)'
                }
            },
            {
                'Update': {
                    'TableName': 'Products',
                    'Key': {'PK': {'S': f'PRODUCT#{product_id}'}, 'SK': {'S': 'STOCK'}},
                    'UpdateExpression': 'SET quantity = quantity - :q',
                    'ConditionExpression': 'quantity >= :q',
                    'ExpressionAttributeValues': {':q': {'N': str(quantity)}}
                }
            }
        ]
    )

# INTERVIEW Q: How to avoid hot partitions?
# - Use high-cardinality partition keys (user_id, order_id, UUID)
# - Add random suffix/shard prefix for write-heavy keys
# - Use write sharding: PK = f"{base_key}#{random.randint(0, N)}" then scatter-gather read


# ============================================================================
# SECTION 9: Lambda — Serverless Functions
# ============================================================================
# KEY CONCEPTS:
#   - Event sources: S3, SQS, SNS, DynamoDB Streams, API Gateway, EventBridge, ALB
#   - Execution model: cold start vs warm start
#   - Concurrency: reserved (guarantees), provisioned (pre-warmed, eliminates cold start)
#   - Lambda Layers — shared code/dependencies
#   - Lambda@Edge / CloudFront Functions — edge compute
#   - Destinations: on success/failure → SQS, SNS, EventBridge, another Lambda
#   - Dead Letter Queue (DLQ) for failed async invocations
#   - Power Tuning — right-size memory (more memory = more CPU = faster = cheaper)
#   - VPC Lambda — can access private RDS/ElastiCache; adds cold start latency
#   - Lambda SnapStart (Java) — pre-initialized snapshots for faster cold starts
#   - Event filtering for SQS/DynamoDB Stream triggers
# ============================================================================

# --- 9.1 Lambda handler patterns ---
def s3_event_handler(event, context):
    """Triggered by S3 ObjectCreated event"""
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        size = record['s3']['object']['size']
        print(f'Processing: s3://{bucket}/{key} ({size} bytes)')
        # process_file(bucket, key)

def sqs_batch_handler(event, context):
    """
    SQS trigger with partial batch failure support.
    Configure: FunctionResponseTypes = ['ReportBatchItemFailures']
    Lambda will retry ONLY failed messages, not the whole batch.
    """
    failed_message_ids = []
    for record in event['Records']:
        try:
            message_body = json.loads(record['body'])
            process_message(message_body)
        except Exception as e:
            print(f'Failed to process {record["messageId"]}: {e}')
            failed_message_ids.append({'itemIdentifier': record['messageId']})

    return {'batchItemFailures': failed_message_ids}  # retry only these

def process_message(msg): pass  # placeholder

def dynamodb_stream_handler(event, context):
    """Triggered by DynamoDB Streams — react to data changes"""
    for record in event['Records']:
        event_name = record['eventName']  # INSERT | MODIFY | REMOVE
        if event_name == 'INSERT':
            new_item = record['dynamodb'].get('NewImage', {})
            print(f'New order: {new_item}')
        elif event_name == 'MODIFY':
            old = record['dynamodb'].get('OldImage', {})
            new = record['dynamodb'].get('NewImage', {})
            # Detect status changes
            if old.get('status') != new.get('status'):
                print(f'Status changed: {old["status"]} → {new["status"]}')
        elif event_name == 'REMOVE':
            old_item = record['dynamodb'].get('OldImage', {})
            print(f'Deleted: {old_item}')

# --- 9.2 Lambda with SQS — configure event source mapping ---
def configure_lambda_sqs_trigger():
    lambda_client = boto3.client('lambda')
    lambda_client.create_event_source_mapping(
        EventSourceArn='arn:aws:sqs:us-east-1:123:orders-queue',
        FunctionName='process-order',
        BatchSize=10,              # Process up to 10 messages per invocation
        MaximumBatchingWindowInSeconds=5,  # Wait up to 5s to fill the batch
        FunctionResponseTypes=['ReportBatchItemFailures'],  # Partial batch failure
        FilterCriteria={           # Only trigger for specific message patterns
            'Filters': [{'Pattern': json.dumps({'body': {'eventType': ['ORDER_CREATED']}})}]
        }
    )

# --- 9.3 Provisioned concurrency to eliminate cold starts ---
def configure_provisioned_concurrency():
    lambda_client = boto3.client('lambda')
    # Publish version first
    version = lambda_client.publish_version(FunctionName='orders-api')
    version_number = version['Version']

    # Set provisioned concurrency on alias
    lambda_client.put_provisioned_concurrency_config(
        FunctionName='orders-api',
        Qualifier='prod',           # alias pointing to version
        ProvisionedConcurrentExecutions=10  # always-warm instances
    )

# INTERVIEW Q: How to handle Lambda cold starts?
# 1. Provisioned concurrency (guaranteed warm) 2. Keep package small (minimize zip)
# 3. Move SDK clients outside handler (reused across warm invocations)
# 4. Use Lambda SnapStart for Java 11+ 5. Avoid VPC if not needed (adds ~100ms cold start)

# Good practice: initialize clients outside handler
import boto3 as _boto3
_s3_client = _boto3.client('s3')  # reused in warm invocations
def optimized_handler(event, context):
    return _s3_client.list_buckets()  # no re-init cost


# ============================================================================
# SECTION 10: SQS — Simple Queue Service
# ============================================================================
# KEY CONCEPTS:
#   - Standard Queue: at-least-once delivery, best-effort ordering, unlimited throughput
#   - FIFO Queue: exactly-once, strict ordering, 300 TPS (3000 with batching)
#   - Visibility Timeout: message hidden while processing; extend if processing takes long
#   - Dead Letter Queue (DLQ): after maxReceiveCount failures, message → DLQ
#   - Long Polling: up to 20s wait; reduces empty receives and cost
#   - Message attributes for filtering by SNS subscriptions
#   - Delay queues: delay delivery of new messages (0–900 seconds)
#   - Large message payloads via S3 + SQS Extended Client Library
# ============================================================================

# --- 10.1 Create FIFO queue with DLQ ---
def create_sqs_queues():
    sqs = boto3.client('sqs', region_name='us-east-1')

    # DLQ first
    dlq = sqs.create_queue(
        QueueName='orders-dlq.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true',
            'MessageRetentionPeriod': '1209600'  # 14 days (max)
        }
    )
    dlq_url = dlq['QueueUrl']
    dlq_attrs = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=['QueueArn'])
    dlq_arn = dlq_attrs['Attributes']['QueueArn']

    # Main queue
    queue = sqs.create_queue(
        QueueName='orders-queue.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'false',   # Provide explicit dedup ID
            'VisibilityTimeout': '300',              # 5 min; match Lambda timeout
            'ReceiveMessageWaitTimeSeconds': '20',   # Long polling
            'RedrivePolicy': json.dumps({
                'deadLetterTargetArn': dlq_arn,
                'maxReceiveCount': '3'               # 3 failures → DLQ
            })
        }
    )
    return queue['QueueUrl']

# --- 10.2 Send and receive messages ---
def send_order_message(queue_url, order_id, order_data):
    sqs = boto3.client('sqs')

    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(order_data),
        MessageGroupId='orders',                      # FIFO: group for ordering
        MessageDeduplicationId=f'order-{order_id}',   # FIFO: prevent duplicates
        MessageAttributes={
            'eventType': {'StringValue': 'ORDER_CREATED', 'DataType': 'String'},
            'priority': {'StringValue': 'HIGH', 'DataType': 'String'}
        }
    )

def process_sqs_messages(queue_url):
    sqs = boto3.client('sqs')

    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,        # Long polling
            MessageAttributeNames=['All']
        )
        for msg in response.get('Messages', []):
            try:
                body = json.loads(msg['Body'])
                # ... process ...
                # Extend visibility timeout if processing is slow
                sqs.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg['ReceiptHandle'],
                    VisibilityTimeout=60  # extend by 60s
                )
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
            except Exception as e:
                print(f'Processing failed: {e}')
                # Don't delete — visibility timeout expires, message reappears

# INTERVIEW Q: SQS Standard vs FIFO?
# Standard: massive scale, no ordering guarantee, at-least-once (idempotency needed)
# FIFO: ordered per group, exactly-once, 3000 msg/s — use for payment workflows


# ============================================================================
# SECTION 11: SNS — Simple Notification Service
# ============================================================================
# KEY CONCEPTS:
#   - Pub/Sub: one topic → fan-out to multiple subscribers
#   - Subscribers: SQS, Lambda, HTTP/S endpoint, Email, SMS, mobile push
#   - Message filtering — subscribers only receive matching messages (saves cost)
#   - FIFO topics for ordered, deduplicated fan-out
#   - SNS + SQS fan-out pattern: reliable async event distribution
#   - Message attributes for filtering
#   - Encryption at rest with KMS; in transit over HTTPS
# ============================================================================

# --- 11.1 SNS Fan-out: one event → multiple SQS queues ---
def setup_sns_fanout():
    sns = boto3.client('sns')
    sqs = boto3.client('sqs')

    # Create topic
    topic = sns.create_topic(
        Name='order-events',
        Attributes={'KmsMasterKeyId': 'alias/aws/sns'}
    )
    topic_arn = topic['TopicArn']

    # Subscribe SQS queues with message filtering
    inventory_queue_arn = 'arn:aws:sqs:us-east-1:123:inventory-queue'
    notification_queue_arn = 'arn:aws:sqs:us-east-1:123:notification-queue'

    # Inventory service — only ORDER_CREATED events
    sns.subscribe(
        TopicArn=topic_arn,
        Protocol='sqs',
        Endpoint=inventory_queue_arn,
        Attributes={
            'FilterPolicy': json.dumps({'eventType': ['ORDER_CREATED', 'ORDER_CANCELLED']}),
            'FilterPolicyScope': 'MessageAttributes'
        }
    )

    # Notification service — ORDER_CREATED and ORDER_SHIPPED only
    sns.subscribe(
        TopicArn=topic_arn,
        Protocol='sqs',
        Endpoint=notification_queue_arn,
        Attributes={
            'FilterPolicy': json.dumps({'eventType': ['ORDER_CREATED', 'ORDER_SHIPPED']})
        }
    )

    return topic_arn

def publish_order_event(topic_arn, event_type, order_data):
    sns = boto3.client('sns')
    sns.publish(
        TopicArn=topic_arn,
        Message=json.dumps(order_data),
        MessageAttributes={
            'eventType': {'DataType': 'String', 'StringValue': event_type}
        },
        Subject=f'Order Event: {event_type}'
    )

# INTERVIEW Q: SNS vs SQS vs EventBridge?
# SNS: push-based pub/sub, fan-out, no persistence
# SQS: pull-based queue, persistence, decoupling, backpressure handling
# EventBridge: event bus, complex routing rules, 90+ AWS service integrations, schema registry


# ============================================================================
# SECTION 12: Step Functions — Orchestrate Microservice Workflows
# ============================================================================
# KEY CONCEPTS:
#   - State machine types: Standard (long-running, audit trail) vs Express (high-volume, short-lived)
#   - States: Task, Choice, Wait, Parallel, Map, Pass, Succeed, Fail
#   - Retry and Catch blocks — built-in error handling
#   - .waitForTaskToken — pause and wait for external callback (human approval, etc.)
#   - Activity workers vs Lambda tasks vs SDK integrations
#   - Step Functions SDK integrations: 200+ AWS services directly (no Lambda needed)
#   - Execution history and visual debugging in console
#   - Choreography (events) vs Orchestration (Step Functions)
# ============================================================================

# --- 12.1 Order processing workflow state machine ---
ORDER_PROCESSING_ASL = {
    "Comment": "Order processing workflow",
    "StartAt": "ValidateOrder",
    "States": {
        "ValidateOrder": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123:function:validate-order",
            "Retry": [{
                "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException"],
                "IntervalSeconds": 2,
                "MaxAttempts": 3,
                "BackoffRate": 2.0  # Exponential backoff
            }],
            "Catch": [{"ErrorEquals": ["ValidationError"], "Next": "OrderFailed"}],
            "Next": "CheckInventory"
        },
        "CheckInventory": {
            "Type": "Task",
            "Resource": "arn:aws:states:::dynamodb:getItem",  # SDK integration, no Lambda
            "Parameters": {
                "TableName": "Products",
                "Key": {"PK": {"S.$": "$.productId"}}
            },
            "ResultPath": "$.inventoryResult",
            "Next": "IsInStock"
        },
        "IsInStock": {
            "Type": "Choice",
            "Choices": [{
                "Variable": "$.inventoryResult.Item.quantity.N",
                "NumericGreaterThan": 0,
                "Next": "ProcessPayment"
            }],
            "Default": "BackorderFlow"
        },
        "ProcessPayment": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke.waitForTaskToken",  # Async with callback
            "Parameters": {
                "FunctionName": "process-payment",
                "Payload": {
                    "orderId.$": "$.orderId",
                    "taskToken.$": "$$.Task.Token"  # Pass token to Lambda; it must call back
                }
            },
            "HeartbeatSeconds": 3600,
            "Next": "ParallelFulfillment"
        },
        "ParallelFulfillment": {
            "Type": "Parallel",
            "Branches": [
                {"StartAt": "UpdateInventory", "States": {"UpdateInventory": {
                    "Type": "Task",
                    "Resource": "arn:aws:lambda:us-east-1:123:function:update-inventory",
                    "End": True
                }}},
                {"StartAt": "SendConfirmationEmail", "States": {"SendConfirmationEmail": {
                    "Type": "Task",
                    "Resource": "arn:aws:lambda:us-east-1:123:function:send-email",
                    "End": True
                }}}
            ],
            "Next": "OrderComplete"
        },
        "BackorderFlow": {
            "Type": "Wait",
            "Seconds": 86400,  # Wait 24h
            "Next": "CheckInventory"
        },
        "OrderComplete": {"Type": "Succeed"},
        "OrderFailed": {"Type": "Fail", "Error": "OrderFailed", "Cause": "Validation or payment error"}
    }
}

# --- 12.2 Create and start state machine ---
def create_and_run_state_machine():
    sf = boto3.client('stepfunctions')

    sm = sf.create_state_machine(
        name='OrderProcessingWorkflow',
        definition=json.dumps(ORDER_PROCESSING_ASL),
        roleArn='arn:aws:iam::123:role/StepFunctionsExecutionRole',
        type='STANDARD'  # STANDARD for long-running; EXPRESS for high-TPS short workflows
    )

    execution = sf.start_execution(
        stateMachineArn=sm['stateMachineArn'],
        name=f'order-{uuid4()}',
        input=json.dumps({'orderId': 'ORD-123', 'productId': 'PROD#abc', 'quantity': 2})
    )
    return execution['executionArn']

def send_task_success_callback(task_token, payment_result):
    """Called by payment service Lambda to resume the workflow"""
    sf = boto3.client('stepfunctions')
    sf.send_task_success(taskToken=task_token, output=json.dumps(payment_result))

from uuid import uuid4


# ============================================================================
# SECTION 13: CloudWatch — Monitoring, Logging, Alarms
# ============================================================================
# KEY CONCEPTS:
#   - Metrics: default (5min) vs detailed monitoring (1min), custom metrics
#   - Alarms: threshold-based actions (SNS, Auto Scaling, EC2 actions)
#   - Composite alarms — combine multiple alarms (AND/OR logic)
#   - Log Groups, Log Streams, Log Insights (SQL-like queries)
#   - CloudWatch Agent — system-level metrics (memory, disk) from EC2
#   - Metric Filters — extract metrics from logs
#   - Container Insights — ECS/EKS cluster metrics
#   - Application Signals — RED metrics (Rate, Errors, Duration)
#   - Dashboards — unified operational view
#   - CloudWatch Synthetics — canary scripts for endpoint monitoring
# ============================================================================

# --- 13.1 Custom metrics and alarms ---
def put_custom_metrics():
    cw = boto3.client('cloudwatch')

    # Custom metric — business KPI (orders per minute)
    cw.put_metric_data(
        Namespace='MyApp/Orders',
        MetricData=[{
            'MetricName': 'OrdersPlaced',
            'Value': 42,
            'Unit': 'Count',
            'Dimensions': [
                {'Name': 'Service', 'Value': 'OrdersService'},
                {'Name': 'Environment', 'Value': 'production'}
            ]
        }]
    )

def create_alarm_with_actions(sns_topic_arn):
    cw = boto3.client('cloudwatch')

    # Alarm: Lambda error rate > 5% over 5 minutes
    cw.put_metric_alarm(
        AlarmName='OrdersLambda-HighErrorRate',
        AlarmDescription='Lambda error rate exceeds 5%',
        Namespace='AWS/Lambda',
        MetricName='Errors',
        Dimensions=[{'Name': 'FunctionName', 'Value': 'process-order'}],
        Statistic='Sum',
        Period=300,               # 5 minutes
        EvaluationPeriods=2,      # 2 consecutive periods
        Threshold=10,
        ComparisonOperator='GreaterThanThreshold',
        TreatMissingData='notBreaching',
        AlarmActions=[sns_topic_arn],
        OKActions=[sns_topic_arn]
    )

    # Alarm: DynamoDB throttled requests
    cw.put_metric_alarm(
        AlarmName='DynamoDB-Throttles',
        Namespace='AWS/DynamoDB',
        MetricName='SystemErrors',
        Dimensions=[{'Name': 'TableName', 'Value': 'Orders'}],
        Statistic='Sum',
        Period=60, EvaluationPeriods=1,
        Threshold=1, ComparisonOperator='GreaterThanOrEqualToThreshold',
        AlarmActions=[sns_topic_arn]
    )

# --- 13.2 CloudWatch Logs Insights query ---
def run_logs_insights_query():
    logs = boto3.client('logs')
    import time

    # Find slow Lambda invocations
    query = """
    fields @timestamp, @duration, @requestId
    | filter @type = "REPORT"
    | stats avg(@duration), max(@duration), percentile(@duration, 99) by bin(5m)
    | sort @timestamp desc
    | limit 20
    """

    response = logs.start_query(
        logGroupName='/aws/lambda/process-order',
        startTime=int(time.time()) - 3600,  # last 1 hour
        endTime=int(time.time()),
        queryString=query
    )
    query_id = response['queryId']

    # Poll for results
    while True:
        results = logs.get_query_results(queryId=query_id)
        if results['status'] in ('Complete', 'Failed', 'Cancelled'):
            return results['results']
        time.sleep(1)

# --- 13.3 Metric filter: extract error count from logs ---
def create_metric_filter():
    logs = boto3.client('logs')
    logs.put_metric_filter(
        logGroupName='/ecs/orders-service',
        filterName='OrderErrors',
        filterPattern='[timestamp, requestId, level=ERROR, ...]',
        metricTransformations=[{
            'metricName': 'OrderServiceErrors',
            'metricNamespace': 'MyApp/Orders',
            'metricValue': '1',
            'defaultValue': 0,
            'unit': 'Count'
        }]
    )


# ============================================================================
# SECTION 14: X-Ray — Distributed Tracing
# ============================================================================
# KEY CONCEPTS:
#   - Traces, Segments, Subsegments — distributed request journey
#   - Service Map — visual dependency graph of microservices
#   - Sampling rules — trace N% of requests (cost control)
#   - Annotations (indexed, filterable) vs Metadata (not indexed)
#   - X-Ray SDK for Lambda, ECS, API Gateway (auto-instrumentation)
#   - AWS Distro for OpenTelemetry (ADOT) — OTel → X-Ray
#   - CloudWatch ServiceLens — combines X-Ray traces with CW metrics/logs
#   - Groups — filter traces by expression for separate dashboards
# ============================================================================

# --- 14.1 X-Ray instrumentation in Lambda ---
# Install: pip install aws-xray-sdk
from aws_xray_sdk.core import xray_recorder, patch_all
patch_all()  # Auto-patches boto3, requests, httplib calls

@xray_recorder.capture('process_order_logic')
def process_order_with_tracing(order_id):
    # Add annotations (indexed — can filter in X-Ray console)
    xray_recorder.current_segment().put_annotation('orderId', order_id)
    xray_recorder.current_segment().put_annotation('service', 'OrdersService')

    # Add metadata (not indexed — debug details)
    xray_recorder.current_segment().put_metadata('orderDetails', {'id': order_id})

    with xray_recorder.in_subsegment('validate-order') as subsegment:
        subsegment.put_annotation('step', 'validation')
        # ... validation logic ...

    with xray_recorder.in_subsegment('dynamo-write') as subsegment:
        # boto3 calls here are auto-traced by patch_all()
        table.put_item(Item={'PK': f'ORDER#{order_id}', 'SK': 'METADATA'})

# --- 14.2 Configure sampling rules ---
def configure_xray_sampling():
    xray = boto3.client('xray')
    xray.create_sampling_rule(
        SamplingRule={
            'RuleName': 'HighVolumeEndpoint',
            'Priority': 1,
            'FixedRate': 0.05,      # 5% of requests
            'ReservoirSize': 10,    # Always trace 10 req/s regardless of %
            'ServiceName': 'orders-service',
            'ServiceType': 'AWS::ECS::Container',
            'Host': '*',
            'HTTPMethod': 'POST',
            'URLPath': '/api/orders',
            'ResourceARN': '*',
            'Version': 1
        }
    )


# ============================================================================
# SECTION 15: CloudFormation — Infrastructure as Code
# ============================================================================
# KEY CONCEPTS:
#   - Templates: YAML/JSON with Parameters, Mappings, Conditions, Resources, Outputs
#   - Stacks: deployed unit; nested stacks for modularity
#   - Change Sets: preview changes before applying (like a plan)
#   - Stack policies: protect critical resources from unintended updates
#   - Drift detection: identify manual changes vs template
#   - DeletionPolicy: Retain, Delete, Snapshot
#   - CloudFormation StackSets: deploy across accounts/regions
#   - Custom Resources: Lambda-backed resources for anything not natively supported
#   - cfn-init, cfn-signal for EC2 bootstrapping
#   - AWS CDK: write CF templates in TypeScript/Python (L1=CFN, L2=opinionated, L3=patterns)
# ============================================================================

# --- 15.1 CloudFormation template as Python dict (used by CDK or boto3 directly) ---
ECS_SERVICE_TEMPLATE = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "ECS Fargate microservice with ALB",
    "Parameters": {
        "Environment": {
            "Type": "String",
            "AllowedValues": ["dev", "staging", "prod"],
            "Default": "dev"
        },
        "DockerImage": {"Type": "String", "Description": "ECR image URI"}
    },
    "Conditions": {
        "IsProd": {"Fn::Equals": [{"Ref": "Environment"}, "prod"]}
    },
    "Resources": {
        "ECSCluster": {
            "Type": "AWS::ECS::Cluster",
            "Properties": {
                "ClusterName": {"Fn::Sub": "${Environment}-orders-cluster"},
                "ClusterSettings": [{"Name": "containerInsights", "Value": "enabled"}]
            }
        },
        "TaskDefinition": {
            "Type": "AWS::ECS::TaskDefinition",
            "Properties": {
                "Family": "orders-service",
                "RequiresCompatibilities": ["FARGATE"],
                "NetworkMode": "awsvpc",
                "Cpu": {"Fn::If": ["IsProd", "1024", "512"]},  # Conditional
                "Memory": {"Fn::If": ["IsProd", "2048", "1024"]},
                "ContainerDefinitions": [{
                    "Name": "orders-service",
                    "Image": {"Ref": "DockerImage"},
                    "PortMappings": [{"ContainerPort": 8080}],
                    "LogConfiguration": {
                        "LogDriver": "awslogs",
                        "Options": {
                            "awslogs-group": {"Fn::Sub": "/ecs/${Environment}/orders-service"},
                            "awslogs-region": {"Ref": "AWS::Region"},
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }]
            }
        },
        "ECSService": {
            "Type": "AWS::ECS::Service",
            "DependsOn": ["ALBListenerRule"],
            "Properties": {
                "Cluster": {"Ref": "ECSCluster"},
                "TaskDefinition": {"Ref": "TaskDefinition"},
                "DesiredCount": {"Fn::If": ["IsProd", 3, 1]},
                "LaunchType": "FARGATE",
                "DeploymentConfiguration": {
                    "MaximumPercent": 200,
                    "MinimumHealthyPercent": 100
                }
            }
        }
    },
    "Outputs": {
        "ClusterName": {
            "Value": {"Ref": "ECSCluster"},
            "Export": {"Name": {"Fn::Sub": "${Environment}-ClusterName"}}
        }
    }
}

# --- 15.2 Deploy stack with change set ---
def deploy_stack_with_changeset(stack_name, template_body, parameters):
    cf = boto3.client('cloudformation')

    changeset_name = f'{stack_name}-cs-{int(__import__("time").time())}'

    cf.create_change_set(
        StackName=stack_name,
        ChangeSetName=changeset_name,
        TemplateBody=json.dumps(template_body),
        Parameters=[{'ParameterKey': k, 'ParameterValue': v} for k, v in parameters.items()],
        Capabilities=['CAPABILITY_NAMED_IAM'],
        ChangeSetType='CREATE'  # or 'UPDATE' for existing stack
    )

    # Wait for change set to be ready
    waiter = cf.get_waiter('change_set_create_complete')
    waiter.wait(StackName=stack_name, ChangeSetName=changeset_name)

    # Review changes (in practice, do this manually or in CI)
    changes = cf.describe_change_set(StackName=stack_name, ChangeSetName=changeset_name)
    print(json.dumps(changes['Changes'], indent=2, default=str))

    # Execute
    cf.execute_change_set(StackName=stack_name, ChangeSetName=changeset_name)

    # Wait for completion
    waiter = cf.get_waiter('stack_create_complete')
    waiter.wait(StackName=stack_name)

# --- 15.3 CloudFormation Custom Resource (Lambda-backed) ---
def custom_resource_handler(event, context):
    """
    Lambda function to handle CloudFormation Custom Resource lifecycle.
    CloudFormation calls this on Create/Update/Delete.
    Must send response to pre-signed S3 URL.
    """
    import urllib.request

    request_type = event['RequestType']  # Create | Update | Delete
    response_data = {}
    physical_id = event.get('PhysicalResourceId', 'custom-resource-id')

    try:
        if request_type == 'Create':
            # Perform create action (e.g., create Route53 record, call external API)
            response_data['Result'] = 'Created successfully'
            physical_id = 'my-custom-resource-123'
        elif request_type == 'Update':
            response_data['Result'] = 'Updated successfully'
        elif request_type == 'Delete':
            response_data['Result'] = 'Deleted successfully'

        status = 'SUCCESS'
    except Exception as e:
        status = 'FAILED'
        response_data['Error'] = str(e)

    # Send response to CloudFormation
    response_body = json.dumps({
        'Status': status,
        'Reason': str(response_data.get('Error', '')),
        'PhysicalResourceId': physical_id,
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'Data': response_data
    }).encode()

    req = urllib.request.Request(
        url=event['ResponseURL'],
        data=response_body,
        headers={'Content-Type': '', 'Content-Length': len(response_body)},
        method='PUT'
    )
    urllib.request.urlopen(req)


# ============================================================================
# BONUS: KEY ARCHITECTURAL PATTERNS & INTERVIEW SCENARIOS
# ============================================================================

# --- Pattern A: Event-driven microservices with outbox pattern ---
# Problem: ensure both DB write and SQS publish happen atomically
# Solution: write to "outbox" table in same DynamoDB transaction,
# a Streams→Lambda worker publishes to SNS/SQS and deletes from outbox

def outbox_pattern_place_order(order_id, customer_id):
    dynamodb_resource.meta.client.transact_write_items(TransactItems=[
        {'Put': {'TableName': 'Orders', 'Item': {
            'PK': {'S': f'ORDER#{order_id}'}, 'SK': {'S': 'METADATA'},
            'status': {'S': 'PENDING'}, 'customerId': {'S': customer_id}
        }}},
        {'Put': {'TableName': 'OutboxEvents', 'Item': {  # Outbox entry
            'PK': {'S': f'EVENT#{order_id}'}, 'SK': {'S': 'ORDER_CREATED'},
            'payload': {'S': json.dumps({'orderId': order_id})},
            'published': {'BOOL': False}
        }}}
    ])
    # DynamoDB Stream → Lambda reads OutboxEvents, publishes to SNS, deletes entry

# --- Pattern B: Circuit breaker with Lambda + Parameter Store ---
# Store circuit state in SSM Parameter Store
# Lambda checks state before calling downstream service
def check_circuit_breaker(service_name):
    ssm = boto3.client('ssm')
    param = ssm.get_parameter(Name=f'/circuit-breaker/{service_name}')
    state = json.loads(param['Parameter']['Value'])

    if state['status'] == 'OPEN':
        failure_time = state['openedAt']
        import time
        if time.time() - failure_time > 60:  # half-open after 60s
            return 'HALF_OPEN'
        raise Exception(f'Circuit breaker OPEN for {service_name}')
    return state['status']

# --- Pattern C: Saga pattern with Step Functions ---
# Orchestrate distributed transaction with compensating transactions on failure
SAGA_ORDER_ASL = {
    "StartAt": "ReserveInventory",
    "States": {
        "ReserveInventory": {
            "Type": "Task", "Resource": "arn:aws:lambda:::function:reserve-inventory",
            "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "CompensateOrder"}],
            "Next": "ChargePayment"
        },
        "ChargePayment": {
            "Type": "Task", "Resource": "arn:aws:lambda:::function:charge-payment",
            "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "ReleaseInventory"}],
            "Next": "ShipOrder"
        },
        "ShipOrder": {"Type": "Task", "Resource": "arn:aws:lambda:::function:ship-order", "End": True},
        # Compensating transactions (rollback)
        "ReleaseInventory": {
            "Type": "Task", "Resource": "arn:aws:lambda:::function:release-inventory", "Next": "CompensateOrder"
        },
        "CompensateOrder": {"Type": "Task", "Resource": "arn:aws:lambda:::function:cancel-order", "End": True}
    }
}

# --- Pattern D: API rate limiting via Lambda + DynamoDB ---
def check_rate_limit(user_id, limit_per_minute=100):
    import time
    window = int(time.time() / 60)  # 1-minute window
    key = {'PK': {'S': f'RATE#{user_id}'}, 'SK': {'S': str(window)}}

    try:
        response = dynamodb_resource.meta.client.update_item(
            TableName='RateLimits',
            Key=key,
            UpdateExpression='ADD hit_count :one SET #ttl = :ttl',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={
                ':one': {'N': '1'},
                ':limit': {'N': str(limit_per_minute)},
                ':ttl': {'N': str(int(time.time()) + 120)},  # auto-expire after 2 min
            },
            ConditionExpression='attribute_not_exists(hit_count) OR hit_count < :limit',
            ReturnValues='UPDATED_NEW'
        )
        return True
    except dynamodb_resource.meta.client.exceptions.ConditionalCheckFailedException:
        return False  # Rate limit exceeded

# ============================================================================
# QUICK-REFERENCE INTERVIEW CHEAT SHEET
# ============================================================================
CHEAT_SHEET = """
=== VPC ===
Security Group: stateful, instance-level, ALLOW only
NACL: stateless, subnet-level, ALLOW + DENY, must define both inbound + outbound
VPC Peering: 1-1, non-transitive | Transit Gateway: N-N, transitive, cross-account
Gateway Endpoint: S3, DynamoDB — free | Interface Endpoint: most services — costs money

=== EBS vs EFS ===
EBS: single AZ, single EC2, block storage | EFS: multi-AZ, multi-EC2, NFS shared
gp3: default choice (cheaper, configurable IOPS) | io2: databases needing high IOPS

=== ELB ===
ALB: L7, path/header routing, microservices | NLB: L4, static IP, low latency, TCP/UDP

=== DynamoDB ===
Hot partition fix: high-cardinality keys + write sharding
GSI: new PK, eventual consistency | LSI: same PK, strong consistency
Transactions: up to 100 items, ACID | TTL: auto-expire in seconds

=== Lambda ===
Cold start fixes: provisioned concurrency, small package, clients outside handler, SnapStart
SQS trigger: ReportBatchItemFailures for partial batch failure (retry only failed messages)
Reserved concurrency: cap concurrent executions (protects downstream services)

=== SQS ===
Standard: at-least-once, best-effort order | FIFO: exactly-once, ordered, 3000 TPS
Visibility timeout: must be >= Lambda timeout | DLQ: after maxReceiveCount failures

=== SNS vs SQS vs EventBridge ===
SNS: push fan-out, no persistence | SQS: pull, persisted, backpressure
EventBridge: complex routing rules, 90+ AWS sources, schema registry

=== Step Functions ===
Standard: long-running, audit trail, 1yr max | Express: high-TPS, 5min max, cheaper
waitForTaskToken: pause state machine, resume via callback
Saga pattern: use Catch states for compensating transactions

=== CloudWatch ===
Default metrics: 5min | Detailed: 1min | Custom: any interval
Logs Insights: SQL-like, query across log groups | Container Insights: ECS/EKS metrics

=== X-Ray ===
Annotations: indexed, use for filtering | Metadata: not indexed, debug details
Service Map: visualize latency/errors between services
Sampling: reservoir (fixed TPS) + fixed rate (%)

=== CloudFormation ===
Change Sets: preview before apply | Drift Detection: find manual changes
StackSets: multi-account/region | Custom Resources: Lambda-backed for unsupported resources
DeletionPolicy Retain: keep resource after stack delete (critical for prod DBs)

=== Microservice Patterns ===
Outbox: DynamoDB transaction + Streams → SNS (atomic publish)
Saga: Step Functions with compensating transactions
Circuit Breaker: SSM/DynamoDB for state + Lambda
CQRS: DynamoDB Streams → separate read model (OpenSearch/DynamoDB GSI)
Sidecar: separate container in ECS task (logging, service mesh proxy)
"""

print(CHEAT_SHEET)
