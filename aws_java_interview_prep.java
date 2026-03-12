// ============================================================================
// AWS INTERVIEW PREPARATION — JAVA DEVELOPER GUIDE
// For 8+ Years SDE Experience
// AWS SDK for Java v2 (software.amazon.awssdk)
// ============================================================================
//
// Maven dependencies (add to pom.xml):
//
// <dependencyManagement>
//   <dependencies>
//     <dependency>
//       <groupId>software.amazon.awssdk</groupId>
//       <artifactId>bom</artifactId>
//       <version>2.25.0</version>
//       <type>pom</type>
//       <scope>import</scope>
//     </dependency>
//   </dependencies>
// </dependencyManagement>
//
// Then individual services: s3, dynamodb, sqs, sns, lambda, sfn,
//   ec2, elasticloadbalancingv2, apigateway, cognitoidentityprovider,
//   route53, secretsmanager, sts
//
// ============================================================================

import software.amazon.awssdk.auth.credentials.*;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.core.sync.ResponseTransformer;
import software.amazon.awssdk.core.async.AsyncRequestBody;
import software.amazon.awssdk.core.waiters.WaiterResponse;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.*;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.awssdk.services.s3.presigner.*;
import software.amazon.awssdk.services.dynamodb.*;
import software.amazon.awssdk.services.dynamodb.model.*;
import software.amazon.awssdk.services.sqs.*;
import software.amazon.awssdk.services.sqs.model.*;
import software.amazon.awssdk.services.sns.*;
import software.amazon.awssdk.services.sns.model.*;
import software.amazon.awssdk.services.lambda.*;
import software.amazon.awssdk.services.lambda.model.*;
import software.amazon.awssdk.services.sfn.*;
import software.amazon.awssdk.services.sfn.model.*;
import software.amazon.awssdk.services.ec2.*;
import software.amazon.awssdk.services.ec2.model.*;
import software.amazon.awssdk.services.elasticloadbalancingv2.*;
import software.amazon.awssdk.services.elasticloadbalancingv2.model.*;
import software.amazon.awssdk.services.cognitoidentityprovider.*;
import software.amazon.awssdk.services.cognitoidentityprovider.model.*;
import software.amazon.awssdk.services.route53.*;
import software.amazon.awssdk.services.route53.model.*;
import software.amazon.awssdk.services.secretsmanager.*;
import software.amazon.awssdk.services.secretsmanager.model.*;
import software.amazon.awssdk.services.sts.*;
import software.amazon.awssdk.services.sts.model.*;

import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import com.amazonaws.services.lambda.runtime.events.*;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;


// ============================================================================
// SECTION 1: AWS SDK v2 FUNDAMENTALS & CLIENT SETUP
// ============================================================================

class AwsSdkSetup {

    /*
     * SDK v2 Key Differences from v1:
     * - Immutable request/response objects (Builder pattern everywhere)
     * - Separate sync and async clients
     * - HTTP clients: Apache (sync), Netty (async)
     * - Auto-configured credentials chain
     * - Enhanced DynamoDB client (higher-level)
     * - Waiters built-in for polling
     */

    // ── Credential Chain (DefaultCredentialsProvider) ─────────────────────────
    // Order of resolution:
    //   1. Java system properties (aws.accessKeyId, aws.secretAccessKey)
    //   2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    //   3. Web identity token (EKS / OIDC)
    //   4. ~/.aws/credentials profile file
    //   5. ECS container credentials
    //   6. EC2 instance profile (IMDSv2)

    public S3Client buildS3Client() {
        // Default (uses credential chain above — RECOMMENDED in production)
        return S3Client.builder()
            .region(Region.US_EAST_1)
            .build();
    }

    public S3Client buildS3ClientExplicit() {
        // Explicit credentials (use only for local dev / testing)
        AwsCredentials credentials = AwsBasicCredentials.create(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        );
        return S3Client.builder()
            .region(Region.US_EAST_1)
            .credentialsProvider(StaticCredentialsProvider.create(credentials))
            .build();
    }

    // ── Assuming a Role (cross-account / least privilege) ─────────────────────

    public StsClient buildStsClient() {
        return StsClient.builder().region(Region.US_EAST_1).build();
    }

    public AwsCredentialsProvider assumeRole(String roleArn, String sessionName) {
        StsClient sts = buildStsClient();
        AssumeRoleRequest request = AssumeRoleRequest.builder()
            .roleArn(roleArn)
            .roleSessionName(sessionName)
            .durationSeconds(3600)
            .build();

        AssumeRoleResponse response = sts.assumeRole(request);
        Credentials tempCreds = response.credentials();

        return StaticCredentialsProvider.create(
            AwsSessionCredentials.create(
                tempCreds.accessKeyId(),
                tempCreds.secretAccessKey(),
                tempCreds.sessionToken()
            )
        );
    }

    // ── Async Client Pattern ──────────────────────────────────────────────────

    public S3AsyncClient buildAsyncS3() {
        return S3AsyncClient.builder()
            .region(Region.US_EAST_1)
            .build();
        // Returns CompletableFuture<T> for all operations
    }
}


// ============================================================================
// SECTION 2: AMAZON S3
// ============================================================================

class S3Examples {

    private final S3Client s3;

    S3Examples() {
        this.s3 = S3Client.builder().region(Region.US_EAST_1).build();
    }

    // ── Bucket Operations ─────────────────────────────────────────────────────

    public void createBucket(String bucketName) {
        s3.createBucket(CreateBucketRequest.builder()
            .bucket(bucketName)
            .build());

        // Wait until bucket exists (waiter — polls until state reached)
        s3.waiter().waitUntilBucketExists(
            HeadBucketRequest.builder().bucket(bucketName).build()
        );
        System.out.println("Bucket ready: " + bucketName);
    }

    public void listBuckets() {
        s3.listBuckets().buckets().forEach(b ->
            System.out.println(b.name() + " created: " + b.creationDate())
        );
    }

    public void deleteBucket(String bucketName) {
        // Must empty bucket first
        listAndDeleteObjects(bucketName);
        s3.deleteBucket(DeleteBucketRequest.builder().bucket(bucketName).build());
    }

    // ── Object Upload ─────────────────────────────────────────────────────────

    public void uploadFile(String bucket, String key, Path filePath) {
        PutObjectRequest request = PutObjectRequest.builder()
            .bucket(bucket)
            .key(key)
            .contentType("application/octet-stream")
            .serverSideEncryption(ServerSideEncryption.AES256) // SSE-S3 encryption
            .storageClass(StorageClass.STANDARD_IA)            // Infrequent access tier
            .metadata(Map.of("uploaded-by", "java-sdk", "version", "1.0"))
            .build();

        s3.putObject(request, RequestBody.fromFile(filePath));
        System.out.println("Uploaded: " + key);
    }

    public void uploadString(String bucket, String key, String content) {
        s3.putObject(
            PutObjectRequest.builder().bucket(bucket).key(key).build(),
            RequestBody.fromString(content)
        );
    }

    // Multipart upload — for files > 100MB (required for > 5GB)
    public void multipartUpload(String bucket, String key, Path largeFile) throws IOException {
        // Initiate
        CreateMultipartUploadResponse initiated = s3.createMultipartUpload(
            CreateMultipartUploadRequest.builder().bucket(bucket).key(key).build()
        );
        String uploadId = initiated.uploadId();

        List<CompletedPart> completedParts = new ArrayList<>();
        long partSize = 5 * 1024 * 1024L;  // 5MB minimum per part
        byte[] fileBytes = Files.readAllBytes(largeFile);
        int totalParts = (int) Math.ceil((double) fileBytes.length / partSize);

        try {
            for (int partNum = 1; partNum <= totalParts; partNum++) {
                int start = (partNum - 1) * (int) partSize;
                int end   = Math.min(start + (int) partSize, fileBytes.length);
                byte[] partData = Arrays.copyOfRange(fileBytes, start, end);

                UploadPartResponse partResp = s3.uploadPart(
                    UploadPartRequest.builder()
                        .bucket(bucket).key(key)
                        .uploadId(uploadId)
                        .partNumber(partNum)
                        .build(),
                    RequestBody.fromBytes(partData)
                );

                completedParts.add(CompletedPart.builder()
                    .partNumber(partNum)
                    .eTag(partResp.eTag())
                    .build());
            }

            // Complete
            s3.completeMultipartUpload(CompleteMultipartUploadRequest.builder()
                .bucket(bucket).key(key).uploadId(uploadId)
                .multipartUpload(CompletedMultipartUpload.builder()
                    .parts(completedParts).build())
                .build());

        } catch (Exception e) {
            // Always abort on failure — otherwise you pay for incomplete parts!
            s3.abortMultipartUpload(AbortMultipartUploadRequest.builder()
                .bucket(bucket).key(key).uploadId(uploadId).build());
            throw e;
        }
    }

    // ── Object Download ───────────────────────────────────────────────────────

    public byte[] downloadObject(String bucket, String key) {
        GetObjectRequest request = GetObjectRequest.builder()
            .bucket(bucket).key(key).build();

        return s3.getObjectAsBytes(request).asByteArray();
    }

    public void downloadToFile(String bucket, String key, Path targetPath) {
        s3.getObject(
            GetObjectRequest.builder().bucket(bucket).key(key).build(),
            ResponseTransformer.toFile(targetPath)
        );
    }

    // ── List & Paginate Objects ───────────────────────────────────────────────

    public List<String> listAllObjects(String bucket, String prefix) {
        List<String> keys = new ArrayList<>();

        // Paginator — automatically handles continuation tokens
        ListObjectsV2Iterable pages = s3.listObjectsV2Paginator(
            ListObjectsV2Request.builder()
                .bucket(bucket)
                .prefix(prefix)
                .maxKeys(1000)
                .build()
        );

        pages.contents().forEach(obj -> {
            keys.add(obj.key());
            System.out.printf("Key: %s, Size: %d bytes, Modified: %s%n",
                obj.key(), obj.size(), obj.lastModified());
        });

        return keys;
    }

    // ── Delete Objects ────────────────────────────────────────────────────────

    public void listAndDeleteObjects(String bucket) {
        ListObjectsV2Iterable pages = s3.listObjectsV2Paginator(
            ListObjectsV2Request.builder().bucket(bucket).build()
        );

        List<ObjectIdentifier> toDelete = new ArrayList<>();
        pages.contents().forEach(obj ->
            toDelete.add(ObjectIdentifier.builder().key(obj.key()).build())
        );

        if (!toDelete.isEmpty()) {
            s3.deleteObjects(DeleteObjectsRequest.builder()
                .bucket(bucket)
                .delete(Delete.builder().objects(toDelete).quiet(true).build())
                .build());
        }
    }

    // ── Presigned URLs ────────────────────────────────────────────────────────
    // Share time-limited access to private objects without exposing credentials

    public URL generatePresignedGetUrl(String bucket, String key, Duration expiry) {
        try (S3Presigner presigner = S3Presigner.builder()
                .region(Region.US_EAST_1).build()) {

            PresignedGetObjectRequest presigned = presigner.presignGetObject(
                GetObjectPresignRequest.builder()
                    .signatureDuration(expiry)
                    .getObjectRequest(GetObjectRequest.builder()
                        .bucket(bucket).key(key).build())
                    .build()
            );

            return presigned.url();
        }
    }

    public URL generatePresignedPutUrl(String bucket, String key, Duration expiry) {
        try (S3Presigner presigner = S3Presigner.builder()
                .region(Region.US_EAST_1).build()) {

            PresignedPutObjectRequest presigned = presigner.presignPutObject(
                PutObjectPresignRequest.builder()
                    .signatureDuration(expiry)
                    .putObjectRequest(PutObjectRequest.builder()
                        .bucket(bucket).key(key).build())
                    .build()
            );

            return presigned.url();
        }
    }

    // ── Bucket Policy & Versioning ────────────────────────────────────────────

    public void enableVersioning(String bucket) {
        s3.putBucketVersioning(PutBucketVersioningRequest.builder()
            .bucket(bucket)
            .versioningConfiguration(VersioningConfiguration.builder()
                .status(BucketVersioningStatus.ENABLED)
                .build())
            .build());
    }

    public void setBucketLifecyclePolicy(String bucket) {
        // Auto-transition to cheaper storage, then expire
        s3.putBucketLifecycleConfiguration(PutBucketLifecycleConfigurationRequest.builder()
            .bucket(bucket)
            .lifecycleConfiguration(BucketLifecycleConfiguration.builder()
                .rules(LifecycleRule.builder()
                    .id("archive-old-objects")
                    .status(ExpirationStatus.ENABLED)
                    .filter(LifecycleRuleFilter.builder().prefix("logs/").build())
                    .transitions(Transition.builder()
                        .days(30)
                        .storageClass(TransitionStorageClass.STANDARD_IA)
                        .build())
                    .expiration(LifecycleExpiration.builder().days(365).build())
                    .build())
                .build())
            .build());
    }

    public void blockPublicAccess(String bucket) {
        s3.putPublicAccessBlock(PutPublicAccessBlockRequest.builder()
            .bucket(bucket)
            .publicAccessBlockConfiguration(PublicAccessBlockConfiguration.builder()
                .blockPublicAcls(true)
                .blockPublicPolicy(true)
                .ignorePublicAcls(true)
                .restrictPublicBuckets(true)
                .build())
            .build());
    }

    /*
     * S3 Interview Key Points:
     * - Storage classes: Standard → Standard-IA → One Zone-IA → Glacier Instant →
     *   Glacier Flexible (hours) → Glacier Deep Archive (12h) → Intelligent-Tiering
     * - Durability: 99.999999999% (11 nines); Availability: 99.99%
     * - Consistency: Strong read-after-write consistency (since Dec 2020)
     * - Max object size: 5TB; Multipart required for > 5GB
     * - S3 Transfer Acceleration: uses CloudFront edge for upload speed
     * - Cross-Region Replication (CRR): async replication, requires versioning
     * - Object Lock: WORM compliance (Governance / Compliance mode)
     * - S3 Event Notifications: trigger Lambda, SQS, SNS on PUT/DELETE
     * - S3 Select: SQL-like queries directly on S3 objects (reduces data transfer)
     */
}


// ============================================================================
// SECTION 3: AWS LAMBDA
// ============================================================================

// ── Lambda Handler Implementations ───────────────────────────────────────────

// Simple string handler
class SimpleHandler implements RequestHandler<String, String> {
    @Override
    public String handleRequest(String input, Context context) {
        context.getLogger().log("Input: " + input);
        context.getLogger().log("Remaining time: " + context.getRemainingTimeInMillis());
        return "Processed: " + input.toUpperCase();
    }
}

// API Gateway proxy handler
class ApiGatewayHandler implements RequestHandler<APIGatewayProxyRequestEvent, APIGatewayProxyResponseEvent> {

    @Override
    public APIGatewayProxyResponseEvent handleRequest(
            APIGatewayProxyRequestEvent event, Context context) {

        context.getLogger().log("Method: " + event.getHttpMethod());
        context.getLogger().log("Path: " + event.getPath());

        String body = event.getBody();
        Map<String, String> headers = event.getHeaders();
        Map<String, String> pathParams = event.getPathParameters();
        Map<String, String> queryParams = event.getQueryStringParameters();

        // Process request
        String responseBody = """
            {"message": "Hello from Lambda!", "path": "%s"}
            """.formatted(event.getPath());

        return new APIGatewayProxyResponseEvent()
            .withStatusCode(200)
            .withHeaders(Map.of(
                "Content-Type", "application/json",
                "Access-Control-Allow-Origin", "*"  // CORS
            ))
            .withBody(responseBody);
    }
}

// SQS event handler
class SqsEventHandler implements RequestHandler<SQSEvent, Void> {

    @Override
    public Void handleRequest(SQSEvent event, Context context) {
        for (SQSEvent.SQSMessage message : event.getRecords()) {
            try {
                processMessage(message.getBody(), context);
            } catch (Exception e) {
                // Throwing here sends the batch to DLQ (if configured)
                // For partial batch failures, use SQS reportBatchItemFailures
                context.getLogger().log("FAILED messageId: " + message.getMessageId());
                throw e;
            }
        }
        return null;
    }

    private void processMessage(String body, Context context) {
        context.getLogger().log("Processing: " + body);
    }
}

// S3 event handler
class S3EventHandler implements RequestHandler<S3Event, String> {

    @Override
    public String handleRequest(S3Event s3Event, Context context) {
        for (S3EventNotification.S3EventNotificationRecord record : s3Event.getRecords()) {
            String bucket = record.getS3().getBucket().getName();
            String key    = record.getS3().getObject().getUrlDecodedKey();
            String event  = record.getEventName();
            context.getLogger().log(event + " → s3://" + bucket + "/" + key);
        }
        return "OK";
    }
}

// Lambda with DynamoDB Streams
class DynamoStreamHandler implements RequestHandler<DynamodbEvent, Void> {
    @Override
    public Void handleRequest(DynamodbEvent event, Context context) {
        for (DynamodbEvent.DynamodbStreamRecord record : event.getRecords()) {
            String eventType = record.getEventName(); // INSERT, MODIFY, REMOVE
            StreamRecord streamRecord = record.getDynamodb();

            Map<String, AttributeValue> newImage = streamRecord.getNewImage(); // After change
            Map<String, AttributeValue> oldImage = streamRecord.getOldImage(); // Before change

            context.getLogger().log(eventType + " event on: " +
                streamRecord.getKeys().get("PK").getS());
        }
        return null;
    }
}

// Lambda operational concerns (SDK usage from within Lambda)
class LambdaInternals {

    /*
     * Lambda Execution Model:
     * ─────────────────────
     * INIT phase:  static code runs once per container (cold start)
     * INVOKE phase: handler() called per invocation (warm start reuses container)
     *
     * Initialize expensive resources OUTSIDE handler to reuse across invocations!
     */

    // GOOD: initialized once at container level (warm invocations reuse this)
    private static final S3Client S3 = S3Client.builder()
        .region(Region.US_EAST_1)
        .build();

    private static final DynamoDbClient DYNAMO = DynamoDbClient.builder()
        .region(Region.US_EAST_1)
        .build();

    /*
     * Lambda Key Concepts for Interviews:
     * ────────────────────────────────────
     * Cold start: container spun up → JVM start → class load → static init → handler
     *   - Mitigate: SnapStart (Java 21), provisioned concurrency, smaller JARs, GraalVM native
     *   - Use AWS Lambda Powertools for Java: logging, tracing, metrics
     *
     * Concurrency models:
     *   - Reserved concurrency: hard limit for a function (no throttling others)
     *   - Provisioned concurrency: pre-warmed instances, eliminates cold start
     *   - Account concurrency limit default: 1000 (soft limit, increasable)
     *
     * Invocation types:
     *   - RequestResponse (sync): API GW, SDK direct
     *   - Event (async): S3, SNS, EventBridge; retries 2x, then DLQ
     *   - DRY_RUN: validate without execution
     *
     * Triggers: API Gateway, ALB, S3, SQS, SNS, DynamoDB Streams,
     *           Kinesis, EventBridge, Cognito, SES, CloudFront
     *
     * Limits: 15min timeout, 10GB RAM, 512MB /tmp (up to 10GB), 250MB deployment pkg
     *
     * Lambda Layers: share common dependencies (JARs) across functions
     *
     * Environment variables: encrypted with KMS; use Secrets Manager for secrets
     *
     * X-Ray tracing: automatic with active tracing; use AWSXRay SDK for subsegments
     */
}


// ============================================================================
// SECTION 4: AMAZON DYNAMODB
// ============================================================================

class DynamoDbExamples {

    private final DynamoDbClient dynamo;

    DynamoDbExamples() {
        this.dynamo = DynamoDbClient.builder().region(Region.US_EAST_1).build();
    }

    // ── Table Design — Single Table Design (STD) ──────────────────────────────
    //
    // Access patterns drive schema (unlike relational DBs!)
    // PK (partition key) + SK (sort key) → composite primary key
    // Use overloaded GSI (Global Secondary Index) for flexible access patterns
    //
    // Example single table for e-commerce:
    //   PK=USER#123,    SK=PROFILE          → User profile
    //   PK=USER#123,    SK=ORDER#2024-001   → User's orders (sorted by date)
    //   PK=ORDER#001,   SK=ITEM#PRODUCT#99  → Order line items
    //   PK=PRODUCT#99,  SK=METADATA         → Product details

    // ── Create Table ──────────────────────────────────────────────────────────

    public void createTable(String tableName) {
        CreateTableRequest request = CreateTableRequest.builder()
            .tableName(tableName)
            .billingMode(BillingMode.PAY_PER_REQUEST)  // On-demand; or PROVISIONED with RCU/WCU
            .keySchema(
                KeySchemaElement.builder().attributeName("PK").keyType(KeyType.HASH).build(),
                KeySchemaElement.builder().attributeName("SK").keyType(KeyType.RANGE).build()
            )
            .attributeDefinitions(
                AttributeDefinition.builder().attributeName("PK").attributeType(ScalarAttributeType.S).build(),
                AttributeDefinition.builder().attributeName("SK").attributeType(ScalarAttributeType.S).build(),
                AttributeDefinition.builder().attributeName("GSI1PK").attributeType(ScalarAttributeType.S).build(),
                AttributeDefinition.builder().attributeName("GSI1SK").attributeType(ScalarAttributeType.S).build()
            )
            .globalSecondaryIndexes(GlobalSecondaryIndex.builder()
                .indexName("GSI1")
                .keySchema(
                    KeySchemaElement.builder().attributeName("GSI1PK").keyType(KeyType.HASH).build(),
                    KeySchemaElement.builder().attributeName("GSI1SK").keyType(KeyType.RANGE).build()
                )
                .projection(Projection.builder().projectionType(ProjectionType.ALL).build())
                .build())
            .streamSpecification(StreamSpecification.builder()
                .streamEnabled(true)
                .streamViewType(StreamViewType.NEW_AND_OLD_IMAGES)
                .build())
            .build();

        dynamo.createTable(request);
        dynamo.waiter().waitUntilTableExists(
            DescribeTableRequest.builder().tableName(tableName).build()
        );
    }

    // ── PutItem ───────────────────────────────────────────────────────────────

    public void putItem(String tableName) {
        Map<String, AttributeValue> item = new HashMap<>();
        item.put("PK",      av("USER#123"));
        item.put("SK",      av("PROFILE"));
        item.put("name",    av("Alice"));
        item.put("email",   av("alice@example.com"));
        item.put("age",     AttributeValue.builder().n("30").build());
        item.put("active",  AttributeValue.builder().bool(true).build());
        item.put("tags",    AttributeValue.builder().ss("admin", "user").build()); // String set
        item.put("scores",  AttributeValue.builder().ns("95", "87", "100").build()); // Number set
        item.put("ttl",     AttributeValue.builder().n(
            String.valueOf(Instant.now().plusSeconds(86400).getEpochSecond())
        ).build()); // TTL — epoch seconds, DynamoDB auto-deletes expired items

        PutItemRequest request = PutItemRequest.builder()
            .tableName(tableName)
            .item(item)
            // Conditional expression — fail if PK already exists (idempotency)
            .conditionExpression("attribute_not_exists(PK)")
            .build();

        try {
            dynamo.putItem(request);
        } catch (ConditionalCheckFailedException e) {
            System.out.println("Item already exists!");
        }
    }

    private AttributeValue av(String value) {
        return AttributeValue.builder().s(value).build();
    }

    // ── GetItem ───────────────────────────────────────────────────────────────

    public Map<String, AttributeValue> getItem(String tableName, String pk, String sk) {
        GetItemResponse response = dynamo.getItem(GetItemRequest.builder()
            .tableName(tableName)
            .key(Map.of(
                "PK", av(pk),
                "SK", av(sk)
            ))
            .consistentRead(true)  // Strong consistency (2x RCU cost vs eventually consistent)
            .projectionExpression("name, email, age")  // Only fetch needed attributes
            .build());

        return response.hasItem() ? response.item() : null;
    }

    // ── UpdateItem ────────────────────────────────────────────────────────────

    public void updateItem(String tableName, String pk, String sk, String newEmail, int newAge) {
        UpdateItemRequest request = UpdateItemRequest.builder()
            .tableName(tableName)
            .key(Map.of("PK", av(pk), "SK", av(sk)))
            .updateExpression("SET email = :email, age = :age, updatedAt = :now ADD loginCount :one")
            .conditionExpression("attribute_exists(PK)")  // Only update if item exists
            .expressionAttributeValues(Map.of(
                ":email", av(newEmail),
                ":age",   AttributeValue.builder().n(String.valueOf(newAge)).build(),
                ":now",   av(Instant.now().toString()),
                ":one",   AttributeValue.builder().n("1").build()
            ))
            .returnValues(ReturnValue.ALL_NEW)  // Return updated item
            .build();

        UpdateItemResponse response = dynamo.updateItem(request);
        System.out.println("Updated: " + response.attributes());
    }

    // ── DeleteItem ────────────────────────────────────────────────────────────

    public void deleteItem(String tableName, String pk, String sk) {
        dynamo.deleteItem(DeleteItemRequest.builder()
            .tableName(tableName)
            .key(Map.of("PK", av(pk), "SK", av(sk)))
            .conditionExpression("attribute_exists(PK)")
            .build());
    }

    // ── Query (efficient — uses index) ────────────────────────────────────────

    public List<Map<String, AttributeValue>> queryUserOrders(
            String tableName, String userId, String fromDate) {

        QueryRequest request = QueryRequest.builder()
            .tableName(tableName)
            .keyConditionExpression("PK = :pk AND begins_with(SK, :skPrefix)")
            .filterExpression("amount > :minAmount")  // Filter applied AFTER read (costs RCU!)
            .expressionAttributeValues(Map.of(
                ":pk",       av("USER#" + userId),
                ":skPrefix", av("ORDER#"),
                ":minAmount", AttributeValue.builder().n("100").build()
            ))
            .scanIndexForward(false)  // Descending order by SK
            .limit(20)               // Items to evaluate (not necessarily return!)
            .build();

        List<Map<String, AttributeValue>> results = new ArrayList<>();
        String lastEvaluatedKey = null;

        // Manual pagination
        do {
            QueryResponse response = dynamo.query(request);
            results.addAll(response.items());

            if (response.hasLastEvaluatedKey()) {
                // There are more pages — use paginator instead for simplicity
                request = request.toBuilder()
                    .exclusiveStartKey(response.lastEvaluatedKey())
                    .build();
            } else {
                break;
            }
        } while (true);

        return results;
    }

    // Automatic pagination with paginator
    public List<Map<String, AttributeValue>> queryWithPaginator(String tableName, String pk) {
        QueryIterable pages = dynamo.queryPaginator(QueryRequest.builder()
            .tableName(tableName)
            .keyConditionExpression("PK = :pk")
            .expressionAttributeValues(Map.of(":pk", av(pk)))
            .build());

        return pages.items().stream().collect(Collectors.toList());
    }

    // ── Scan (expensive — reads entire table) ─────────────────────────────────
    // Avoid Scan in production; prefer Query with well-designed GSIs

    public List<Map<String, AttributeValue>> parallelScan(String tableName, int segments) {
        List<Map<String, AttributeValue>> all = new ArrayList<>();

        // Parallel scan splits table into N segments for throughput
        for (int segment = 0; segment < segments; segment++) {
            final int seg = segment;
            ScanRequest request = ScanRequest.builder()
                .tableName(tableName)
                .totalSegments(segments)
                .segment(seg)
                .filterExpression("active = :true")
                .expressionAttributeValues(Map.of(":true", AttributeValue.builder().bool(true).build()))
                .build();

            dynamo.scanPaginator(request).items().forEach(all::add);
        }
        return all;
    }

    // ── Transactions ──────────────────────────────────────────────────────────
    // Up to 100 items, up to 4MB; 2x the RCU/WCU cost

    public void transferBalance(String tableName, String fromUser, String toUser, int amount) {
        dynamo.transactWriteItems(TransactWriteItemsRequest.builder()
            .transactItems(
                // Debit from source
                TransactWriteItem.builder()
                    .update(Update.builder()
                        .tableName(tableName)
                        .key(Map.of("PK", av(fromUser), "SK", av("BALANCE")))
                        .updateExpression("SET balance = balance - :amt")
                        .conditionExpression("balance >= :amt")  // Sufficient funds check
                        .expressionAttributeValues(Map.of(
                            ":amt", AttributeValue.builder().n(String.valueOf(amount)).build()
                        ))
                        .build())
                    .build(),
                // Credit to destination
                TransactWriteItem.builder()
                    .update(Update.builder()
                        .tableName(tableName)
                        .key(Map.of("PK", av(toUser), "SK", av("BALANCE")))
                        .updateExpression("SET balance = balance + :amt")
                        .expressionAttributeValues(Map.of(
                            ":amt", AttributeValue.builder().n(String.valueOf(amount)).build()
                        ))
                        .build())
                    .build()
            )
            .clientRequestToken(UUID.randomUUID().toString()) // Idempotency token
            .build());
    }

    // ── Batch Operations ──────────────────────────────────────────────────────

    public void batchWrite(String tableName, List<Map<String, AttributeValue>> items) {
        // Max 25 items per batch; SDK handles UnprocessedItems retry
        List<WriteRequest> writes = items.stream()
            .map(item -> WriteRequest.builder()
                .putRequest(PutRequest.builder().item(item).build())
                .build())
            .collect(Collectors.toList());

        // Split into chunks of 25
        for (int i = 0; i < writes.size(); i += 25) {
            List<WriteRequest> chunk = writes.subList(i, Math.min(i + 25, writes.size()));

            BatchWriteItemResponse response = dynamo.batchWriteItem(
                BatchWriteItemRequest.builder()
                    .requestItems(Map.of(tableName, chunk))
                    .build()
            );

            // Handle unprocessed items (exponential backoff recommended)
            if (!response.unprocessedItems().isEmpty()) {
                System.out.println("Unprocessed: " + response.unprocessedItems().size());
            }
        }
    }

    /*
     * DynamoDB Interview Key Points:
     * ─────────────────────────────
     * Capacity modes:
     *   - On-demand: auto-scales, pay per request, no planning needed
     *   - Provisioned: fixed RCU/WCU, cheaper at steady load, use auto-scaling
     *
     * Read consistency:
     *   - Eventually consistent: ~1 second lag, 1 RCU per 4KB
     *   - Strongly consistent: immediate, 2 RCU per 4KB
     *   - Transactions: 2 RCU per 4KB (TransactGetItems)
     *
     * GSI vs LSI:
     *   - LSI (Local): same PK, different SK; must be created at table creation; shares capacity
     *   - GSI (Global): different PK; can be added anytime; own capacity; eventual consistency
     *
     * Hot partition problem: too many requests on one PK → throttling
     *   - Solutions: write sharding (PK = USER#123#shard_{0-9}), caching (DAX), scatter/gather
     *
     * DynamoDB Accelerator (DAX):
     *   - In-memory cache, microsecond read latency, fully managed
     *   - Item cache + query cache
     *   - Drop-in replacement (same SDK calls via DaxClient)
     *
     * DynamoDB Streams + Lambda: event-driven patterns (CDC, materialized views)
     *
     * TTL: epoch seconds attribute; items auto-deleted within 48h (no WCU)
     *
     * Capacity units:
     *   - 1 RCU = 1 strongly consistent read of 4KB (or 2 eventually consistent)
     *   - 1 WCU = 1 write of 1KB
     */
}


// ============================================================================
// SECTION 5: AMAZON SQS
// ============================================================================

class SqsExamples {

    private final SqsClient sqs;

    SqsExamples() {
        this.sqs = SqsClient.builder().region(Region.US_EAST_1).build();
    }

    // ── Create Queue ──────────────────────────────────────────────────────────

    public String createStandardQueue(String queueName) {
        CreateQueueResponse response = sqs.createQueue(CreateQueueRequest.builder()
            .queueName(queueName)
            .attributes(Map.of(
                QueueAttributeName.VISIBILITY_TIMEOUT,        "30",    // Seconds
                QueueAttributeName.MESSAGE_RETENTION_PERIOD,  "86400", // 1 day (default 4 days)
                QueueAttributeName.RECEIVE_MESSAGE_WAIT_TIME_SECONDS, "20", // Long polling!
                QueueAttributeName.MAXIMUM_MESSAGE_SIZE,      "262144" // 256KB
            ))
            .build());
        return response.queueUrl();
    }

    public String createFifoQueue(String queueName) {
        // FIFO queue name MUST end with .fifo
        CreateQueueResponse response = sqs.createQueue(CreateQueueRequest.builder()
            .queueName(queueName + ".fifo")
            .attributes(Map.of(
                QueueAttributeName.FIFO_QUEUE,                       "true",
                QueueAttributeName.CONTENT_BASED_DEDUPLICATION,      "false", // We supply dedup ID
                QueueAttributeName.VISIBILITY_TIMEOUT,               "30",
                QueueAttributeName.DEDUPLICATION_SCOPE,              "messageGroup",
                QueueAttributeName.FIFO_THROUGHPUT_LIMIT,            "perMessageGroupId"
            ))
            .build());
        return response.queueUrl();
    }

    public String createQueueWithDLQ(String queueName, String dlqArn) {
        // Redrive policy: after maxReceiveCount failures, route to DLQ
        String redrivePolicy = """
            {"maxReceiveCount": "3", "deadLetterTargetArn": "%s"}
            """.formatted(dlqArn);

        CreateQueueResponse response = sqs.createQueue(CreateQueueRequest.builder()
            .queueName(queueName)
            .attributes(Map.of(
                QueueAttributeName.REDRIVE_POLICY, redrivePolicy
            ))
            .build());
        return response.queueUrl();
    }

    // ── Send Messages ─────────────────────────────────────────────────────────

    public void sendMessage(String queueUrl, String body) {
        SendMessageResponse response = sqs.sendMessage(SendMessageRequest.builder()
            .queueUrl(queueUrl)
            .messageBody(body)
            .delaySeconds(0)  // Delay visibility in queue (0–900 seconds)
            .messageAttributes(Map.of(
                "source", MessageAttributeValue.builder()
                    .dataType("String").stringValue("order-service").build(),
                "version", MessageAttributeValue.builder()
                    .dataType("Number").stringValue("1").build()
            ))
            .build());
        System.out.println("MessageId: " + response.messageId());
    }

    public void sendFifoMessage(String queueUrl, String body, String groupId, String orderId) {
        sqs.sendMessage(SendMessageRequest.builder()
            .queueUrl(queueUrl)
            .messageBody(body)
            .messageGroupId(groupId)              // Messages in same group are ordered
            .messageDeduplicationId(orderId)      // Prevents duplicates within 5 min window
            .build());
    }

    public void sendBatch(String queueUrl, List<String> bodies) {
        // Max 10 messages per batch, up to 256KB total
        List<SendMessageBatchRequestEntry> entries = new ArrayList<>();
        for (int i = 0; i < bodies.size(); i++) {
            entries.add(SendMessageBatchRequestEntry.builder()
                .id("msg-" + i)          // Local batch ID for error mapping
                .messageBody(bodies.get(i))
                .build());
        }

        SendMessageBatchResponse response = sqs.sendMessageBatch(
            SendMessageBatchRequest.builder()
                .queueUrl(queueUrl)
                .entries(entries)
                .build()
        );

        if (!response.failed().isEmpty()) {
            response.failed().forEach(f ->
                System.err.println("Failed: " + f.id() + " — " + f.message())
            );
        }
    }

    // ── Receive & Process Messages ────────────────────────────────────────────

    public void pollAndProcess(String queueUrl) {
        while (true) {
            ReceiveMessageResponse response = sqs.receiveMessage(
                ReceiveMessageRequest.builder()
                    .queueUrl(queueUrl)
                    .maxNumberOfMessages(10)         // Max 10 per poll
                    .waitTimeSeconds(20)             // Long polling (reduces API calls & cost!)
                    .visibilityTimeout(30)           // Hide from other consumers for 30s
                    .messageAttributeNames("All")
                    .attributeNames(QueueAttributeName.ALL)
                    .build()
            );

            for (Message message : response.messages()) {
                try {
                    processMessage(message);
                    // Delete ONLY after successful processing (at-least-once delivery)
                    sqs.deleteMessage(DeleteMessageRequest.builder()
                        .queueUrl(queueUrl)
                        .receiptHandle(message.receiptHandle())
                        .build());
                } catch (Exception e) {
                    // Don't delete — message returns to queue after visibility timeout
                    // Extends visibility if processing takes longer than expected
                    sqs.changeMessageVisibility(ChangeMessageVisibilityRequest.builder()
                        .queueUrl(queueUrl)
                        .receiptHandle(message.receiptHandle())
                        .visibilityTimeout(60)  // Extend by another 60s
                        .build());
                }
            }
        }
    }

    private void processMessage(Message message) {
        System.out.println("Processing: " + message.messageId() + " → " + message.body());
    }

    public void deleteBatch(String queueUrl, List<Message> messages) {
        List<DeleteMessageBatchRequestEntry> entries = messages.stream()
            .map(m -> DeleteMessageBatchRequestEntry.builder()
                .id(m.messageId())
                .receiptHandle(m.receiptHandle())
                .build())
            .collect(Collectors.toList());

        sqs.deleteMessageBatch(DeleteMessageBatchRequest.builder()
            .queueUrl(queueUrl)
            .entries(entries)
            .build());
    }

    /*
     * SQS Interview Key Points:
     * ─────────────────────────
     * Types:
     *   - Standard: at-least-once delivery, best-effort ordering, nearly unlimited TPS
     *   - FIFO: exactly-once processing, strict ordering, 3000 msg/s (with batching)
     *
     * Visibility timeout: period during which a received message is hidden.
     *   If not deleted within timeout, message reappears → another consumer processes it.
     *   Design handlers to be IDEMPOTENT!
     *
     * Long polling vs short polling:
     *   - Short: returns immediately even if empty (wasteful, costs more)
     *   - Long: waits up to 20s for a message (cheaper, fewer empty responses)
     *
     * Message size: up to 256KB. For larger: S3 + SQS Extended Client Library
     *
     * DLQ: receives messages that fail maxReceiveCount times
     *   - Monitor DLQ depth with CloudWatch alarm
     *   - Use SQS DLQ Redrive to replay messages back to source
     *
     * Fan-out pattern: SNS topic → multiple SQS queues (decouple producers/consumers)
     *
     * Lambda triggers: Lambda polls SQS; batch size configurable; partial batch failure support
     */
}


// ============================================================================
// SECTION 6: AMAZON SNS
// ============================================================================

class SnsExamples {

    private final SnsClient sns;

    SnsExamples() {
        this.sns = SnsClient.builder().region(Region.US_EAST_1).build();
    }

    // ── Topic Operations ──────────────────────────────────────────────────────

    public String createTopic(String topicName) {
        CreateTopicResponse response = sns.createTopic(CreateTopicRequest.builder()
            .name(topicName)
            .attributes(Map.of(
                "DisplayName",     "Order Notifications",
                "KmsMasterKeyId",  "alias/aws/sns"  // Encryption at rest
            ))
            .build());
        return response.topicArn();
    }

    public String createFifoTopic(String topicName) {
        CreateTopicResponse response = sns.createTopic(CreateTopicRequest.builder()
            .name(topicName + ".fifo")
            .attributes(Map.of(
                "FifoTopic",                    "true",
                "ContentBasedDeduplication",    "true"
            ))
            .build());
        return response.topicArn();
    }

    // ── Subscribe ─────────────────────────────────────────────────────────────

    public String subscribeEmail(String topicArn, String email) {
        SubscribeResponse response = sns.subscribe(SubscribeRequest.builder()
            .topicArn(topicArn)
            .protocol("email")
            .endpoint(email)
            .build());
        // Returns "PendingConfirmation" until user confirms email
        return response.subscriptionArn();
    }

    public String subscribeToSqsQueue(String topicArn, String sqsQueueArn) {
        SubscribeResponse response = sns.subscribe(SubscribeRequest.builder()
            .topicArn(topicArn)
            .protocol("sqs")
            .endpoint(sqsQueueArn)
            .attributes(Map.of("RawMessageDelivery", "true"))  // Skip SNS envelope wrapper
            .build());
        return response.subscriptionArn();
    }

    public String subscribeToLambda(String topicArn, String lambdaArn) {
        SubscribeResponse response = sns.subscribe(SubscribeRequest.builder()
            .topicArn(topicArn)
            .protocol("lambda")
            .endpoint(lambdaArn)
            .build());
        return response.subscriptionArn();
    }

    // Subscription with filter policy (only receive matching messages)
    public String subscribeWithFilter(String topicArn, String sqsArn, String filterJson) {
        // Filter example: {"eventType": ["ORDER_CREATED", "ORDER_SHIPPED"], "amount": [{"numeric": [">=", 100]}]}
        SubscribeResponse response = sns.subscribe(SubscribeRequest.builder()
            .topicArn(topicArn)
            .protocol("sqs")
            .endpoint(sqsArn)
            .attributes(Map.of("FilterPolicy", filterJson))
            .build());
        return response.subscriptionArn();
    }

    // ── Publish ───────────────────────────────────────────────────────────────

    public void publish(String topicArn, String subject, String message) {
        PublishResponse response = sns.publish(PublishRequest.builder()
            .topicArn(topicArn)
            .subject(subject)
            .message(message)
            .messageAttributes(Map.of(
                "eventType", MessageAttributeValue.builder()
                    .dataType("String").stringValue("ORDER_CREATED").build(),
                "amount", MessageAttributeValue.builder()
                    .dataType("Number").stringValue("250.00").build()
            ))
            .build());
        System.out.println("Published messageId: " + response.messageId());
    }

    // Message per protocol (different body for email vs SQS vs Lambda)
    public void publishWithStructure(String topicArn) {
        String message = """
            {
              "default": "Default message",
              "email": "Dear customer, your order has been shipped.",
              "sqs": "{\\"orderId\\": \\"123\\", \\"status\\": \\"SHIPPED\\"}",
              "lambda": "{\\"orderId\\": \\"123\\", \\"trigger\\": \\"fulfillment\\"}"
            }
            """;

        sns.publish(PublishRequest.builder()
            .topicArn(topicArn)
            .message(message)
            .messageStructure("json")  // Enables per-protocol message
            .build());
    }

    // Direct publish to phone (SMS)
    public void sendSms(String phoneNumber, String message) {
        sns.publish(PublishRequest.builder()
            .phoneNumber(phoneNumber)  // E.164 format: +14155552671
            .message(message)
            .messageAttributes(Map.of(
                "AWS.SNS.SMS.SMSType", MessageAttributeValue.builder()
                    .dataType("String").stringValue("Transactional").build()
            ))
            .build());
    }

    /*
     * SNS Interview Key Points:
     * ─────────────────────────
     * SNS = Push-based pub/sub; SQS = Pull-based queue
     *
     * Protocols: HTTP/HTTPS, Email, SQS, Lambda, SMS, Mobile Push (APNs, FCM)
     *
     * Fan-out: one SNS publish → N SQS queues (classic pattern)
     *   e.g., OrderCreated → [FulfillmentQueue, AnalyticsQueue, NotificationQueue]
     *
     * Filter policies: reduce processing by filtering at SNS level (not in Lambda)
     *   Reduces cost and latency
     *
     * Message retry: HTTPS endpoint → 3 retries with exponential backoff
     *
     * Dead-letter queues for SNS subscriptions: undeliverable messages → DLQ
     *
     * SNS + SQS vs EventBridge:
     *   - SNS/SQS: simple, high-throughput, lower cost
     *   - EventBridge: rich filtering, schema registry, 20+ AWS service sources,
     *     replay events, archiving — use for complex event routing
     */
}


// ============================================================================
// SECTION 7: AWS STEP FUNCTIONS
// ============================================================================

class StepFunctionsExamples {

    private final SfnClient sfn;

    StepFunctionsExamples() {
        this.sfn = SfnClient.builder().region(Region.US_EAST_1).build();
    }

    // ── State Machine Definition (ASL — Amazon States Language) ──────────────

    // Step Functions uses JSON-based ASL to define workflows.
    // Types: Standard (long-running, exactly-once, up to 1yr), Express (high-throughput, at-least-once, 5min)

    static final String ORDER_WORKFLOW_ASL = """
        {
          "Comment": "Order Processing Workflow",
          "StartAt": "ValidateOrder",
          "States": {
            "ValidateOrder": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:us-east-1:123456789:function:ValidateOrder",
              "Next": "CheckInventory",
              "Retry": [{
                "ErrorEquals": ["Lambda.ServiceException", "Lambda.TooManyRequestsException"],
                "IntervalSeconds": 2,
                "MaxAttempts": 3,
                "BackoffRate": 2
              }],
              "Catch": [{
                "ErrorEquals": ["ValidationError"],
                "Next": "OrderFailed",
                "ResultPath": "$.error"
              }]
            },
            "CheckInventory": {
              "Type": "Task",
              "Resource": "arn:aws:states:::dynamodb:getItem",
              "Parameters": {
                "TableName": "Inventory",
                "Key": { "productId": { "S.$": "$.productId" } }
              },
              "Next": "InventoryChoice"
            },
            "InventoryChoice": {
              "Type": "Choice",
              "Choices": [{
                "Variable": "$.Item.stock.N",
                "NumericGreaterThan": 0,
                "Next": "ProcessPayment"
              }],
              "Default": "OrderBackordered"
            },
            "ProcessPayment": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:us-east-1:123456789:function:ProcessPayment",
              "Next": "NotifyCustomer"
            },
            "NotifyCustomer": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sns:publish",
              "Parameters": {
                "TopicArn": "arn:aws:sns:us-east-1:123456789:OrderNotifications",
                "Message.$": "States.Format('Order {} confirmed!', $.orderId)"
              },
              "End": true
            },
            "OrderBackordered": {
              "Type": "Wait",
              "Seconds": 3600,
              "Next": "CheckInventory"
            },
            "OrderFailed": {
              "Type": "Fail",
              "Error": "OrderFailed",
              "Cause": "Validation failed"
            }
          }
        }
        """;

    // Parallel state — run branches concurrently
    static final String PARALLEL_STATE_EXAMPLE = """
        "ParallelProcessing": {
          "Type": "Parallel",
          "Branches": [
            {
              "StartAt": "SendEmail",
              "States": {
                "SendEmail": { "Type": "Task", "Resource": "...", "End": true }
              }
            },
            {
              "StartAt": "UpdateAnalytics",
              "States": {
                "UpdateAnalytics": { "Type": "Task", "Resource": "...", "End": true }
              }
            }
          ],
          "Next": "Done"
        }
        """;

    // Map state — dynamic parallelism over an array
    static final String MAP_STATE_EXAMPLE = """
        "ProcessItems": {
          "Type": "Map",
          "InputPath": "$.items",
          "ItemsPath": "$",
          "MaxConcurrency": 10,
          "Iterator": {
            "StartAt": "ProcessSingleItem",
            "States": {
              "ProcessSingleItem": { "Type": "Task", "Resource": "...", "End": true }
            }
          },
          "End": true
        }
        """;

    // ── Create & Execute State Machine ────────────────────────────────────────

    public String createStateMachine(String name, String roleArn) {
        CreateStateMachineResponse response = sfn.createStateMachine(
            CreateStateMachineRequest.builder()
                .name(name)
                .definition(ORDER_WORKFLOW_ASL)
                .roleArn(roleArn)
                .type(StateMachineType.STANDARD)
                .loggingConfiguration(LoggingConfiguration.builder()
                    .level(LogLevel.ERROR)
                    .includeExecutionData(true)
                    .build())
                .build()
        );
        return response.stateMachineArn();
    }

    public String startExecution(String stateMachineArn, String orderId, String inputJson) {
        StartExecutionResponse response = sfn.startExecution(
            StartExecutionRequest.builder()
                .stateMachineArn(stateMachineArn)
                .name("order-" + orderId + "-" + System.currentTimeMillis())
                .input(inputJson)
                .build()
        );
        return response.executionArn();
    }

    public void waitForCompletion(String executionArn) throws InterruptedException {
        while (true) {
            DescribeExecutionResponse desc = sfn.describeExecution(
                DescribeExecutionRequest.builder().executionArn(executionArn).build()
            );

            ExecutionStatus status = desc.status();
            System.out.println("Status: " + status);

            if (status == ExecutionStatus.RUNNING) {
                Thread.sleep(2000);
            } else {
                System.out.println("Output: " + desc.output());
                break;
            }
        }
    }

    // ── Activity Tasks (external workers polling for work) ───────────────────

    public void activityWorkerLoop(String activityArn) {
        while (true) {
            // Long poll — blocks up to 60s waiting for a task
            GetActivityTaskResponse task = sfn.getActivityTask(
                GetActivityTaskRequest.builder().activityArn(activityArn).build()
            );

            if (task.taskToken() == null) continue;  // No task available

            try {
                String result = doWork(task.input());

                sfn.sendTaskSuccess(SendTaskSuccessRequest.builder()
                    .taskToken(task.taskToken())
                    .output(result)
                    .build());
            } catch (Exception e) {
                sfn.sendTaskFailure(SendTaskFailureRequest.builder()
                    .taskToken(task.taskToken())
                    .error("WorkerError")
                    .cause(e.getMessage())
                    .build());
            }
        }
    }

    private String doWork(String input) { return """{"result": "processed"}"""; }

    // ── Callback Pattern (waitForTaskToken) ───────────────────────────────────
    // Pause workflow until external system calls back with token
    // Resource: "arn:aws:states:::sqs:sendMessage.waitForTaskToken"
    // Passes "$.taskToken" in message; external system calls SendTaskSuccess/Failure

    /*
     * Step Functions Interview Key Points:
     * ─────────────────────────────────────
     * State types: Task, Choice, Wait, Parallel, Map, Pass, Succeed, Fail
     *
     * Standard vs Express:
     *   Standard: exactly-once, up to 1yr, full audit history, $0.025/1K transitions
     *   Express: at-least-once, max 5min, CloudWatch logs, cheaper for high-frequency
     *
     * SDK integrations (optimistic vs pesimistic):
     *   - .sync (synchronous): step waits for async job to complete
     *   - .waitForTaskToken: human approval / external callback patterns
     *
     * Error handling: Retry (exponential backoff) + Catch (redirect on specific error)
     *
     * Common patterns:
     *   - Saga pattern: distributed transaction with compensating transactions
     *   - Human approval: send email → wait → approve/reject → continue
     *   - Scatter-gather: Map state → parallel processing → aggregate
     *   - Circuit breaker: Choice state checks failure count before proceeding
     */
}


// ============================================================================
// SECTION 8: AMAZON EC2 & VPC
// ============================================================================

class Ec2AndVpcExamples {

    private final Ec2Client ec2;

    Ec2AndVpcExamples() {
        this.ec2 = Ec2Client.builder().region(Region.US_EAST_1).build();
    }

    // ── VPC Creation ──────────────────────────────────────────────────────────

    public String createVpc(String cidrBlock) {
        CreateVpcResponse vpc = ec2.createVpc(CreateVpcRequest.builder()
            .cidrBlock(cidrBlock)  // e.g., "10.0.0.0/16"
            .build());

        String vpcId = vpc.vpc().vpcId();

        // Enable DNS hostnames
        ec2.modifyVpcAttribute(ModifyVpcAttributeRequest.builder()
            .vpcId(vpcId)
            .enableDnsHostnames(AttributeBooleanValue.builder().value(true).build())
            .build());

        tagResource(vpcId, "Name", "prod-vpc");
        return vpcId;
    }

    public String createSubnet(String vpcId, String cidr, String az, boolean isPublic) {
        CreateSubnetResponse response = ec2.createSubnet(CreateSubnetRequest.builder()
            .vpcId(vpcId)
            .cidrBlock(cidr)           // e.g., "10.0.1.0/24"
            .availabilityZone(az)      // e.g., "us-east-1a"
            .build());

        String subnetId = response.subnet().subnetId();

        if (isPublic) {
            // Auto-assign public IPs in public subnet
            ec2.modifySubnetAttribute(ModifySubnetAttributeRequest.builder()
                .subnetId(subnetId)
                .mapPublicIpOnLaunch(AttributeBooleanValue.builder().value(true).build())
                .build());
        }

        tagResource(subnetId, "Name", isPublic ? "public-subnet" : "private-subnet");
        return subnetId;
    }

    public String createInternetGateway(String vpcId) {
        CreateInternetGatewayResponse igw = ec2.createInternetGateway(
            CreateInternetGatewayRequest.builder().build()
        );
        String igwId = igw.internetGateway().internetGatewayId();
        ec2.attachInternetGateway(AttachInternetGatewayRequest.builder()
            .internetGatewayId(igwId)
            .vpcId(vpcId)
            .build());
        return igwId;
    }

    // ── Security Groups ───────────────────────────────────────────────────────

    public String createSecurityGroup(String vpcId, String name, String description) {
        CreateSecurityGroupResponse response = ec2.createSecurityGroup(
            CreateSecurityGroupRequest.builder()
                .vpcId(vpcId)
                .groupName(name)
                .description(description)
                .build()
        );
        String sgId = response.groupId();

        // Allow inbound HTTPS from anywhere
        ec2.authorizeSecurityGroupIngress(AuthorizeSecurityGroupIngressRequest.builder()
            .groupId(sgId)
            .ipPermissions(
                IpPermission.builder()
                    .ipProtocol("tcp")
                    .fromPort(443)
                    .toPort(443)
                    .ipRanges(IpRange.builder().cidrIp("0.0.0.0/0").description("HTTPS").build())
                    .build(),
                // Allow SSH from specific IP only (not 0.0.0.0/0!)
                IpPermission.builder()
                    .ipProtocol("tcp")
                    .fromPort(22)
                    .toPort(22)
                    .ipRanges(IpRange.builder().cidrIp("203.0.113.0/32").description("Admin").build())
                    .build()
            )
            .build());

        return sgId;
    }

    // ── Launch EC2 Instance ───────────────────────────────────────────────────

    public String launchInstance(String amiId, String instanceType,
                                  String subnetId, String sgId, String keyPair) {
        String userData = """
            #!/bin/bash
            yum update -y
            yum install -y java-21-amazon-corretto
            java -jar /opt/app/service.jar
            """;

        RunInstancesResponse response = ec2.runInstances(RunInstancesRequest.builder()
            .imageId(amiId)
            .instanceType(InstanceType.fromValue(instanceType))  // e.g., "t3.medium"
            .minCount(1)
            .maxCount(1)
            .subnetId(subnetId)
            .securityGroupIds(sgId)
            .keyName(keyPair)
            .iamInstanceProfile(IamInstanceProfileSpecification.builder()
                .name("EC2-SSM-Role")  // Instance profile for SSM access (no SSH key needed!)
                .build())
            .userData(Base64.getEncoder().encodeToString(userData.getBytes()))
            .blockDeviceMappings(BlockDeviceMapping.builder()
                .deviceName("/dev/xvda")
                .ebs(EbsBlockDevice.builder()
                    .volumeSize(30)
                    .volumeType(VolumeType.GP3)
                    .encrypted(true)
                    .deleteOnTermination(true)
                    .build())
                .build())
            .tagSpecifications(TagSpecification.builder()
                .resourceType(ResourceType.INSTANCE)
                .tags(Tag.builder().key("Name").value("app-server").build(),
                      Tag.builder().key("Environment").value("prod").build())
                .build())
            .build());

        String instanceId = response.instances().get(0).instanceId();

        // Wait until running
        ec2.waiter().waitUntilInstanceRunning(
            DescribeInstancesRequest.builder().instanceIds(instanceId).build()
        );

        return instanceId;
    }

    // ── Describe & Filter Instances ───────────────────────────────────────────

    public List<Instance> findInstancesByTag(String tagKey, String tagValue) {
        DescribeInstancesResponse response = ec2.describeInstances(
            DescribeInstancesRequest.builder()
                .filters(
                    Filter.builder()
                        .name("tag:" + tagKey)
                        .values(tagValue)
                        .build(),
                    Filter.builder()
                        .name("instance-state-name")
                        .values("running")
                        .build()
                )
                .build()
        );

        return response.reservations().stream()
            .flatMap(r -> r.instances().stream())
            .collect(Collectors.toList());
    }

    // ── Stop / Start / Terminate ──────────────────────────────────────────────

    public void stopInstances(List<String> instanceIds) {
        ec2.stopInstances(StopInstancesRequest.builder().instanceIds(instanceIds).build());
        ec2.waiter().waitUntilInstanceStopped(
            DescribeInstancesRequest.builder().instanceIds(instanceIds).build()
        );
    }

    public void terminateInstances(List<String> instanceIds) {
        ec2.terminateInstances(
            TerminateInstancesRequest.builder().instanceIds(instanceIds).build()
        );
    }

    private void tagResource(String resourceId, String key, String value) {
        ec2.createTags(CreateTagsRequest.builder()
            .resources(resourceId)
            .tags(Tag.builder().key(key).value(value).build())
            .build());
    }

    /*
     * EC2 & VPC Interview Key Points:
     * ────────────────────────────────
     * Instance families:
     *   t-series: burstable (T2/T3), CPU credits
     *   m-series: general purpose
     *   c-series: compute-optimized
     *   r-series: memory-optimized
     *   p/g-series: GPU
     *   i-series: storage-optimized NVMe SSD
     *
     * Pricing models:
     *   On-Demand: pay per second, no commitment
     *   Reserved: 1-3yr commitment, up to 75% discount (Standard vs Convertible)
     *   Spot: up to 90% discount, can be interrupted with 2min notice
     *   Savings Plans: flexible commitment (Compute vs EC2 Instance)
     *
     * Placement groups:
     *   Cluster: low latency, same AZ, same rack — HPC
     *   Spread: max 7 instances per AZ across different hardware — critical HA
     *   Partition: large distributed systems (Kafka, HDFS) — racks isolated
     *
     * VPC key concepts:
     *   - CIDR: VPC range (/16 max), subnets (/28 min, AWS reserves 5 IPs per subnet)
     *   - Public subnet: has route to Internet Gateway
     *   - Private subnet: no direct internet; uses NAT Gateway for outbound
     *   - NAT Gateway: managed, AZ-scoped, put one per AZ for HA; billed per GB
     *   - Security Groups: stateful, instance level, whitelist only
     *   - NACLs: stateless, subnet level, allow + deny rules, evaluated in order
     *   - VPC Peering: private routing between VPCs (non-transitive)
     *   - Transit Gateway: hub for many VPCs + on-prem (replaces peering mesh)
     *   - PrivateLink / VPC Endpoints: access AWS services without internet
     *     - Gateway endpoint: S3, DynamoDB (free)
     *     - Interface endpoint: other services (per-hour cost)
     */
}


// ============================================================================
// SECTION 9: ELASTIC LOAD BALANCING (ALB / NLB)
// ============================================================================

class ElbExamples {

    private final ElasticLoadBalancingV2Client elb;

    ElbExamples() {
        this.elb = ElasticLoadBalancingV2Client.builder().region(Region.US_EAST_1).build();
    }

    // ── Create ALB ────────────────────────────────────────────────────────────

    public String createAlb(String name, List<String> subnetIds, String sgId) {
        CreateLoadBalancerResponse response = elb.createLoadBalancer(
            CreateLoadBalancerRequest.builder()
                .name(name)
                .type(LoadBalancerTypeEnum.APPLICATION)  // ALB
                .scheme(LoadBalancerSchemeEnum.INTERNET_FACING)
                .subnets(subnetIds)                      // At least 2 AZs
                .securityGroups(sgId)
                .ipAddressType(IpAddressType.IPV4)
                .build()
        );

        String albArn = response.loadBalancers().get(0).loadBalancerArn();

        elb.waiter().waitUntilLoadBalancerAvailable(
            DescribeLoadBalancersRequest.builder().loadBalancerArns(albArn).build()
        );
        return albArn;
    }

    // ── Create Target Group ───────────────────────────────────────────────────

    public String createTargetGroup(String name, String vpcId) {
        CreateTargetGroupResponse response = elb.createTargetGroup(
            CreateTargetGroupRequest.builder()
                .name(name)
                .protocol(ProtocolEnum.HTTPS)
                .port(8443)
                .targetType(TargetTypeEnum.INSTANCE)     // or IP (for ECS/Lambda), LAMBDA
                .vpcId(vpcId)
                .healthCheckEnabled(true)
                .healthCheckPath("/actuator/health")
                .healthCheckIntervalSeconds(30)
                .healthCheckTimeoutSeconds(5)
                .healthyThresholdCount(2)
                .unhealthyThresholdCount(3)
                .matcher(Matcher.builder().httpCode("200").build())
                .build()
        );
        return response.targetGroups().get(0).targetGroupArn();
    }

    // ── Register Targets ──────────────────────────────────────────────────────

    public void registerInstances(String targetGroupArn, List<String> instanceIds) {
        List<TargetDescription> targets = instanceIds.stream()
            .map(id -> TargetDescription.builder().id(id).port(8443).build())
            .collect(Collectors.toList());

        elb.registerTargets(RegisterTargetsRequest.builder()
            .targetGroupArn(targetGroupArn)
            .targets(targets)
            .build());
    }

    // ── Create Listener with Routing Rules ───────────────────────────────────

    public void createListenerWithRules(String albArn, String defaultTgArn,
                                         String apiTgArn, String adminTgArn,
                                         String certArn) {
        // Create HTTPS listener
        CreateListenerResponse listenerResponse = elb.createListener(
            CreateListenerRequest.builder()
                .loadBalancerArn(albArn)
                .protocol(ProtocolEnum.HTTPS)
                .port(443)
                .sslPolicy("ELBSecurityPolicy-TLS13-1-2-2021-06")
                .certificates(Certificate.builder().certificateArn(certArn).build())
                .defaultActions(Action.builder()
                    .type(ActionTypeEnum.FORWARD)
                    .targetGroupArn(defaultTgArn)
                    .build())
                .build()
        );

        String listenerArn = listenerResponse.listeners().get(0).listenerArn();

        // Add path-based routing rule: /api/* → api target group
        elb.createRule(CreateRuleRequest.builder()
            .listenerArn(listenerArn)
            .priority(10)
            .conditions(RuleCondition.builder()
                .field("path-pattern")
                .values("/api/*")
                .build())
            .actions(Action.builder()
                .type(ActionTypeEnum.FORWARD)
                .targetGroupArn(apiTgArn)
                .build())
            .build());

        // Host-based routing: admin.example.com → admin target group
        elb.createRule(CreateRuleRequest.builder()
            .listenerArn(listenerArn)
            .priority(20)
            .conditions(RuleCondition.builder()
                .field("host-header")
                .values("admin.example.com")
                .build())
            .actions(Action.builder()
                .type(ActionTypeEnum.FORWARD)
                .targetGroupArn(adminTgArn)
                .build())
            .build());

        // HTTP → HTTPS redirect listener
        elb.createListener(CreateListenerRequest.builder()
            .loadBalancerArn(albArn)
            .protocol(ProtocolEnum.HTTP)
            .port(80)
            .defaultActions(Action.builder()
                .type(ActionTypeEnum.REDIRECT)
                .redirectConfig(RedirectActionConfig.builder()
                    .protocol("HTTPS")
                    .port("443")
                    .statusCode(RedirectActionStatusCodeEnum.HTTP_301)
                    .build())
                .build())
            .build());
    }

    /*
     * ELB Interview Key Points:
     * ──────────────────────────
     * ALB (Application Load Balancer):
     *   - Layer 7 (HTTP/HTTPS, WebSocket, gRPC)
     *   - Path-based, host-based, query string, header routing
     *   - Native Lambda, ECS, EKS targets
     *   - WAF integration
     *   - Sticky sessions via cookies
     *
     * NLB (Network Load Balancer):
     *   - Layer 4 (TCP, UDP, TLS)
     *   - Ultra-low latency, millions of req/s
     *   - Static IP per AZ; Elastic IP support
     *   - Use for: gaming, IoT, VoIP, PrivateLink
     *
     * CLB (Classic): legacy, don't use
     *
     * Gateway Load Balancer (GWLB):
     *   - Layer 3, transparent network appliances (firewalls, IDS)
     *
     * Cross-zone load balancing:
     *   - ALB: enabled by default (no cost)
     *   - NLB/GWLB: disabled by default (cross-AZ data transfer cost)
     *
     * Connection draining / deregistration delay: allows in-flight requests to complete
     *   (default 300s) before removing target from rotation.
     *
     * Health checks: instance must pass N consecutive checks to be healthy.
     *   Unhealthy targets are removed from rotation automatically.
     *
     * Access logs: stored to S3; useful for debugging and security audits.
     */
}


// ============================================================================
// SECTION 10: AMAZON API GATEWAY
// ============================================================================

class ApiGatewayExamples {

    /*
     * API Gateway types:
     *  - REST API:     full-featured, request/response transformation, stages, caching
     *  - HTTP API:     simpler, cheaper, faster (lower latency), JWT authorizers, no transformation
     *  - WebSocket API: persistent connections, @connect/@disconnect/$default routes
     *
     * Integration types:
     *  - Lambda proxy:     request/response forwarded as-is to Lambda
     *  - Lambda custom:    request/response transformed via mapping templates (VTL)
     *  - HTTP proxy:       forward to HTTP backend
     *  - AWS service:      directly call AWS API (S3, DynamoDB, SQS, etc.)
     *  - Mock:             return static response (useful for CORS, testing)
     *
     * Deployment concepts:
     *  - Stage: named deployment (dev, staging, prod) with per-stage variables
     *  - Canary deployment: route X% of traffic to new version
     *  - Usage plan + API key: throttling and quota per client
     *
     * Security:
     *  - Cognito authorizer: validate JWT from user pool
     *  - Lambda authorizer: custom token/request-based auth (e.g., OAuth, SAML)
     *  - IAM auth: Signature Version 4 (service-to-service)
     *  - Resource policy: IP allow/deny, cross-account access
     *
     * Throttling:
     *  - Account limit: 10,000 req/s (burst 5,000)
     *  - Per-stage, per-method limits configurable
     *  - Returns 429 Too Many Requests when exceeded
     *
     * Caching:
     *  - Per-stage, 0.5GB to 237GB
     *  - Cache key: method, URL, headers, query params (configurable)
     *  - TTL 5 min default (0 = no cache, max 3600s)
     *
     * Request/Response flow (REST API):
     *  Client → Method Request (validate) → Integration Request (transform)
     *         → Backend → Integration Response (transform) → Method Response → Client
     *
     * Errors:
     *  - 4xx: client errors (400 bad request, 401 unauthorized, 403 forbidden, 429 throttled)
     *  - 5xx: server errors (500 integration error, 502 bad gateway, 503 unavailable, 504 timeout)
     *
     * VPC Link: route API Gateway to private resources in VPC (NLB behind the scenes)
     *
     * Lambda proxy integration request event structure:
     *  { httpMethod, path, pathParameters, queryStringParameters,
     *    headers, multiValueHeaders, body, isBase64Encoded,
     *    requestContext: { authorizer: { claims: {...} } } }
     *
     * CORS: API Gateway can handle CORS preflight (OPTIONS) automatically.
     *   Set Access-Control-Allow-Origin in response headers.
     *   For Lambda proxy: add headers in Lambda response.
     */
}


// ============================================================================
// SECTION 11: AMAZON COGNITO
// ============================================================================

class CognitoExamples {

    private final CognitoIdentityProviderClient cognito;

    CognitoExamples() {
        this.cognito = CognitoIdentityProviderClient.builder()
            .region(Region.US_EAST_1).build();
    }

    // ── Create User Pool ──────────────────────────────────────────────────────

    public String createUserPool(String poolName) {
        CreateUserPoolResponse response = cognito.createUserPool(
            CreateUserPoolRequest.builder()
                .poolName(poolName)
                .policies(UserPoolPolicyType.builder()
                    .passwordPolicy(PasswordPolicyType.builder()
                        .minimumLength(12)
                        .requireUppercase(true)
                        .requireLowercase(true)
                        .requireNumbers(true)
                        .requireSymbols(true)
                        .build())
                    .build())
                .mfaConfiguration(UserPoolMfaType.OPTIONAL)
                .autoVerifiedAttributes(VerifiedAttributeType.EMAIL)
                .usernameAttributes(UsernameAttributeType.EMAIL)
                .schema(
                    SchemaAttributeType.builder()
                        .name("email")
                        .required(true)
                        .attributeDataType(AttributeDataType.STRING)
                        .build(),
                    SchemaAttributeType.builder()
                        .name("custom:role")
                        .mutable(true)
                        .attributeDataType(AttributeDataType.STRING)
                        .build()
                )
                .emailConfiguration(EmailConfigurationType.builder()
                    .emailSendingAccount(EmailSendingAccountType.COGNITO_DEFAULT)
                    .build())
                .accountRecoverySetting(AccountRecoverySettingType.builder()
                    .recoveryMechanisms(RecoveryOptionType.builder()
                        .name(RecoveryOptionNameType.VERIFIED_EMAIL_ADDRESS)
                        .priority(1)
                        .build())
                    .build())
                .build()
        );

        return response.userPool().id();
    }

    // ── Create App Client ─────────────────────────────────────────────────────

    public String createAppClient(String userPoolId, String clientName) {
        CreateUserPoolClientResponse response = cognito.createUserPoolClient(
            CreateUserPoolClientRequest.builder()
                .userPoolId(userPoolId)
                .clientName(clientName)
                .generateSecret(false)             // False for public clients (SPA, mobile)
                .explicitAuthFlows(
                    ExplicitAuthFlowsType.ALLOW_USER_PASSWORD_AUTH,
                    ExplicitAuthFlowsType.ALLOW_REFRESH_TOKEN_AUTH,
                    ExplicitAuthFlowsType.ALLOW_USER_SRP_AUTH  // SRP = Secure Remote Password (no plain password over wire)
                )
                .allowedOAuthFlows(OAuthFlowType.CODE)  // Authorization code flow
                .allowedOAuthScopes("openid", "email", "profile")
                .callbackUrls("https://yourapp.com/callback")
                .logoutUrls("https://yourapp.com/logout")
                .allowedOAuthFlowsUserPoolClient(true)
                .tokenValidityUnits(TokenValidityUnitsType.builder()
                    .accessToken(TimeUnitsType.HOURS).idToken(TimeUnitsType.HOURS)
                    .refreshToken(TimeUnitsType.DAYS).build())
                .accessTokenValidity(1)
                .idTokenValidity(1)
                .refreshTokenValidity(30)
                .build()
        );
        return response.userPoolClient().clientId();
    }

    // ── User Management ───────────────────────────────────────────────────────

    public void adminCreateUser(String userPoolId, String email) {
        cognito.adminCreateUser(AdminCreateUserRequest.builder()
            .userPoolId(userPoolId)
            .username(email)
            .temporaryPassword("TempPass123!")
            .userAttributes(
                AttributeType.builder().name("email").value(email).build(),
                AttributeType.builder().name("email_verified").value("true").build(),
                AttributeType.builder().name("custom:role").value("USER").build()
            )
            .desiredDeliveryMediums(DeliveryMediumType.EMAIL)
            .build());
    }

    // Sign-up flow (user registers themselves)
    public void signUp(String clientId, String email, String password) {
        cognito.signUp(SignUpRequest.builder()
            .clientId(clientId)
            .username(email)
            .password(password)
            .userAttributes(
                AttributeType.builder().name("email").value(email).build()
            )
            .build());
        // User receives confirmation code via email
    }

    public void confirmSignUp(String clientId, String email, String confirmationCode) {
        cognito.confirmSignUp(ConfirmSignUpRequest.builder()
            .clientId(clientId)
            .username(email)
            .confirmationCode(confirmationCode)
            .build());
    }

    // Admin authentication (server-side, with secret)
    public AuthenticationResultType adminLogin(String userPoolId, String clientId,
                                                String email, String password) {
        AdminInitiateAuthResponse response = cognito.adminInitiateAuth(
            AdminInitiateAuthRequest.builder()
                .userPoolId(userPoolId)
                .clientId(clientId)
                .authFlow(AuthFlowType.ADMIN_USER_PASSWORD_AUTH)
                .authParameters(Map.of(
                    "USERNAME", email,
                    "PASSWORD", password
                ))
                .build()
        );
        // Returns: AccessToken (1h), IdToken (1h), RefreshToken (30d)
        return response.authenticationResult();
    }

    // Refresh tokens
    public AuthenticationResultType refreshSession(String clientId, String refreshToken) {
        InitiateAuthResponse response = cognito.initiateAuth(
            InitiateAuthRequest.builder()
                .clientId(clientId)
                .authFlow(AuthFlowType.REFRESH_TOKEN_AUTH)
                .authParameters(Map.of("REFRESH_TOKEN", refreshToken))
                .build()
        );
        return response.authenticationResult();
    }

    // Add user to group
    public void addUserToGroup(String userPoolId, String username, String groupName) {
        cognito.adminAddUserToGroup(AdminAddUserToGroupRequest.builder()
            .userPoolId(userPoolId)
            .username(username)
            .groupName(groupName)
            .build());
    }

    /*
     * Cognito Interview Key Points:
     * ──────────────────────────────
     * User Pool vs Identity Pool:
     *   User Pool: authentication — manages users, issues JWTs (access, id, refresh tokens)
     *   Identity Pool: authorization — exchanges JWT/social token for temporary AWS credentials
     *     (assumes IAM role) → access S3, DynamoDB directly from client
     *
     * Token types:
     *   - ID Token: user identity claims (email, custom attributes) — for API backend
     *   - Access Token: scopes, groups — for API Gateway authorizer
     *   - Refresh Token: long-lived, used to get new tokens
     *
     * JWT verification: validate signature against JWKS endpoint
     *   https://cognito-idp.{region}.amazonaws.com/{userPoolId}/.well-known/jwks.json
     *
     * Triggers (Lambda pre/post hooks):
     *   Pre sign-up, Post confirmation, Pre/Post auth, Pre token generation,
     *   Custom message, User migration, Define/Create/Verify auth challenge
     *
     * Hosted UI: Cognito-provided login/signup pages (customizable)
     *
     * Federated identity: Google, Facebook, Apple, SAML, OIDC providers
     *
     * Advanced security: adaptive authentication, compromised credential detection
     *
     * Groups + IAM roles: map Cognito groups to IAM roles for fine-grained access
     */
}


// ============================================================================
// SECTION 12: AMAZON ROUTE 53
// ============================================================================

class Route53Examples {

    private final Route53Client route53;

    Route53Examples() {
        this.route53 = Route53Client.builder().region(Region.AWS_GLOBAL).build();
        // Route 53 is global — always use AWS_GLOBAL
    }

    // ── Create Hosted Zone ────────────────────────────────────────────────────

    public String createHostedZone(String domainName, boolean isPrivate, String vpcId) {
        CreateHostedZoneRequest.Builder builder = CreateHostedZoneRequest.builder()
            .name(domainName)
            .callerReference(UUID.randomUUID().toString())  // Idempotency token
            .hostedZoneConfig(HostedZoneConfig.builder()
                .comment("Managed by Java SDK")
                .privateZone(isPrivate)
                .build());

        if (isPrivate) {
            builder.vpc(VPC.builder()
                .vpcId(vpcId)
                .vpcRegion(VPCRegion.US_EAST_1)
                .build());
        }

        return route53.createHostedZone(builder.build()).hostedZone().id();
    }

    // ── Create DNS Records ────────────────────────────────────────────────────

    public void createARecord(String hostedZoneId, String name, String ipAddress) {
        changeRecords(hostedZoneId, ChangeAction.CREATE,
            RRType.A, name, 300,
            ResourceRecord.builder().value(ipAddress).build());
    }

    public void createCnameRecord(String hostedZoneId, String name, String target) {
        changeRecords(hostedZoneId, ChangeAction.CREATE,
            RRType.CNAME, name, 300,
            ResourceRecord.builder().value(target).build());
    }

    public void createAliasRecord(String hostedZoneId, String name,
                                   String albDnsName, String albHostedZoneId) {
        // Alias record: points to AWS resource (ALB, CloudFront, S3) — free queries, auto-updates IP
        route53.changeResourceRecordSets(ChangeResourceRecordSetsRequest.builder()
            .hostedZoneId(hostedZoneId)
            .changeBatch(ChangeBatch.builder()
                .changes(Change.builder()
                    .action(ChangeAction.CREATE)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name)
                        .type(RRType.A)
                        .aliasTarget(AliasTarget.builder()
                            .dnsName(albDnsName)
                            .hostedZoneId(albHostedZoneId)
                            .evaluateTargetHealth(true)  // Return SERVFAIL if ALB unhealthy
                            .build())
                        .build())
                    .build())
                .build())
            .build());
    }

    // ── Weighted Routing (A/B, canary, blue-green) ────────────────────────────

    public void createWeightedRecords(String hostedZoneId, String name,
                                       String v1Ip, String v2Ip) {
        // Route 80% to v1, 20% to v2
        ChangeBatch batch = ChangeBatch.builder()
            .changes(
                Change.builder().action(ChangeAction.CREATE)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name).type(RRType.A)
                        .setIdentifier("v1")    // Unique ID within the record set
                        .weight(80L)
                        .ttl(60L)
                        .resourceRecords(ResourceRecord.builder().value(v1Ip).build())
                        .build())
                    .build(),
                Change.builder().action(ChangeAction.CREATE)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name).type(RRType.A)
                        .setIdentifier("v2")
                        .weight(20L)
                        .ttl(60L)
                        .resourceRecords(ResourceRecord.builder().value(v2Ip).build())
                        .build())
                    .build()
            )
            .build();

        route53.changeResourceRecordSets(ChangeResourceRecordSetsRequest.builder()
            .hostedZoneId(hostedZoneId)
            .changeBatch(batch)
            .build());
    }

    // ── Failover Routing ──────────────────────────────────────────────────────

    public void createFailoverRecords(String hostedZoneId, String name,
                                       String primaryIp, String secondaryIp,
                                       String healthCheckId) {
        ChangeBatch batch = ChangeBatch.builder()
            .changes(
                Change.builder().action(ChangeAction.CREATE)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name).type(RRType.A)
                        .setIdentifier("primary")
                        .failover(ResourceRecordSetFailover.PRIMARY)
                        .healthCheckId(healthCheckId)  // Only route here when healthy
                        .ttl(60L)
                        .resourceRecords(ResourceRecord.builder().value(primaryIp).build())
                        .build())
                    .build(),
                Change.builder().action(ChangeAction.CREATE)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name).type(RRType.A)
                        .setIdentifier("secondary")
                        .failover(ResourceRecordSetFailover.SECONDARY)
                        .ttl(60L)
                        .resourceRecords(ResourceRecord.builder().value(secondaryIp).build())
                        .build())
                    .build()
            )
            .build();

        route53.changeResourceRecordSets(ChangeResourceRecordSetsRequest.builder()
            .hostedZoneId(hostedZoneId).changeBatch(batch).build());
    }

    // ── Health Checks ─────────────────────────────────────────────────────────

    public String createHealthCheck(String endpoint, int port, String path) {
        CreateHealthCheckResponse response = route53.createHealthCheck(
            CreateHealthCheckRequest.builder()
                .callerReference(UUID.randomUUID().toString())
                .healthCheckConfig(HealthCheckConfig.builder()
                    .type(HealthCheckType.HTTPS)
                    .fullyQualifiedDomainName(endpoint)
                    .port(port)
                    .resourcePath(path)
                    .requestInterval(30)         // 30 seconds between checks
                    .failureThreshold(3)         // 3 consecutive failures = unhealthy
                    .enableSNI(true)
                    .build())
                .build()
        );
        return response.healthCheck().id();
    }

    private void changeRecords(String hostedZoneId, ChangeAction action,
                                RRType type, String name, long ttl,
                                ResourceRecord... records) {
        route53.changeResourceRecordSets(ChangeResourceRecordSetsRequest.builder()
            .hostedZoneId(hostedZoneId)
            .changeBatch(ChangeBatch.builder()
                .changes(Change.builder()
                    .action(action)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name(name).type(type).ttl(ttl)
                        .resourceRecords(records)
                        .build())
                    .build())
                .build())
            .build());
    }

    /*
     * Route 53 Interview Key Points:
     * ────────────────────────────────
     * Routing policies:
     *   Simple:       single resource, no health check
     *   Weighted:     A/B testing, blue-green, gradual migration
     *   Latency:      route to lowest-latency region
     *   Failover:     active-passive; primary fails → secondary
     *   Geolocation:  by continent, country, state (US)
     *   Geoproximity: shift traffic based on geographic bias (requires Traffic Flow)
     *   Multi-value:  up to 8 healthy records per response (not load balancer!)
     *   IP-based:     route by CIDR block (new, for custom ISP routing)
     *
     * Alias vs CNAME:
     *   Alias: free, works at zone apex (example.com), auto-resolves AWS resource IP
     *   CNAME: paid query, can't be at apex, points to any domain
     *
     * TTL: lower TTL before migrations (e.g., 60s during cutover)
     *
     * Health checks: monitor endpoint, other health checks (calculated), CloudWatch alarms
     *
     * Private hosted zones: DNS resolution within VPC; requires enableDnsHostnames + enableDnsSupport
     *
     * Route 53 Resolver: hybrid cloud DNS
     *   - Inbound endpoint: on-premises → VPC DNS resolution
     *   - Outbound endpoint: VPC → on-premises DNS resolution
     *   - Rules: forward specific domains to custom resolvers
     *
     * DNSSEC: Route 53 supports DNSSEC signing for public hosted zones
     *
     * Domain registration: Route 53 is also a domain registrar
     */
}


// ============================================================================
// SECTION 13: AWS SECRETS MANAGER
// ============================================================================

class SecretsManagerExamples {

    private final SecretsManagerClient secretsClient;

    SecretsManagerExamples() {
        this.secretsClient = SecretsManagerClient.builder()
            .region(Region.US_EAST_1).build();
    }

    public String getSecretValue(String secretName) {
        GetSecretValueResponse response = secretsClient.getSecretValue(
            GetSecretValueRequest.builder()
                .secretId(secretName)
                .build()
        );
        return response.secretString();
    }

    public void createSecret(String name, String value) {
        secretsClient.createSecret(CreateSecretRequest.builder()
            .name(name)
            .secretString(value)
            .description("Database credentials")
            .build());
    }

    public void rotateSecret(String secretArn, String lambdaArn) {
        secretsClient.rotateSecret(RotateSecretRequest.builder()
            .secretId(secretArn)
            .rotationLambdaARN(lambdaArn)
            .rotationRules(RotationRulesType.builder()
                .automaticallyAfterDays(30L)
                .build())
            .build());
    }

    /*
     * Secrets Manager vs SSM Parameter Store:
     *   Secrets Manager: auto-rotation, cross-account, $0.40/secret/month
     *   SSM Parameter Store: no auto-rotation, simpler, free (Standard tier), or $0.05 (Advanced)
     *   Use Secrets Manager for DB passwords, API keys requiring rotation
     *   Use Parameter Store for config values, feature flags, non-sensitive config
     */
}


// ============================================================================
// SECTION 14: ARCHITECTURE PATTERNS — AWS NATIVE
// ============================================================================

class ArchitecturePatterns {

    /*
     * ── Serverless Event-Driven Architecture ──────────────────────────────────
     *
     * API Gateway → Lambda → DynamoDB
     *             ↓ async
     *           SQS Queue → Lambda worker → [S3, SNS, RDS, external API]
     *
     * ── Fan-Out Pattern ───────────────────────────────────────────────────────
     *
     * Publisher → SNS Topic → SQS Queue 1 → Lambda (fulfillment)
     *                       → SQS Queue 2 → Lambda (analytics)
     *                       → SQS Queue 3 → Lambda (notifications)
     *                       → Lambda (real-time)
     *
     * ── CQRS + Event Sourcing on AWS ──────────────────────────────────────────
     *
     * Command API → Lambda → DynamoDB (write store)
     *                      → DynamoDB Stream → Lambda → Elasticsearch (read store)
     *                                                 → Lambda → Read DynamoDB projection
     *
     * ── Saga Pattern (distributed transactions) ───────────────────────────────
     *
     * Step Functions orchestrates:
     *   PlaceOrder → ReserveInventory → ChargePayment → ShipOrder
     *                     ↓ fail                            ↓ fail
     *             ReleaseInventory             RefundPayment → ReleaseInventory
     *
     * ── Blue-Green Deployment on AWS ──────────────────────────────────────────
     *
     * Route 53 weighted: 100% Blue
     * Deploy Green (new version)
     * Shift: 90/10 → 70/30 → 0/100
     * Rollback: shift back to 100% Blue instantly
     *
     * ── Strangler Fig (Monolith → Microservices) ──────────────────────────────
     *
     * API Gateway routes /v2/orders → Lambda microservice
     *                    /v1/*      → monolith (ALB + EC2)
     * Gradually migrate routes
     *
     * ── Circuit Breaker on Lambda ─────────────────────────────────────────────
     *
     * Lambda → [check SSM param / DynamoDB flag] → if open, return cached/degraded
     *        → if closed, call downstream, track failures
     *        → if failures >= threshold, write "open" to DynamoDB
     *
     * ── Idempotency ───────────────────────────────────────────────────────────
     *
     * Strategy 1: DynamoDB conditional put (attribute_not_exists)
     * Strategy 2: Lambda Powertools idempotency decorator (DynamoDB-backed)
     * Strategy 3: SQS FIFO + deduplication ID
     * Strategy 4: Idempotency key in API header → cache in ElastiCache
     */

    // ── Lambda Retry / DLQ Handling ───────────────────────────────────────────

    // Pattern: SQS → Lambda with partial batch failure reporting
    // Lambda processes batch; for partial failures, only failed items requeue

    static final String PARTIAL_BATCH_RESPONSE_PATTERN =
        """
        // In Lambda handler with SQS trigger:
        // Enable: FunctionResponseTypes: ["ReportBatchItemFailures"]
        //
        // Return: {
        //   "batchItemFailures": [
        //     { "itemIdentifier": "messageId_of_failed_message" }
        //   ]
        // }
        //
        // Successfully processed messages are deleted; only failures requeue.
        """;
}


// ============================================================================
// SECTION 15: AWS INTERVIEW Q&A QUICK REFERENCE
// ============================================================================

/*
 * ============================================================================
 * TOP AWS INTERVIEW QUESTIONS FOR JAVA/SDE 8+ YEARS
 * ============================================================================
 *
 * S3
 * ──
 * Q: What is S3 consistency model?
 *    Strong read-after-write consistency for all operations (since Dec 2020).
 *    Both new PUTs and overwrite PUTs/DELETEs are strongly consistent.
 *
 * Q: How do you secure S3 objects?
 *    Bucket policy + IAM + ACL (legacy). Block Public Access. SSE (S3/KMS/C).
 *    Pre-signed URLs for temporary access. VPC endpoint for private access.
 *    Object Lock (WORM) for compliance.
 *
 * Q: S3 vs EFS vs EBS?
 *    S3: object store, HTTP, infinite scale, any client, global
 *    EFS: NFS, POSIX, shared across multiple EC2/Lambda, auto-scales
 *    EBS: block store, one EC2 at a time (except Multi-Attach), lower latency
 *
 * LAMBDA
 * ──────
 * Q: How do you reduce Lambda cold starts?
 *    1. SnapStart (Java 21): restore from snapshot, sub-second cold start
 *    2. Provisioned concurrency: pre-warmed instances
 *    3. Optimize deployment: Lambda Layers, smaller JARs, Graal native image
 *    4. Keep handler classes small; move init to static block
 *
 * Q: Lambda vs Fargate vs EC2?
 *    Lambda: event-driven, 15min max, pay per ms, auto-scale to 0
 *    Fargate: containerized, no server management, longer tasks, predictable load
 *    EC2: full control, longest tasks, stateful, lowest cost at scale
 *
 * Q: How does Lambda concurrency work?
 *    Concurrent executions = requests × avg duration.
 *    Reserved concurrency: cap for specific function (no burst from it).
 *    Provisioned concurrency: always-warm pool (costs $).
 *    Burst limit: 3,000 initial burst then +500/min per region.
 *
 * DYNAMODB
 * ────────
 * Q: When would you use DynamoDB vs RDS?
 *    DynamoDB: single-digit ms at any scale, flexible schema, known access patterns,
 *              serverless, global tables, event-driven via Streams.
 *    RDS: complex joins, ad-hoc queries, ACID transactions, reporting, relational data.
 *
 * Q: How do you avoid hot partitions in DynamoDB?
 *    Write sharding: append random suffix to PK (USER#123#1, USER#123#2 ... #N)
 *    Scatter-gather: write to N shards, read all N and aggregate
 *    DAX: cache hot reads
 *    Time-based partitioning: PK includes time bucket
 *
 * Q: What is the maximum item size in DynamoDB?
 *    400KB per item. For larger: store in S3, store reference (S3 URL) in DynamoDB.
 *
 * SQS / SNS
 * ─────────
 * Q: SQS vs SNS vs EventBridge?
 *    SQS: durable queue, decoupling, at-least-once, pull-based
 *    SNS: pub/sub, push-based, fan-out to N subscribers
 *    EventBridge: event bus, rich filtering, schema registry, 20+ native sources,
 *                 replay, archiving — use for complex event routing/integration
 *
 * Q: How do you handle exactly-once processing with SQS?
 *    Use SQS FIFO with message deduplication ID (5-min dedup window).
 *    For Standard SQS: make your Lambda/consumer idempotent (DynamoDB conditional write).
 *
 * VPC / NETWORKING
 * ────────────────
 * Q: Security Group vs NACL?
 *    SG: stateful (return traffic auto-allowed), instance level, allow only
 *    NACL: stateless (must allow both inbound and outbound), subnet level, allow + deny
 *
 * Q: How does a private instance access the internet?
 *    Via NAT Gateway (managed, in public subnet) + route in private subnet RT to NAT GW.
 *    NAT Gateway is AZ-scoped — deploy one per AZ for HA.
 *
 * Q: What is VPC Peering vs Transit Gateway?
 *    Peering: 1-to-1, non-transitive (A↔B, B↔C does NOT mean A↔C)
 *    Transit Gateway: hub-and-spoke, transitive, connects VPCs + VPNs + Direct Connect
 *
 * ARCHITECTURE
 * ────────────
 * Q: How do you design for high availability on AWS?
 *    Multi-AZ deployments. ALB across AZs. RDS Multi-AZ. DynamoDB Global Tables.
 *    Route 53 health checks + failover. Auto Scaling groups. SQS for async decoupling.
 *
 * Q: How do you handle distributed transactions across microservices?
 *    Saga pattern: choreography (events) or orchestration (Step Functions).
 *    Outbox pattern: write to DB + outbox table atomically → CDC → publish event.
 *    Avoid distributed 2PC — prefer eventual consistency with compensating transactions.
 *
 * Q: Explain the 12-Factor app in AWS context.
 *    Config via environment variables (SSM/Secrets Manager).
 *    Stateless processes (Lambda/ECS → S3/DynamoDB for state).
 *    Logs as streams → CloudWatch Logs → S3/OpenSearch.
 *    Dev/prod parity → CDK/Terraform infrastructure as code.
 *
 * SECURITY
 * ────────
 * Q: IAM best practices?
 *    Least privilege. No root account usage. Roles for EC2/Lambda (instance profiles).
 *    MFA on human users. Rotate access keys. Use AWS Organizations SCPs for guardrails.
 *    Resource-based policies for cross-account. Never hard-code credentials in code.
 *
 * Q: How do you encrypt data in AWS?
 *    In-transit: TLS/HTTPS (ACM manages certs), VPN, TLS between services.
 *    At rest: SSE-S3, SSE-KMS (audit via CloudTrail), SSE-C (you manage key).
 *    KMS: managed, automatic rotation, envelope encryption.
 *    Client-side: encrypt before sending (AWS Encryption SDK).
 *
 * OBSERVABILITY
 * ─────────────
 * Q: How do you monitor a serverless application?
 *    CloudWatch Metrics: Lambda duration/errors/throttles/concurrent executions.
 *    CloudWatch Logs: structured JSON logs, Log Insights for queries.
 *    X-Ray: distributed tracing, service map, cold start visibility.
 *    Lambda Powertools: structured logging, tracing, custom metrics (EMF).
 *    Alarms: error rate > N%, DLQ depth > 0, Lambda throttles > 0.
 *
 * ============================================================================
 * KEY JAVA AWS SDK v2 BEST PRACTICES
 * ============================================================================
 *
 * 1.  Reuse SDK clients — they are thread-safe and expensive to create.
 *     Create once (static/singleton), share across invocations.
 *
 * 2.  Use DefaultCredentialsProvider — never hard-code credentials.
 *
 * 3.  Always set region explicitly — avoid implicit defaults.
 *
 * 4.  Use paginators for list operations — never assume single page.
 *
 * 5.  Handle SdkException (SdkClientException for local errors,
 *     AwsServiceException for service errors).
 *
 * 6.  Implement exponential backoff for retryable errors (SDK retries by default,
 *     but check RetryPolicy for customization).
 *
 * 7.  Close async clients and HTTP clients (try-with-resources or explicit close).
 *
 * 8.  Use waiter API for polling (not manual loops) — built-in backoff.
 *
 * 9.  Use S3 Transfer Manager for large file transfers (parallel multipart).
 *
 * 10. For Lambda: initialize SDK clients outside handler (static block)
 *     to reuse across warm invocations.
 *
 * 11. Set timeouts on SDK clients (apiCallTimeout, apiCallAttemptTimeout)
 *     to prevent Lambda timeouts due to hanging SDK calls.
 *
 * 12. Use Enhanced DynamoDB Client (DynamoDbEnhancedClient) with @DynamoDbBean
 *     annotations for cleaner, type-safe DynamoDB access.
 *
 * 13. Use SSM Parameter Store / Secrets Manager for config — never in env vars (plain text).
 *     Cache secret values; don't call Secrets Manager on every Lambda invocation.
 *
 * 14. Use structured logging (JSON) in Lambda — CloudWatch Logs Insights queryable.
 *
 * 15. Add X-Ray tracing: AWSXRayRecorderBuilder + @XRayEnabled or withTracingInterceptor.
 *
 * Good luck with your AWS interview!
 */
