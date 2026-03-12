// ============================================================================
// KAFKA + .NET INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience | Microservices & .NET Focus
// ============================================================================
//
// TOPIC COVERAGE:
//   1.  Kafka Core Concepts (Brokers, Topics, Partitions, Offsets, Replication)
//   2.  Producer Configuration & Delivery Semantics
//   3.  Consumer Groups, Partition Assignment & Rebalancing
//   4.  Exactly-Once Semantics (EOS) & Idempotent Producers
//   5.  Confluent.Kafka in .NET - Full Producer/Consumer Setup
//   6.  Schema Registry & Avro/JSON Schema Serialization
//   7.  Error Handling, Dead Letter Queues & Retry Patterns
//   8.  Kafka Streams & KSQL Concepts (with .NET consumer equivalents)
//   9.  Outbox Pattern & Transactional Messaging
//  10.  Kafka in Microservices: Event-Driven Architecture Patterns
//  11.  Consumer Lag Monitoring & Observability
//  12.  Backpressure, Flow Control & High-Throughput Tuning
//  13.  Security: TLS, SASL, ACLs
//  14.  Compacted Topics & Event Sourcing
//  15.  Saga Pattern over Kafka
//  16.  Interview Q&A Quick Reference
//
// ============================================================================

using Confluent.Kafka;
using Confluent.Kafka.Admin;
using Confluent.SchemaRegistry;
using Confluent.SchemaRegistry.Serdes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Transactions;

// ============================================================================
// SECTION 1: KAFKA CORE CONCEPTS
// ============================================================================

#region Core Concepts (Theory + Code Commentary)

/*
 * KAFKA ARCHITECTURE FUNDAMENTALS
 * ================================
 *
 * BROKER:
 *   - A Kafka server. A cluster = multiple brokers.
 *   - Brokers store partitions and serve producers/consumers.
 *   - One broker is elected the Controller (manages partition leadership).
 *
 * TOPIC:
 *   - A named, append-only log of records.
 *   - Divided into PARTITIONS for parallelism and scalability.
 *   - Topics are identified by name; no schema enforced at broker level.
 *
 * PARTITION:
 *   - Ordered, immutable sequence of records.
 *   - Each partition is owned by ONE broker (the leader).
 *   - Other brokers may hold replicas (followers).
 *   - KEY RULE: records with the same key go to the same partition → ordering guarantee per key.
 *   - Partitions are the unit of parallelism for producers AND consumers.
 *
 * OFFSET:
 *   - A monotonically increasing integer per partition.
 *   - Consumers commit offsets to track progress.
 *   - Committing is separate from processing → source of at-least-once vs exactly-once tension.
 *
 * REPLICATION:
 *   - Replication Factor = N means 1 leader + (N-1) followers.
 *   - ISR (In-Sync Replicas): followers that are caught up.
 *   - acks=all means the leader waits for all ISR to acknowledge before confirming.
 *
 * CONSUMER GROUP:
 *   - Multiple consumers sharing a group.id split partitions among themselves.
 *   - Each partition is consumed by EXACTLY ONE consumer in the group.
 *   - Different groups = independent reads of same topic (fan-out).
 *
 * RETENTION:
 *   - Time-based: retention.ms (default 7 days).
 *   - Size-based: retention.bytes.
 *   - Compacted topics: retain only the latest record per key (tombstone = delete).
 *
 * ZOOKEEPER vs KRaft:
 *   - Kafka historically required ZooKeeper for cluster metadata.
 *   - KRaft (Kafka Raft) replaces ZooKeeper from Kafka 3.3 (production-ready).
 *   - Interview tip: mention KRaft awareness; it reduces operational complexity.
 */

public static class KafkaCoreConceptsDemo
{
    // Topic naming conventions matter in microservices
    // Pattern: <domain>.<entity>.<event>
    public const string OrderCreatedTopic     = "order.order.created";
    public const string OrderShippedTopic     = "order.order.shipped";
    public const string PaymentProcessedTopic = "payment.payment.processed";
    public const string DeadLetterTopic       = "dlq.order.order.created"; // DLQ naming: dlq.<original>

    // Partition count decisions:
    // - More partitions = more parallelism but more overhead (file handles, rebalancing)
    // - Rule of thumb: num_partitions >= max consumer instances you'll ever deploy
    // - Never reduce partitions after creation (use new topic + migration)
    public const int DefaultPartitions       = 12;
    public const int DefaultReplicationFactor = 3;
}

#endregion

// ============================================================================
// SECTION 2: PRODUCER CONFIGURATION & DELIVERY SEMANTICS
// ============================================================================

#region Producer - Configuration & Delivery Guarantees

/*
 * DELIVERY SEMANTICS:
 *
 * AT-MOST-ONCE:   acks=0  → fire and forget, message may be lost. Fastest.
 * AT-LEAST-ONCE:  acks=1 or acks=all, retries > 0 → duplicates possible on retry.
 * EXACTLY-ONCE:   acks=all + enable.idempotence=true + transactional.id set.
 *
 * IDEMPOTENT PRODUCER:
 *   - Assigns each record a sequence number.
 *   - Broker deduplicates retries within a session (per producer epoch).
 *   - Enabled automatically when enable.idempotence=true.
 *   - Requires: acks=all, retries=Int32.Max, max.in.flight=5 (or 1 for strict ordering).
 *
 * PRODUCER BATCHING:
 *   - linger.ms: wait this long to accumulate records into a batch.
 *   - batch.size: max bytes per batch.
 *   - compression.type: lz4 / snappy / gzip / zstd. Use lz4 for low-latency, zstd for high-throughput.
 *
 * COMMON INTERVIEW TRAP:
 *   Q: "Can you lose messages with acks=1?"
 *   A: YES. If the leader acks then crashes before follower replication, the message is lost
 *      when the follower becomes the new leader. Use acks=all for durability.
 */

public class KafkaProducerFactory
{
    // At-least-once producer (most common in microservices)
    public static IProducer<string, string> CreateAtLeastOnceProducer(string bootstrapServers)
    {
        var config = new ProducerConfig
        {
            BootstrapServers = bootstrapServers,
            // Durability: wait for all ISR to acknowledge
            Acks                        = Acks.All,
            // Retries: high value; rely on delivery.timeout.ms for overall bound
            MessageSendMaxRetries       = int.MaxValue,
            // Retry backoff
            RetryBackoffMs              = 100,
            // Max time producer waits for full delivery (including retries)
            DeliveryTimeoutMs           = 120_000,     // 2 minutes
            // Batching for throughput
            LingerMs                    = 5,           // 5ms batching window
            BatchSize                   = 16_384,      // 16KB batch
            CompressionType             = CompressionType.Lz4,
            // Important: prevent out-of-order delivery on retry
            MaxInFlight                 = 5,
            // Enable idempotence (prevents duplicates from retries within session)
            EnableIdempotence           = true,
            // Socket tuning
            SocketKeepaliveEnable       = true,
            // Metadata refresh
            TopicMetadataRefreshIntervalMs = 300_000,
        };

        return new ProducerBuilder<string, string>(config)
            .SetErrorHandler((_, e)    => Console.Error.WriteLine($"Producer error: {e.Reason} | Fatal: {e.IsFatal}"))
            .SetLogHandler((_, log)    => Console.WriteLine($"[{log.Level}] {log.Message}"))
            .SetStatisticsHandler((_, json) => { /* Parse rdkafka stats JSON for metrics */ })
            .Build();
    }

    // Transactional producer for exactly-once semantics
    public static IProducer<string, string> CreateExactlyOnceProducer(
        string bootstrapServers,
        string transactionalId)   // Must be unique per producer instance
    {
        var config = new ProducerConfig
        {
            BootstrapServers  = bootstrapServers,
            Acks              = Acks.All,
            EnableIdempotence = true,
            TransactionalId   = transactionalId, // Enables EOS; survives producer restarts
            // With transactions, max.in.flight must be <= 5
            MaxInFlight       = 5,
            MessageSendMaxRetries = int.MaxValue,
        };

        return new ProducerBuilder<string, string>(config).Build();
    }
}

// Typed, injectable Kafka producer service
public interface IEventPublisher
{
    Task PublishAsync<T>(string topic, string key, T @event, CancellationToken ct = default);
}

public class KafkaEventPublisher : IEventPublisher, IDisposable
{
    private readonly IProducer<string, string> _producer;
    private readonly ILogger<KafkaEventPublisher> _logger;

    public KafkaEventPublisher(
        IOptions<KafkaProducerOptions> options,
        ILogger<KafkaEventPublisher> logger)
    {
        _logger   = logger;
        _producer = KafkaProducerFactory.CreateAtLeastOnceProducer(options.Value.BootstrapServers);
    }

    public async Task PublishAsync<T>(string topic, string key, T @event, CancellationToken ct = default)
    {
        var payload = JsonSerializer.Serialize(@event);
        var message = new Message<string, string>
        {
            Key     = key,
            Value   = payload,
            Headers = new Headers
            {
                // Envelope headers for traceability
                { "event-type",    Encoding.UTF8.GetBytes(typeof(T).Name) },
                { "correlation-id", Encoding.UTF8.GetBytes(Activity.Current?.TraceId.ToString() ?? Guid.NewGuid().ToString()) },
                { "timestamp",     Encoding.UTF8.GetBytes(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString()) },
                { "source-service", Encoding.UTF8.GetBytes("order-service") },
            }
        };

        try
        {
            var result = await _producer.ProduceAsync(topic, message, ct);
            _logger.LogInformation(
                "Published {EventType} to {Topic} [{Partition}@{Offset}]",
                typeof(T).Name, topic, result.Partition.Value, result.Offset.Value);
        }
        catch (ProduceException<string, string> ex)
        {
            _logger.LogError(ex, "Failed to produce to {Topic}: {Reason}", topic, ex.Error.Reason);
            throw;
        }
    }

    // Synchronous fire-and-forget with delivery callback (high throughput)
    public void PublishFireAndForget<T>(string topic, string key, T @event)
    {
        var payload = JsonSerializer.Serialize(@event);
        _producer.Produce(
            topic,
            new Message<string, string> { Key = key, Value = payload },
            deliveryReport =>
            {
                if (deliveryReport.Error.IsError)
                    _logger.LogError("Delivery failed: {Reason}", deliveryReport.Error.Reason);
                else
                    _logger.LogDebug("Delivered to {TopicPartitionOffset}", deliveryReport.TopicPartitionOffset);
            });
        // Call Flush before shutdown to drain in-flight messages
    }

    public void Dispose()
    {
        // Flush ensures all queued messages are delivered before shutdown
        _producer.Flush(TimeSpan.FromSeconds(30));
        _producer.Dispose();
    }
}

public class KafkaProducerOptions
{
    public string BootstrapServers { get; set; } = "localhost:9092";
}

#endregion

// ============================================================================
// SECTION 3: CONSUMER GROUPS, REBALANCING & OFFSET MANAGEMENT
// ============================================================================

#region Consumer - Groups, Offsets & Rebalancing

/*
 * CONSUMER GROUP REBALANCING:
 *   Triggered when: consumer joins/leaves, heartbeat timeout, subscription change.
 *
 *   EAGER (Stop-The-World) rebalance:
 *     - All consumers revoke all partitions → rebalance → reassign.
 *     - Default in older clients. High pause.
 *
 *   COOPERATIVE (Incremental) rebalance:
 *     - Only affected partitions are revoked/reassigned.
 *     - Enabled via partition.assignment.strategy = CooperativeSticky.
 *     - Production best practice.
 *
 * OFFSET COMMIT STRATEGIES:
 *   auto.commit (enable.auto.commit=true): simple, at-least-once if process crashes mid-batch.
 *   Manual sync commit:  commitSync() after processing — safest, blocks.
 *   Manual async commit: commitAsync()  — higher throughput, possible duplicate on failure.
 *   Manual per-message: commit after each message — lowest throughput, strongest guarantee.
 *
 * IMPORTANT: auto.commit commits the LAST POLLED offset, not the last PROCESSED offset.
 *   This can cause message loss if consumer crashes after commit but before processing.
 *   Always disable auto-commit in critical microservices and commit manually.
 *
 * SESSION TIMEOUT vs HEARTBEAT INTERVAL:
 *   session.timeout.ms:      Broker removes consumer from group if no heartbeat (default 45s).
 *   heartbeat.interval.ms:   How often consumer sends heartbeat (should be ~1/3 of session.timeout).
 *   max.poll.interval.ms:    Max time between poll() calls before consumer is considered dead.
 *                            CRITICAL: if processing is slow, increase this or you'll get rebalances.
 */

public class KafkaConsumerService : BackgroundService
{
    private readonly ILogger<KafkaConsumerService> _logger;
    private readonly KafkaConsumerOptions _options;
    private readonly IOrderEventHandler _handler;

    public KafkaConsumerService(
        IOptions<KafkaConsumerOptions> options,
        IOrderEventHandler handler,
        ILogger<KafkaConsumerService> logger)
    {
        _options = options.Value;
        _handler = handler;
        _logger  = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers             = _options.BootstrapServers,
            GroupId                      = _options.GroupId,
            // Never auto-commit — we commit manually after successful processing
            EnableAutoCommit             = false,
            // Start from earliest if no committed offset exists
            AutoOffsetReset              = AutoOffsetReset.Earliest,
            // Cooperative sticky: minimizes partition revocation during rebalance
            PartitionAssignmentStrategy  = PartitionAssignmentStrategy.CooperativeSticky,
            // If processing takes longer than this, broker kicks us out of the group
            MaxPollIntervalMs            = 300_000,  // 5 minutes
            SessionTimeoutMs             = 45_000,
            HeartbeatIntervalMs          = 15_000,
            // Fetch tuning
            FetchMinBytes                = 1,
            FetchWaitMaxMs               = 500,
            MaxPartitionFetchBytes       = 1_048_576, // 1MB
            // For exactly-once read (used in transactional read-process-write)
            IsolationLevel               = IsolationLevel.ReadCommitted,
            // Enable debug metrics
            StatisticsIntervalMs         = 60_000,
        };

        using var consumer = new ConsumerBuilder<string, string>(config)
            .SetPartitionsAssignedHandler((c, partitions) =>
            {
                _logger.LogInformation("Assigned partitions: {Partitions}",
                    string.Join(", ", partitions.Select(p => p.ToString())));
                // Optionally override offset: return partitions.Select(tp => new TopicPartitionOffset(tp, Offset.Beginning))
            })
            .SetPartitionsRevokedHandler((c, partitions) =>
            {
                _logger.LogWarning("Revoking partitions (commit before revoke): {Partitions}",
                    string.Join(", ", partitions.Select(p => p.ToString())));
                // CRITICAL: commit offsets here for cooperative rebalance
                c.Commit(partitions.Select(tp => new TopicPartitionOffset(tp.TopicPartition, tp.Offset)));
            })
            .SetPartitionsLostHandler((c, partitions) =>
            {
                // Lost = timeout/crash during rebalance, cannot commit
                _logger.LogError("Partitions LOST (no commit possible): {Partitions}",
                    string.Join(", ", partitions.Select(p => p.ToString())));
            })
            .SetErrorHandler((_, e) =>
            {
                if (e.IsFatal)
                    _logger.LogCritical("Fatal consumer error: {Reason}", e.Reason);
                else
                    _logger.LogWarning("Consumer error: {Reason}", e.Reason);
            })
            .Build();

        consumer.Subscribe(_options.Topics);

        try
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                ConsumeResult<string, string>? result = null;
                try
                {
                    // Poll with timeout — returns null if no message within window
                    result = consumer.Consume(TimeSpan.FromMilliseconds(500));
                    if (result is null || result.IsPartitionEOF) continue;

                    _logger.LogDebug("Received [{Topic}|{Partition}@{Offset}] Key={Key}",
                        result.Topic, result.Partition.Value, result.Offset.Value, result.Message.Key);

                    await _handler.HandleAsync(result.Message.Key, result.Message.Value, stoppingToken);

                    // Commit AFTER successful processing (store then commit)
                    // StoreOffset marks as "ready to commit" — actual commit batched
                    consumer.StoreOffset(result);
                    consumer.Commit(result); // explicit commit per message (safe but slower)
                }
                catch (ConsumeException ex)
                {
                    _logger.LogError(ex, "Consume error at offset {Offset}", result?.Offset.Value);
                    // Non-fatal: continue polling
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Processing error for message {Key}", result?.Message.Key);
                    // Send to DLQ or implement retry (see Section 7)
                    // Do NOT commit on processing failure unless you intentionally skip
                }
            }
        }
        finally
        {
            // Graceful shutdown: commit pending offsets and leave group
            consumer.Close(); // triggers final rebalance + offset commit
        }
    }
}

public class KafkaConsumerOptions
{
    public string BootstrapServers { get; set; } = "localhost:9092";
    public string GroupId          { get; set; } = "order-service";
    public List<string> Topics     { get; set; } = new();
}

public interface IOrderEventHandler
{
    Task HandleAsync(string key, string value, CancellationToken ct);
}

#endregion

// ============================================================================
// SECTION 4: EXACTLY-ONCE SEMANTICS (EOS) & TRANSACTIONS
// ============================================================================

#region Exactly-Once Semantics

/*
 * EXACTLY-ONCE IN KAFKA:
 *   Kafka's EOS guarantees that a message is WRITTEN to the destination topic
 *   EXACTLY ONCE even if the producer retries. It does NOT guarantee your downstream
 *   DB side-effects happen exactly once — you handle that with idempotency keys.
 *
 * HOW IT WORKS (Transactional Producer):
 *   1. Producer registers a transactional.id with the broker (gets a PID + epoch).
 *   2. Producer calls BeginTransaction().
 *   3. Producer sends messages (may span multiple topics/partitions).
 *   4. Producer calls CommitTransaction() or AbortTransaction().
 *   5. Broker makes all messages in the transaction atomically visible.
 *   6. Consumers with isolation.level=read_committed only see committed data.
 *
 * READ-PROCESS-WRITE (Consume → Transform → Produce):
 *   - Consume from topic A, process, produce to topic B.
 *   - Include consumer offset commit inside the transaction → atomic.
 *   - SendOffsetsToTransaction() tells broker: "commit these consumer offsets
 *     as part of this transaction."
 *
 * ZOMBIE FENCING:
 *   - If a producer crashes and restarts with the same transactional.id,
 *     the broker bumps the epoch, fencing (rejecting) the old zombie producer.
 *   - This is critical in Kubernetes where pods can restart with same config.
 */

public class TransactionalConsumerProducer : BackgroundService
{
    private readonly ILogger<TransactionalConsumerProducer> _logger;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var consumerConfig = new ConsumerConfig
        {
            BootstrapServers  = "localhost:9092",
            GroupId           = "eos-processor",
            EnableAutoCommit  = false,
            AutoOffsetReset   = AutoOffsetReset.Earliest,
            IsolationLevel    = IsolationLevel.ReadCommitted, // Only read committed txn messages
        };

        var producerConfig = new ProducerConfig
        {
            BootstrapServers  = "localhost:9092",
            TransactionalId   = "eos-processor-1",  // Unique per instance
            EnableIdempotence = true,
            Acks              = Acks.All,
        };

        using var consumer = new ConsumerBuilder<string, string>(consumerConfig).Build();
        using var producer = new ProducerBuilder<string, string>(producerConfig).Build();

        // MUST call InitTransactions once before any transaction
        producer.InitTransactions(TimeSpan.FromSeconds(30));
        consumer.Subscribe("order.order.created");

        while (!stoppingToken.IsCancellationRequested)
        {
            var batch = ConsumeBatch(consumer, batchSize: 50, timeout: TimeSpan.FromMilliseconds(500));
            if (batch.Count == 0) continue;

            producer.BeginTransaction();
            try
            {
                foreach (var record in batch)
                {
                    var enriched = EnrichEvent(record.Message.Value);
                    await producer.ProduceAsync(
                        "order.order.enriched",
                        new Message<string, string> { Key = record.Message.Key, Value = enriched },
                        stoppingToken);
                }

                // Atomically commit consumer offsets within the producer transaction
                // This guarantees read-process-write is atomic
                var offsets = batch
                    .GroupBy(r => r.TopicPartition)
                    .Select(g => new TopicPartitionOffset(
                        g.Key,
                        g.Max(r => r.Offset) + 1)) // +1: next offset to consume
                    .ToList();

                producer.SendOffsetsToTransaction(offsets, consumer.ConsumerGroupMetadata, TimeSpan.FromSeconds(10));
                producer.CommitTransaction();

                _logger.LogInformation("EOS batch committed: {Count} records", batch.Count);
            }
            catch (KafkaException ex) when (ex.Error.Code == ErrorCode.InvalidProducerEpoch)
            {
                // Zombie fencing — our epoch was bumped by a newer instance
                _logger.LogCritical("Zombie fencing detected — shutting down this instance");
                producer.AbortTransaction();
                throw; // Let the host restart with fresh transactional.id
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Transaction failed, aborting");
                producer.AbortTransaction(); // Records NOT visible to consumers
                // Retry logic or DLQ here
            }
        }
    }

    private List<ConsumeResult<string, string>> ConsumeBatch(
        IConsumer<string, string> consumer,
        int batchSize,
        TimeSpan timeout)
    {
        var batch = new List<ConsumeResult<string, string>>(batchSize);
        var deadline = DateTime.UtcNow + timeout;

        while (batch.Count < batchSize && DateTime.UtcNow < deadline)
        {
            var result = consumer.Consume(TimeSpan.FromMilliseconds(50));
            if (result is not null && !result.IsPartitionEOF)
                batch.Add(result);
        }
        return batch;
    }

    private string EnrichEvent(string raw) => raw; // Business logic placeholder
}

#endregion

// ============================================================================
// SECTION 5: SCHEMA REGISTRY & SERIALIZATION
// ============================================================================

#region Schema Registry & Avro/JSON Schema

/*
 * WHY SCHEMA REGISTRY:
 *   - Without schema enforcement, producers can publish malformed payloads
 *     that silently break consumers (schema drift = distributed system nightmare).
 *   - Registry stores versioned schemas (Avro, Protobuf, JSON Schema).
 *   - Wire format: 1 magic byte (0x00) + 4-byte schema ID + Avro/Protobuf payload.
 *   - Consumers look up schema by ID to deserialize.
 *
 * COMPATIBILITY MODES:
 *   BACKWARD:  New schema can read data written with old schema.  (consumers upgrade first)
 *   FORWARD:   Old schema can read data written with new schema.  (producers upgrade first)
 *   FULL:      Both backward + forward compatible.
 *   NONE:      No compatibility check (dangerous in production).
 *
 * INTERVIEW TRAP:
 *   Q: "A producer adds a new required field. What breaks?"
 *   A: Old consumers fail to deserialize unless the field has a default value.
 *      With BACKWARD compatibility enforced, the registry rejects a schema that
 *      adds a required field without a default → prevents the deployment.
 */

// Avro-generated class (typically from .avsc file via Avro.CodeGen NuGet)
// For interview: show you understand the wire protocol even without generated code
public class OrderCreatedEvent
{
    public string OrderId    { get; set; } = default!;
    public string CustomerId { get; set; } = default!;
    public decimal Amount    { get; set; }
    public string Currency   { get; set; } = "USD";
    public DateTime CreatedAt { get; set; }
}

public class SchemaRegistryProducerExample
{
    // JSON Schema serializer (simpler for .NET microservices than Avro)
    public static async Task RunAsync()
    {
        var schemaRegistryConfig = new SchemaRegistryConfig
        {
            Url = "http://schema-registry:8081",
            // Authentication if Confluent Cloud:
            // BasicAuthUserInfo = "key:secret"
        };

        var jsonSerializerConfig = new JsonSerializerConfig
        {
            BufferBytes          = 100,
            AutoRegisterSchemas  = true, // Register schema on first use (dev only)
            // Production: set to false and pre-register schemas in CI/CD
            UseLatestVersion     = false,
        };

        using var schemaRegistry = new CachedSchemaRegistryClient(schemaRegistryConfig);
        using var producer = new ProducerBuilder<string, OrderCreatedEvent>(
                new ProducerConfig { BootstrapServers = "localhost:9092", Acks = Acks.All })
            .SetValueSerializer(new JsonSerializer<OrderCreatedEvent>(schemaRegistry, jsonSerializerConfig))
            .Build();

        var @event = new OrderCreatedEvent
        {
            OrderId    = Guid.NewGuid().ToString(),
            CustomerId = "cust-123",
            Amount     = 99.99m,
            CreatedAt  = DateTime.UtcNow,
        };

        await producer.ProduceAsync(
            "order.order.created",
            new Message<string, OrderCreatedEvent> { Key = @event.OrderId, Value = @event });
    }
}

public class SchemaRegistryConsumerExample
{
    public static async Task RunAsync(CancellationToken ct)
    {
        var schemaRegistryConfig = new SchemaRegistryConfig { Url = "http://schema-registry:8081" };

        using var schemaRegistry = new CachedSchemaRegistryClient(schemaRegistryConfig);
        using var consumer = new ConsumerBuilder<string, OrderCreatedEvent>(
                new ConsumerConfig
                {
                    BootstrapServers = "localhost:9092",
                    GroupId          = "notification-service",
                    EnableAutoCommit = false,
                    AutoOffsetReset  = AutoOffsetReset.Earliest,
                })
            .SetValueDeserializer(new JsonDeserializer<OrderCreatedEvent>().AsSyncOverAsync())
            .Build();

        consumer.Subscribe("order.order.created");

        while (!ct.IsCancellationRequested)
        {
            var result = consumer.Consume(ct);
            var @event = result.Message.Value;
            Console.WriteLine($"Order {event.OrderId} from {event.CustomerId} for {event.Amount} {event.Currency}");
            consumer.Commit(result);
        }
    }
}

#endregion

// ============================================================================
// SECTION 6: ADMIN CLIENT - TOPIC MANAGEMENT
// ============================================================================

#region Admin Client

/*
 * AdminClient use cases in microservices:
 *   - Create topics on service startup if not exists (bootstrap pattern).
 *   - Query partition count to make routing decisions.
 *   - Reset consumer group offsets for replays.
 *   - Monitor consumer lag programmatically.
 *
 * INTERVIEW TIP:
 *   Never create topics in production from application code with auto.create.topics.enable=true.
 *   Typos in topic names silently create new topics. Use IaC (Terraform/Ansible) or AdminClient
 *   with strict topic existence checks.
 */

public class KafkaTopicManager
{
    private readonly IAdminClient _adminClient;
    private readonly ILogger<KafkaTopicManager> _logger;

    public KafkaTopicManager(string bootstrapServers, ILogger<KafkaTopicManager> logger)
    {
        _logger = logger;
        _adminClient = new AdminClientBuilder(
            new AdminClientConfig { BootstrapServers = bootstrapServers })
            .Build();
    }

    public async Task EnsureTopicExistsAsync(
        string topicName,
        int numPartitions    = 12,
        short replicationFactor = 3)
    {
        try
        {
            var metadata = _adminClient.GetMetadata(topicName, TimeSpan.FromSeconds(10));
            if (metadata.Topics.Any(t => t.Topic == topicName && t.Error.IsError == false))
            {
                _logger.LogInformation("Topic {Topic} already exists", topicName);
                return;
            }

            await _adminClient.CreateTopicsAsync(new[]
            {
                new TopicSpecification
                {
                    Name              = topicName,
                    NumPartitions     = numPartitions,
                    ReplicationFactor = replicationFactor,
                    Configs = new Dictionary<string, string>
                    {
                        ["retention.ms"]        = TimeSpan.FromDays(7).TotalMilliseconds.ToString("0"),
                        ["compression.type"]    = "lz4",
                        ["min.insync.replicas"]  = "2",  // ISR must be >= 2 before acks=all succeeds
                        ["cleanup.policy"]       = "delete", // or "compact" for event sourcing
                        ["max.message.bytes"]    = "1048576", // 1MB max per message
                    }
                }
            });

            _logger.LogInformation("Topic {Topic} created with {Partitions} partitions", topicName, numPartitions);
        }
        catch (CreateTopicsException ex) when (ex.Results[0].Error.Code == ErrorCode.TopicAlreadyExists)
        {
            _logger.LogInformation("Topic {Topic} already exists (race condition)", topicName);
        }
    }

    // Describe consumer group lag
    public async Task<Dictionary<string, long>> GetConsumerLagAsync(
        string groupId,
        string topicName)
    {
        // Get current committed offsets for the group
        var groupOffsets = await _adminClient.ListConsumerGroupOffsetsAsync(
            new[] { new ConsumerGroupTopicPartitions(groupId) });

        // Get latest end offsets for topic
        using var tempConsumer = new ConsumerBuilder<Ignore, Ignore>(
                new ConsumerConfig { BootstrapServers = "localhost:9092", GroupId = "admin-lag-check" })
            .Build();

        var topicPartitions = groupOffsets.First().Partitions
            .Where(p => p.Topic == topicName)
            .Select(p => p.TopicPartition)
            .ToList();

        var endOffsets = tempConsumer.QueryWatermarkOffsets(
            topicPartitions.First(),
            TimeSpan.FromSeconds(5));

        var lag = new Dictionary<string, long>();
        foreach (var tp in groupOffsets.First().Partitions.Where(p => p.Topic == topicName))
        {
            var wo = tempConsumer.QueryWatermarkOffsets(tp.TopicPartition, TimeSpan.FromSeconds(5));
            lag[$"{tp.Topic}-{tp.Partition.Value}"] = wo.High - tp.Offset.Value;
        }

        tempConsumer.Close();
        return lag;
    }

    // Reset offsets to beginning (replay all messages)
    public async Task ResetOffsetsToBeginningAsync(string groupId, string topicName)
    {
        var metadata  = _adminClient.GetMetadata(topicName, TimeSpan.FromSeconds(10));
        var topicMeta = metadata.Topics.First(t => t.Topic == topicName);

        var offsets = topicMeta.Partitions.Select(p =>
            new TopicPartitionOffset(
                new TopicPartition(topicName, new Partition(p.PartitionId)),
                Offset.Beginning))
            .ToList();

        await _adminClient.AlterConsumerGroupOffsetsAsync(
            new[] { new ConsumerGroupTopicPartitions(groupId, offsets) });

        _logger.LogWarning("Reset offsets to beginning for group {Group} on {Topic}", groupId, topicName);
    }
}

#endregion

// ============================================================================
// SECTION 7: ERROR HANDLING, RETRIES & DEAD LETTER QUEUES
// ============================================================================

#region Error Handling & DLQ

/*
 * ERROR CATEGORIES IN KAFKA CONSUMERS:
 *
 *   TRANSIENT errors (should retry):
 *     - DB connection timeout, external API unavailable, network blip.
 *     - Strategy: exponential backoff, bounded retry count.
 *
 *   POISON PILL / PERMANENT errors (should NOT retry infinitely):
 *     - Malformed JSON, schema mismatch, business rule violation.
 *     - Strategy: send to Dead Letter Queue (DLQ) with error metadata.
 *
 * DLQ PATTERN:
 *   - DLQ topic: dlq.<original-topic>
 *   - Preserve original message + add headers: error-reason, retry-count, failed-at.
 *   - Separate DLQ consumer service re-processes or alerts on-call.
 *
 * RETRY TOPIC PATTERN (for delayed retry):
 *   - retry.30s.<topic>  → consumed after 30s delay
 *   - retry.5m.<topic>   → consumed after 5min delay
 *   - dlq.<topic>        → manual intervention needed
 *   - Consumer reads from retry topic, checks timestamp header, sleeps if too early.
 *   - This is preferable to Thread.Sleep inside the main consumer (blocks the partition).
 */

public class RetryPolicy
{
    public int MaxRetries        { get; set; } = 3;
    public TimeSpan InitialDelay { get; set; } = TimeSpan.FromSeconds(1);
    public double Multiplier     { get; set; } = 2.0; // Exponential backoff
}

public class DeadLetterQueuePublisher
{
    private readonly IProducer<string, string> _producer;

    public DeadLetterQueuePublisher(string bootstrapServers)
    {
        _producer = new ProducerBuilder<string, string>(
            new ProducerConfig { BootstrapServers = bootstrapServers, Acks = Acks.All })
            .Build();
    }

    public async Task SendToDeadLetterAsync<TError>(
        ConsumeResult<string, string> original,
        TError exception,
        int retryCount) where TError : Exception
    {
        var dlqTopic  = $"dlq.{original.Topic}";
        var headers   = new Headers();

        // Preserve original headers
        if (original.Message.Headers != null)
            foreach (var h in original.Message.Headers)
                headers.Add(h);

        // Add DLQ-specific headers
        headers.Add("dlq-reason",        Encoding.UTF8.GetBytes(exception.Message));
        headers.Add("dlq-exception-type", Encoding.UTF8.GetBytes(exception.GetType().Name));
        headers.Add("dlq-retry-count",   Encoding.UTF8.GetBytes(retryCount.ToString()));
        headers.Add("dlq-original-topic", Encoding.UTF8.GetBytes(original.Topic));
        headers.Add("dlq-original-partition", Encoding.UTF8.GetBytes(original.Partition.Value.ToString()));
        headers.Add("dlq-original-offset",    Encoding.UTF8.GetBytes(original.Offset.Value.ToString()));
        headers.Add("dlq-failed-at",     Encoding.UTF8.GetBytes(DateTimeOffset.UtcNow.ToString("o")));

        await _producer.ProduceAsync(dlqTopic, new Message<string, string>
        {
            Key     = original.Message.Key,
            Value   = original.Message.Value,
            Headers = headers,
        });
    }
}

// Resilient consumer that retries with backoff, then DLQ
public class ResilientConsumerLoop
{
    private readonly RetryPolicy _retryPolicy;
    private readonly DeadLetterQueuePublisher _dlq;
    private readonly ILogger _logger;

    public ResilientConsumerLoop(RetryPolicy retryPolicy, DeadLetterQueuePublisher dlq, ILogger logger)
    {
        _retryPolicy = retryPolicy;
        _dlq         = dlq;
        _logger      = logger;
    }

    public async Task ProcessWithRetryAsync(
        ConsumeResult<string, string> record,
        Func<ConsumeResult<string, string>, Task> processFunc)
    {
        int attempt = 0;
        while (true)
        {
            try
            {
                await processFunc(record);
                return; // Success
            }
            catch (Exception ex) when (IsTransient(ex) && attempt < _retryPolicy.MaxRetries)
            {
                attempt++;
                var delay = TimeSpan.FromMilliseconds(
                    _retryPolicy.InitialDelay.TotalMilliseconds * Math.Pow(_retryPolicy.Multiplier, attempt - 1));
                _logger.LogWarning(ex, "Transient error on attempt {Attempt}, retrying in {Delay}ms", attempt, delay.TotalMilliseconds);
                await Task.Delay(delay);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Permanent failure for message {Key}, sending to DLQ", record.Message.Key);
                await _dlq.SendToDeadLetterAsync(record, ex, attempt);
                return; // Do NOT rethrow — commit offset and move on
            }
        }
    }

    private static bool IsTransient(Exception ex)
        => ex is TimeoutException
        || ex is TaskCanceledException
        || (ex.Message?.Contains("connection", StringComparison.OrdinalIgnoreCase) ?? false);
}

#endregion

// ============================================================================
// SECTION 8: OUTBOX PATTERN — TRANSACTIONAL MESSAGING
// ============================================================================

#region Outbox Pattern

/*
 * THE DUAL-WRITE PROBLEM:
 *   In a microservice, you often need to:
 *   1. Save to DB (e.g., create order).
 *   2. Publish event to Kafka.
 *
 *   If you do these separately, you can:
 *   - Save to DB but crash before publishing → event lost.
 *   - Publish to Kafka but DB write fails → event without corresponding state.
 *
 * OUTBOX PATTERN SOLUTION:
 *   1. Within the same DB transaction, write the domain record AND an OutboxMessage row.
 *   2. A background "relay" reads unprocessed OutboxMessage rows and publishes to Kafka.
 *   3. On successful publish, mark the OutboxMessage as processed (or delete it).
 *   4. At-least-once delivery guaranteed; consumers must be idempotent.
 *
 * POLLING vs CDC:
 *   Polling Relay: simple, slight latency, DB load.
 *   CDC (Change Data Capture) via Debezium: reads DB transaction log (binlog/WAL),
 *     near-zero latency, no polling load, but more infrastructure.
 *
 * INTERVIEW TIP:
 *   "How do you guarantee an event is published after a DB write?"
 *   → Outbox pattern. Never dual-write directly; always transactional outbox.
 */

public class OutboxMessage
{
    public Guid        Id            { get; set; } = Guid.NewGuid();
    public string      Topic         { get; set; } = default!;
    public string      MessageKey    { get; set; } = default!;
    public string      Payload       { get; set; } = default!;
    public string      EventType     { get; set; } = default!;
    public DateTime    CreatedAt     { get; set; } = DateTime.UtcNow;
    public bool        Processed     { get; set; }
    public DateTime?   ProcessedAt   { get; set; }
    public int         RetryCount    { get; set; }
}

// Domain service writes to both Orders table and Outbox in one transaction
public class OrderService
{
    // Simulated DbContext-style interface
    private readonly IOrderRepository _orders;
    private readonly IOutboxRepository _outbox;
    private readonly IDbTransaction _dbTransaction;

    public OrderService(IOrderRepository orders, IOutboxRepository outbox, IDbTransaction tx)
    {
        _orders      = orders;
        _outbox      = outbox;
        _dbTransaction = tx;
    }

    public async Task CreateOrderAsync(CreateOrderCommand cmd)
    {
        // Both writes happen in ONE database transaction
        using var tx = _dbTransaction;
        try
        {
            var order = new Order
            {
                Id         = Guid.NewGuid(),
                CustomerId = cmd.CustomerId,
                Amount     = cmd.Amount,
                Status     = OrderStatus.Created,
                CreatedAt  = DateTime.UtcNow,
            };

            await _orders.InsertAsync(order);

            // Outbox entry in the SAME transaction
            var outboxMsg = new OutboxMessage
            {
                Topic      = KafkaCoreConceptsDemo.OrderCreatedTopic,
                MessageKey = order.Id.ToString(),
                Payload    = JsonSerializer.Serialize(new OrderCreatedEvent
                {
                    OrderId    = order.Id.ToString(),
                    CustomerId = order.CustomerId,
                    Amount     = order.Amount,
                    CreatedAt  = order.CreatedAt,
                }),
                EventType  = nameof(OrderCreatedEvent),
            };

            await _outbox.InsertAsync(outboxMsg);
            tx.Commit();
        }
        catch
        {
            tx.Rollback(); // Both order AND outbox message are rolled back
            throw;
        }
    }
}

// Background relay reads outbox and publishes to Kafka
public class OutboxRelayService : BackgroundService
{
    private readonly IOutboxRepository _outbox;
    private readonly IEventPublisher _publisher;
    private readonly ILogger<OutboxRelayService> _logger;
    private static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(1);

    public OutboxRelayService(IOutboxRepository outbox, IEventPublisher publisher, ILogger<OutboxRelayService> logger)
    {
        _outbox    = outbox;
        _publisher = publisher;
        _logger    = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            var messages = await _outbox.GetUnprocessedAsync(batchSize: 100);

            foreach (var msg in messages)
            {
                try
                {
                    await _publisher.PublishAsync<string>(msg.Topic, msg.MessageKey, msg.Payload, stoppingToken);
                    await _outbox.MarkProcessedAsync(msg.Id);
                    _logger.LogDebug("Relayed outbox message {Id} to {Topic}", msg.Id, msg.Topic);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to relay outbox message {Id}, will retry", msg.Id);
                    await _outbox.IncrementRetryCountAsync(msg.Id);
                }
            }

            if (messages.Count == 0)
                await Task.Delay(PollInterval, stoppingToken);
        }
    }
}

// Stub types
public record CreateOrderCommand(string CustomerId, decimal Amount);
public class Order { public Guid Id; public string CustomerId = ""; public decimal Amount; public OrderStatus Status; public DateTime CreatedAt; }
public enum OrderStatus { Created, Confirmed, Shipped, Delivered, Cancelled }
public interface IOrderRepository { Task InsertAsync(Order o); }
public interface IOutboxRepository
{
    Task InsertAsync(OutboxMessage m);
    Task<List<OutboxMessage>> GetUnprocessedAsync(int batchSize);
    Task MarkProcessedAsync(Guid id);
    Task IncrementRetryCountAsync(Guid id);
}
public interface IDbTransaction : IDisposable { void Commit(); void Rollback(); }

#endregion

// ============================================================================
// SECTION 9: EVENT-DRIVEN MICROSERVICES PATTERNS
// ============================================================================

#region Event-Driven Architecture Patterns

/*
 * EVENT TYPES IN MICROSERVICES:
 *
 *   DOMAIN EVENT:     Something that happened in the domain (OrderCreated, PaymentFailed).
 *                     Published by the aggregate that owns the state.
 *   INTEGRATION EVENT: Cross-service domain event. Same concept but explicitly for inter-service comms.
 *   COMMAND:           Request to do something (ProcessPayment). Has one handler, sent to specific topic.
 *   QUERY:             Request for data. Kafka is NOT ideal for queries; use REST/gRPC instead.
 *
 * EVENT ENVELOPE PATTERN:
 *   Wrap all events in a common envelope with metadata. Allows generic consumers,
 *   routing, tracing, and schema evolution without changing the wire format structure.
 *
 * CHOREOGRAPHY vs ORCHESTRATION:
 *   Choreography:   Each service reacts to events independently. Loose coupling, harder to trace.
 *   Orchestration:  Central saga orchestrator directs services via commands. Easier to trace, central failure point.
 *
 * COMPETING CONSUMERS:
 *   Multiple instances of the same consumer group share partitions.
 *   Scale out = add consumer instances up to partition count.
 *   Beyond that, extra instances are idle (no partition to assign).
 *   → Always set partition count >= max expected consumer instances.
 */

// Generic event envelope for all Kafka messages
public sealed record EventEnvelope<T>
{
    public string     EventId       { get; init; } = Guid.NewGuid().ToString();
    public string     EventType     { get; init; } = typeof(T).Name;
    public string     SchemaVersion { get; init; } = "1.0";
    public string     CorrelationId { get; init; } = Activity.Current?.TraceId.ToString() ?? Guid.NewGuid().ToString();
    public string     CausationId   { get; init; } = string.Empty; // ID of the event that caused this one
    public string     Source        { get; init; } = string.Empty; // Service that produced this
    public DateTimeOffset OccurredAt { get; init; } = DateTimeOffset.UtcNow;
    public T          Data          { get; init; } = default!;
}

// Topic-per-event-type routing in a generic publisher
public class RoutingEventPublisher
{
    private readonly IProducer<string, string> _producer;
    private readonly Dictionary<Type, string> _topicMap;

    public RoutingEventPublisher(IProducer<string, string> producer)
    {
        _producer = producer;
        _topicMap = new Dictionary<Type, string>
        {
            [typeof(OrderCreatedEvent)] = KafkaCoreConceptsDemo.OrderCreatedTopic,
            // Add more mappings as domain grows
        };
    }

    public async Task PublishAsync<T>(string aggregateId, T @event)
    {
        if (!_topicMap.TryGetValue(typeof(T), out var topic))
            throw new InvalidOperationException($"No topic mapped for {typeof(T).Name}");

        var envelope = new EventEnvelope<T>
        {
            Data   = @event,
            Source = "order-service",
        };

        await _producer.ProduceAsync(topic, new Message<string, string>
        {
            Key   = aggregateId, // Ensures ordering per aggregate
            Value = JsonSerializer.Serialize(envelope),
        });
    }
}

// Partition key strategies (critical for ordering guarantees)
public static class PartitionKeyStrategies
{
    // Order events: use OrderId → all order events in order (same partition)
    public static string ForOrder(string orderId) => orderId;

    // User events: use UserId → all user events ordered per user
    public static string ForUser(string userId) => userId;

    // Null key: round-robin across partitions (no ordering needed, max throughput)
    public static string? RoundRobin() => null;

    // Geographic key: route by region for geo-affinity
    public static string ForRegion(string region, string entityId) => $"{region}:{entityId}";
}

#endregion

// ============================================================================
// SECTION 10: SAGA PATTERN OVER KAFKA
// ============================================================================

#region Saga Pattern

/*
 * SAGA PATTERN:
 *   Long-running business transactions that span multiple microservices.
 *   Each step either succeeds or triggers a compensating transaction.
 *
 * CHOREOGRAPHY-BASED SAGA (event-driven, no central coordinator):
 *   OrderService    → publishes OrderCreated
 *   PaymentService  → consumes OrderCreated, publishes PaymentProcessed or PaymentFailed
 *   InventoryService → consumes PaymentProcessed, publishes InventoryReserved or InventoryFailed
 *   ShippingService → consumes InventoryReserved, publishes OrderShipped
 *
 *   Compensation:
 *   If InventoryFailed → InventoryService publishes InventoryFailed
 *   PaymentService consumes InventoryFailed → publishes PaymentRefunded (compensating)
 *   OrderService consumes PaymentRefunded → updates order to Cancelled
 *
 * PITFALLS:
 *   - Hard to visualize/debug the full saga flow.
 *   - Risk of cyclic event chains.
 *   - Compensating transactions must be idempotent.
 *
 * ORCHESTRATION-BASED SAGA (central saga orchestrator):
 *   SagaOrchestrator sends commands (ProcessPayment) to each service's command topic.
 *   Receives reply events (PaymentProcessed / PaymentFailed).
 *   Maintains saga state machine (in DB or Redis).
 *   Easier to monitor/trace; orchestrator is a potential bottleneck.
 */

// Saga state machine for Order fulfillment
public enum OrderSagaState
{
    Started,
    PaymentPending,
    PaymentConfirmed,
    PaymentFailed,
    InventoryPending,
    InventoryReserved,
    InventoryFailed,
    ShippingPending,
    Completed,
    Compensating,
    Cancelled,
}

public class OrderSagaOrchestrator
{
    private readonly IEventPublisher _publisher;
    private readonly ISagaStateRepository _sagaRepo;
    private readonly ILogger<OrderSagaOrchestrator> _logger;

    public OrderSagaOrchestrator(IEventPublisher pub, ISagaStateRepository repo, ILogger<OrderSagaOrchestrator> logger)
    {
        _publisher = pub;
        _sagaRepo  = repo;
        _logger    = logger;
    }

    // Step 1: Start saga when order is created
    public async Task StartAsync(string orderId, string customerId, decimal amount)
    {
        var saga = new SagaInstance
        {
            SagaId     = Guid.NewGuid().ToString(),
            OrderId    = orderId,
            State      = OrderSagaState.PaymentPending,
            StartedAt  = DateTime.UtcNow,
        };
        await _sagaRepo.SaveAsync(saga);

        // Send command to payment service's command topic
        await _publisher.PublishAsync("payment.commands", orderId, new ProcessPaymentCommand
        {
            SagaId     = saga.SagaId,
            OrderId    = orderId,
            CustomerId = customerId,
            Amount     = amount,
        });
    }

    // Step 2: React to payment result
    public async Task HandlePaymentResultAsync(string sagaId, bool success, string? failureReason)
    {
        var saga = await _sagaRepo.GetAsync(sagaId)
            ?? throw new InvalidOperationException($"Saga {sagaId} not found");

        if (success)
        {
            saga.State = OrderSagaState.InventoryPending;
            await _sagaRepo.SaveAsync(saga);
            await _publisher.PublishAsync("inventory.commands", saga.OrderId, new ReserveInventoryCommand
            {
                SagaId  = sagaId,
                OrderId = saga.OrderId,
            });
        }
        else
        {
            saga.State = OrderSagaState.Cancelled;
            await _sagaRepo.SaveAsync(saga);
            _logger.LogWarning("Saga {SagaId} cancelled: payment failed — {Reason}", sagaId, failureReason);
            // No compensation needed at this step (payment wasn't taken)
            await _publisher.PublishAsync("order.commands", saga.OrderId, new CancelOrderCommand
            {
                SagaId  = sagaId,
                OrderId = saga.OrderId,
                Reason  = failureReason ?? "PaymentFailed",
            });
        }
    }
}

// Stub saga types
public class SagaInstance { public string SagaId = ""; public string OrderId = ""; public OrderSagaState State; public DateTime StartedAt; }
public interface ISagaStateRepository { Task SaveAsync(SagaInstance s); Task<SagaInstance?> GetAsync(string sagaId); }
public record ProcessPaymentCommand(string SagaId, string OrderId, string CustomerId, decimal Amount);
public record ReserveInventoryCommand(string SagaId, string OrderId);
public record CancelOrderCommand(string SagaId, string OrderId, string Reason);

#endregion

// ============================================================================
// SECTION 11: COMPACTED TOPICS & EVENT SOURCING
// ============================================================================

#region Compacted Topics & Event Sourcing

/*
 * LOG COMPACTION:
 *   - Kafka retains the LATEST record for each message key.
 *   - Older records with the same key are deleted during background compaction.
 *   - TOMBSTONE: a record with null value signals "delete this key from the log."
 *   - Use case: materializing the latest state of an entity (like a KV store).
 *   - Config: cleanup.policy=compact
 *   - min.cleanable.dirty.ratio: how full the log can get before compaction runs.
 *
 * COMPACTED TOPIC USE CASES:
 *   - User profile snapshots (latest profile per userId).
 *   - Feature flags per tenant.
 *   - Product catalog (latest product data per productId).
 *   - Configuration updates broadcast to all service instances.
 *
 * EVENT SOURCING + KAFKA:
 *   - All state changes are stored as an immutable event log (the Kafka topic IS the source of truth).
 *   - Consumers rebuild state by replaying from offset 0.
 *   - Snapshots: periodically snapshot current state to avoid replaying from scratch.
 *   - Separate read model (CQRS): project events into a read-optimized store (Elasticsearch, Redis, PG).
 *
 * INTERVIEW TIP:
 *   Q: "What's the difference between a compacted topic and event sourcing?"
 *   A: Compacted topic = only latest value per key (state store semantics).
 *      Event sourcing = full history of all events (audit log semantics).
 *      They serve different purposes; both use Kafka but with different cleanup policies.
 */

// Publishing a compacted "state snapshot" message
public class UserProfilePublisher
{
    private readonly IProducer<string, string> _producer;

    // Topic: cleanup.policy=compact
    private const string UserProfilesTopic = "users.profiles";

    public UserProfilePublisher(IProducer<string, string> producer) => _producer = producer;

    public async Task UpsertProfileAsync(UserProfile profile)
    {
        await _producer.ProduceAsync(UserProfilesTopic, new Message<string, string>
        {
            Key   = profile.UserId,   // Key = userId → compaction key
            Value = JsonSerializer.Serialize(profile),
        });
    }

    public async Task DeleteProfileAsync(string userId)
    {
        // Tombstone: null value signals compaction should delete this key
        await _producer.ProduceAsync(UserProfilesTopic, new Message<string, string>
        {
            Key   = userId,
            Value = null!,  // Tombstone
        });
    }
}

// In-memory cache built by consuming a compacted topic
public class UserProfileCache : BackgroundService
{
    private readonly Dictionary<string, UserProfile> _cache = new();
    private readonly ReaderWriterLockSlim _lock = new();

    public UserProfile? Get(string userId)
    {
        _lock.EnterReadLock();
        try { return _cache.TryGetValue(userId, out var p) ? p : null; }
        finally { _lock.ExitReadLock(); }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = "localhost:9092",
            GroupId          = $"user-profile-cache-{Environment.MachineName}",
            AutoOffsetReset  = AutoOffsetReset.Earliest, // Must start from beginning for compacted topic
            EnableAutoCommit = true,
        };

        using var consumer = new ConsumerBuilder<string, string>(config).Build();
        consumer.Subscribe("users.profiles");

        while (!stoppingToken.IsCancellationRequested)
        {
            var result = consumer.Consume(TimeSpan.FromMilliseconds(500));
            if (result is null || result.IsPartitionEOF) continue;

            _lock.EnterWriteLock();
            try
            {
                if (result.Message.Value is null)
                    _cache.Remove(result.Message.Key); // Tombstone = delete
                else
                    _cache[result.Message.Key] = JsonSerializer.Deserialize<UserProfile>(result.Message.Value)!;
            }
            finally { _lock.ExitWriteLock(); }
        }
    }
}

public record UserProfile(string UserId, string Email, string Name, DateTime UpdatedAt);

#endregion

// ============================================================================
// SECTION 12: OBSERVABILITY — LAG MONITORING & METRICS
// ============================================================================

#region Observability

/*
 * KEY KAFKA METRICS TO MONITOR IN PRODUCTION:
 *
 *   CONSUMER LAG:   High lag = consumers can't keep up.
 *                   Tool: Kafka's built-in consumer group describe, Burrow, Datadog, Prometheus JMX exporter.
 *                   Alert if lag grows continuously (not just spikes).
 *
 *   PRODUCE RATE:   Messages/sec produced per topic/partition.
 *   CONSUME RATE:   Messages/sec consumed.
 *   UNDER-REPLICATED PARTITIONS: > 0 means some replicas are falling behind (data at risk).
 *   ACTIVE CONTROLLER COUNT:     Must be 1. 0 = no controller (outage). 2+ = split brain.
 *   OFFLINE PARTITIONS:          Must be 0. Any offline = unavailability.
 *   REQUEST LATENCY:             ProduceRequestLatency, FetchRequestLatency.
 *
 * .NET METRICS INTEGRATION:
 *   - Parse rdkafka statistics JSON (SetStatisticsHandler on producer/consumer).
 *   - Expose via System.Diagnostics.Metrics / OpenTelemetry.
 *   - Scrape with Prometheus; alert in Grafana.
 *
 * DISTRIBUTED TRACING:
 *   - Propagate W3C Trace Context headers in Kafka message headers.
 *   - OpenTelemetry has Confluent.Kafka instrumentation: adds spans for produce/consume automatically.
 */

// Statistics handler: parse rdkafka JSON stats and emit as metrics
public class KafkaMetricsEmitter
{
    private static readonly System.Diagnostics.Metrics.Meter Meter = new("Kafka.Consumer", "1.0");
    private static readonly System.Diagnostics.Metrics.ObservableGauge<long> ConsumerLagGauge
        = Meter.CreateObservableGauge<long>("kafka.consumer.lag", () => GetCurrentLag());

    private static long _currentLag;

    // Called by SetStatisticsHandler on the consumer
    public static void OnStatistics(string statisticsJson)
    {
        try
        {
            // rdkafka stats: parse consumer_lag per partition
            using var doc = JsonDocument.Parse(statisticsJson);
            long totalLag = 0;

            if (doc.RootElement.TryGetProperty("topics", out var topics))
            {
                foreach (var topic in topics.EnumerateObject())
                {
                    if (topic.Value.TryGetProperty("partitions", out var partitions))
                    {
                        foreach (var partition in partitions.EnumerateObject())
                        {
                            if (partition.Value.TryGetProperty("consumer_lag", out var lag))
                            {
                                var lagValue = lag.GetInt64();
                                if (lagValue >= 0) totalLag += lagValue;
                            }
                        }
                    }
                }
            }

            Interlocked.Exchange(ref _currentLag, totalLag);
        }
        catch { /* Don't let metrics parsing crash the consumer */ }
    }

    private static long GetCurrentLag() => Interlocked.Read(ref _currentLag);
}

// Distributed tracing: inject/extract trace context in Kafka headers
public static class KafkaTraceContext
{
    private const string TraceParentHeader = "traceparent";
    private const string TraceStateHeader  = "tracestate";

    // Producer: inject current trace context into message headers
    public static void InjectTraceContext(Headers headers)
    {
        var activity = Activity.Current;
        if (activity is null) return;

        var traceParent = $"00-{activity.TraceId}-{activity.SpanId}-01";
        headers.Add(TraceParentHeader, Encoding.UTF8.GetBytes(traceParent));
        if (activity.TraceStateString is not null)
            headers.Add(TraceStateHeader, Encoding.UTF8.GetBytes(activity.TraceStateString));
    }

    // Consumer: extract and restore trace context from message headers
    public static Activity? ExtractTraceContext(Headers? headers, string operationName)
    {
        if (headers is null) return null;

        var traceParentHeader = headers.FirstOrDefault(h => h.Key == TraceParentHeader);
        if (traceParentHeader is null) return null;

        var traceParent = Encoding.UTF8.GetString(traceParentHeader.GetValueBytes());
        // Parse W3C traceparent: 00-{traceId}-{parentSpanId}-{flags}
        var parts = traceParent.Split('-');
        if (parts.Length < 4) return null;

        var activityContext = new ActivityContext(
            ActivityTraceId.CreateFromString(parts[1]),
            ActivitySpanId.CreateFromString(parts[2]),
            ActivityTraceFlags.Recorded);

        return new ActivitySource("Kafka.Consumer")
            .StartActivity(operationName, ActivityKind.Consumer, activityContext);
    }
}

#endregion

// ============================================================================
// SECTION 13: SECURITY — TLS, SASL & ACLs
// ============================================================================

#region Security

/*
 * KAFKA SECURITY LAYERS:
 *
 *   ENCRYPTION (TLS):
 *     - Encrypts data in transit between clients and brokers.
 *     - security.protocol=SSL or SASL_SSL.
 *     - Requires: CA certificate, client certificate (optional for mTLS).
 *
 *   AUTHENTICATION (SASL):
 *     - SASL/PLAIN:     Username/password. Simple but requires TLS to avoid plaintext creds.
 *     - SASL/SCRAM:     Challenge-response (SCRAM-SHA-256/512). Safer than PLAIN.
 *     - SASL/OAUTHBEARER: OAuth 2.0 token-based (Confluent Cloud uses this).
 *     - SASL/GSSAPI:    Kerberos. Common in enterprise environments.
 *
 *   AUTHORIZATION (ACLs):
 *     - Kafka ACLs: allow/deny per principal (user/service account) per resource (topic/group).
 *     - Operations: READ, WRITE, CREATE, DELETE, DESCRIBE, ALTER.
 *     - Principle of least privilege: each service only has READ/WRITE on its own topics.
 *
 * INTERVIEW TIP:
 *   Q: "How do you secure Kafka in production on Kubernetes?"
 *   A: mTLS for encryption + authentication, SASL/SCRAM or OAUTHBEARER for identity,
 *      ACLs per service account, secrets via Vault or K8s Secrets (encrypted at rest),
 *      network policies to restrict broker access to app namespaces only.
 */

public class SecureKafkaConfigFactory
{
    // TLS + SASL/SCRAM (typical on-premise production)
    public static ConsumerConfig CreateScramConsumerConfig(
        string bootstrapServers,
        string username,
        string password,
        string caCertPath)
    {
        return new ConsumerConfig
        {
            BootstrapServers         = bootstrapServers,
            SecurityProtocol         = SecurityProtocol.SaslSsl,
            SaslMechanism            = SaslMechanism.ScramSha256,
            SaslUsername             = username,
            SaslPassword             = password,
            SslCaLocation            = caCertPath,
            // For mTLS (client cert auth):
            // SslCertificateLocation = "/certs/client-cert.pem",
            // SslKeyLocation         = "/certs/client-key.pem",
            SslEndpointIdentificationAlgorithm = SslEndpointIdentificationAlgorithm.Https,
            GroupId          = "order-service",
            EnableAutoCommit = false,
            AutoOffsetReset  = AutoOffsetReset.Earliest,
        };
    }

    // OAuth Bearer (Confluent Cloud / MSK IAM)
    public static ProducerConfig CreateOAuthProducerConfig(string bootstrapServers)
    {
        return new ProducerConfig
        {
            BootstrapServers = bootstrapServers,
            SecurityProtocol = SecurityProtocol.SaslSsl,
            SaslMechanism    = SaslMechanism.OAuthBearer,
            // Token refresh callback is set on the builder:
            // .SetOAuthBearerTokenRefreshHandler((client, cfg) => {
            //     var token = FetchTokenFromIdentityProvider();
            //     client.OAuthBearerSetToken(token.AccessToken, token.ExpiresAtMs, token.Principal);
            // })
            Acks = Acks.All,
        };
    }
}

#endregion

// ============================================================================
// SECTION 14: HIGH-THROUGHPUT TUNING & BACKPRESSURE
// ============================================================================

#region High Throughput & Backpressure

/*
 * PRODUCER THROUGHPUT TUNING:
 *   linger.ms ↑         → larger batches, higher latency, higher throughput
 *   batch.size ↑        → more messages per batch
 *   compression.type    → lz4 for speed, zstd for max ratio
 *   buffer.memory ↑     → more in-flight data buffered
 *   max.in.flight ↑     → more parallel requests (up to 5 with idempotence)
 *
 * CONSUMER THROUGHPUT TUNING:
 *   fetch.min.bytes ↑   → wait for more data before returning (reduces round trips)
 *   fetch.wait.max.ms ↑ → controls fetch latency
 *   max.partition.fetch.bytes ↑ → more data per partition per fetch
 *   Parallel processing: use Channels or Parallel.ForEachAsync within consumer loop
 *
 * BACKPRESSURE IN .NET CONSUMERS:
 *   - Channel<T> (System.Threading.Channels) between consumer thread and processing workers.
 *   - Consumer writes to Channel (BoundedChannel with capacity cap).
 *   - When channel is full, consumer blocks → natural backpressure.
 *   - Multiple worker Tasks drain the channel concurrently.
 */

public class HighThroughputConsumerService : BackgroundService
{
    private readonly ILogger<HighThroughputConsumerService> _logger;
    // Bounded channel = backpressure mechanism
    private readonly System.Threading.Channels.Channel<ConsumeResult<string, string>> _channel
        = System.Threading.Channels.Channel.CreateBounded<ConsumeResult<string, string>>(
            new System.Threading.Channels.BoundedChannelOptions(capacity: 1000)
            {
                FullMode         = System.Threading.Channels.BoundedChannelFullMode.Wait,
                SingleReader     = false,
                SingleWriter     = true,
            });

    private const int WorkerCount = 8;

    public HighThroughputConsumerService(ILogger<HighThroughputConsumerService> logger)
        => _logger = logger;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Start N worker tasks that drain the channel concurrently
        var workers = Enumerable.Range(0, WorkerCount)
            .Select(_ => Task.Run(() => ProcessChannelAsync(stoppingToken), stoppingToken))
            .ToArray();

        // Single consumer thread → single partition fetch, writes to channel
        await ConsumeToChannelAsync(stoppingToken);

        _channel.Writer.Complete();
        await Task.WhenAll(workers);
    }

    private async Task ConsumeToChannelAsync(CancellationToken ct)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers    = "localhost:9092",
            GroupId             = "high-throughput-group",
            EnableAutoCommit    = false,
            AutoOffsetReset     = AutoOffsetReset.Earliest,
            FetchMinBytes       = 65536,    // 64KB — wait to accumulate more data
            FetchWaitMaxMs      = 100,
            MaxPartitionFetchBytes = 10_485_760, // 10MB per partition
        };

        using var consumer = new ConsumerBuilder<string, string>(config).Build();
        consumer.Subscribe("order.order.created");

        while (!ct.IsCancellationRequested)
        {
            var result = consumer.Consume(TimeSpan.FromMilliseconds(50));
            if (result is null || result.IsPartitionEOF) continue;

            // WriteAsync blocks if channel is full → backpressure on consumer
            await _channel.Writer.WriteAsync(result, ct);
        }

        consumer.Close();
    }

    private async Task ProcessChannelAsync(CancellationToken ct)
    {
        await foreach (var result in _channel.Reader.ReadAllAsync(ct))
        {
            try
            {
                // Process concurrently — IMPORTANT: per-partition ordering NOT guaranteed here
                // Use this only when message ordering doesn't matter
                await ProcessAsync(result.Message.Key, result.Message.Value);
                _logger.LogDebug("Processed {Key}", result.Message.Key);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Worker error on {Key}", result.Message.Key);
            }
        }
    }

    private Task ProcessAsync(string key, string value) => Task.CompletedTask;
}

#endregion

// ============================================================================
// SECTION 15: DI REGISTRATION — CLEAN MICROSERVICE SETUP
// ============================================================================

#region Dependency Injection & Service Registration

public static class KafkaServiceExtensions
{
    // Clean extension method for registering all Kafka services
    public static IServiceCollection AddKafkaMessaging(
        this IServiceCollection services,
        Action<KafkaOptions> configure)
    {
        services.Configure<KafkaOptions>(configure);

        // Producer (singleton — one producer per application, thread-safe)
        services.AddSingleton<IEventPublisher, KafkaEventPublisher>();

        // Topic manager for startup checks
        services.AddSingleton<KafkaTopicManager>(sp =>
        {
            var opts   = sp.GetRequiredService<IOptions<KafkaOptions>>().Value;
            var logger = sp.GetRequiredService<ILogger<KafkaTopicManager>>();
            return new KafkaTopicManager(opts.BootstrapServers, logger);
        });

        // Consumer as a hosted service (runs in background)
        services.AddHostedService<KafkaConsumerService>();

        // Outbox relay
        services.AddHostedService<OutboxRelayService>();

        return services;
    }
}

public class KafkaOptions
{
    public string BootstrapServers { get; set; } = "localhost:9092";
    public string GroupId          { get; set; } = "default-group";
    public string SchemaRegistryUrl { get; set; } = "http://localhost:8081";
    public List<string> Topics     { get; set; } = new();
    public SaslOptions? Sasl       { get; set; }
}

public class SaslOptions
{
    public string Mechanism { get; set; } = "SCRAM-SHA-256";
    public string Username  { get; set; } = default!;
    public string Password  { get; set; } = default!;
    public string CaCertPath { get; set; } = default!;
}

// Startup: ensure topics exist before accepting traffic
public class KafkaStartupService : IHostedService
{
    private readonly KafkaTopicManager _topicManager;
    private readonly KafkaOptions _options;

    public KafkaStartupService(KafkaTopicManager tm, IOptions<KafkaOptions> opts)
    {
        _topicManager = tm;
        _options      = opts.Value;
    }

    public async Task StartAsync(CancellationToken ct)
    {
        foreach (var topic in _options.Topics)
            await _topicManager.EnsureTopicExistsAsync(topic);
    }

    public Task StopAsync(CancellationToken ct) => Task.CompletedTask;
}

// Program.cs wiring (ASP.NET Core 8)
public class ProgramExample
{
    public static void ConfigureServices(IServiceCollection services, Microsoft.Extensions.Configuration.IConfiguration config)
    {
        services.AddKafkaMessaging(opt =>
        {
            opt.BootstrapServers   = config["Kafka:BootstrapServers"] ?? "localhost:9092";
            opt.GroupId            = config["Kafka:GroupId"] ?? "order-service";
            opt.SchemaRegistryUrl  = config["Kafka:SchemaRegistryUrl"] ?? "http://localhost:8081";
            opt.Topics             = new List<string>
            {
                KafkaCoreConceptsDemo.OrderCreatedTopic,
                KafkaCoreConceptsDemo.PaymentProcessedTopic,
            };
        });
    }
}

#endregion

// ============================================================================
// SECTION 16: INTERVIEW Q&A QUICK REFERENCE
// ============================================================================

#region Interview Q&A

/*
 * ============================================================================
 * KAFKA INTERVIEW QUESTIONS — QUICK REFERENCE (8 YOE LEVEL)
 * ============================================================================
 *
 * CORE CONCEPTS
 * -------------
 * Q1: What is the role of a partition in Kafka?
 *     - Unit of parallelism, ordering, and storage.
 *     - Each partition has one leader + N-1 followers.
 *     - Messages with the same key go to the same partition (sticky hashing by default).
 *     - Cannot reduce partition count after creation.
 *
 * Q2: What is an ISR and why does it matter?
 *     - In-Sync Replica: follower caught up within replica.lag.time.max.ms.
 *     - acks=all waits for all ISR to ack → data durability.
 *     - min.insync.replicas=2: at least 2 replicas must ack; prevents acking with only leader.
 *     - If ISR < min.insync.replicas with acks=all → producer gets NotEnoughReplicas error.
 *
 * Q3: Difference between acks=0, acks=1, acks=all?
 *     - acks=0: no ack, fire and forget, possible data loss.
 *     - acks=1: leader ack only, data loss if leader crashes before replication.
 *     - acks=all: all ISR ack, strongest durability. Always use in production.
 *
 * Q4: How does Kafka guarantee ordering?
 *     - Ordering guaranteed WITHIN a partition.
 *     - Cross-partition: no ordering guarantee.
 *     - To order all events for an entity: use entity ID as message key.
 *     - max.in.flight.per.connection=1 enforces strict ordering but hurts throughput.
 *     - With idempotence enabled, max.in.flight can be 5 with ordering preserved.
 *
 * Q5: What happens when a consumer crashes mid-processing?
 *     - If auto-commit: offset was already committed → message LOST.
 *     - If manual commit (commit after processing): offset NOT committed → at-least-once on restart.
 *     - Never auto-commit in critical services.
 *
 * CONSUMER GROUPS & REBALANCING
 * ------------------------------
 * Q6: What triggers a consumer group rebalance?
 *     - Consumer joins or leaves the group.
 *     - Heartbeat not received within session.timeout.ms.
 *     - poll() not called within max.poll.interval.ms (slow processing).
 *     - Subscription change.
 *
 * Q7: Eager vs cooperative rebalance?
 *     - Eager: all partitions revoked from all consumers, then reassigned (full stop).
 *     - Cooperative (Incremental Sticky): only affected partitions revoked/reassigned.
 *     - Use PartitionAssignmentStrategy.CooperativeSticky in production.
 *
 * Q8: Can two consumers in the same group read the same partition?
 *     - NO. Each partition assigned to exactly one consumer per group.
 *     - Extra consumers beyond partition count sit idle.
 *     - If you need fan-out: use a different consumer group.
 *
 * EXACTLY-ONCE & TRANSACTIONS
 * ----------------------------
 * Q9: How do you achieve exactly-once in Kafka?
 *     - Producer: enable.idempotence=true + transactional.id set + acks=all.
 *     - Consumer: isolation.level=read_committed.
 *     - ReadProcessWrite: SendOffsetsToTransaction() inside producer transaction.
 *     - This guarantees exactly-once at Kafka level; DB side effects need idempotency keys.
 *
 * Q10: What is zombie fencing?
 *      - When a producer with the same transactional.id restarts, the broker increments
 *        the epoch, fencing (rejecting) the old zombie instance's writes.
 *      - Critical in Kubernetes where pods restart with the same config.
 *
 * PERFORMANCE & TUNING
 * ---------------------
 * Q11: How do you increase producer throughput?
 *      - linger.ms > 0 to batch more messages.
 *      - compression.type=lz4 or zstd.
 *      - max.in.flight=5 (with idempotence).
 *      - buffer.memory increase.
 *      - Async fire-and-forget (Produce() not ProduceAsync() for max throughput).
 *
 * Q12: Consumer is lagging — what do you check first?
 *      - Partition count vs consumer count (more consumers won't help if partitions are fewer).
 *      - max.poll.interval.ms too low → rebalances causing pauses.
 *      - Slow downstream (DB, external API) — add parallel processing via Channels.
 *      - GC pauses in the .NET app triggering heartbeat timeouts.
 *      - fetch.min.bytes / fetch.wait.max.ms tuning for throughput.
 *
 * Q13: What is consumer lag and how do you monitor it?
 *      - Lag = (end offset) - (last committed offset) per partition.
 *      - Monitor with: kafka-consumer-groups.sh --describe, Burrow, Datadog, Prometheus.
 *      - Alert on continuously growing lag (not spikes); growing lag = consumer can't keep up.
 *
 * MICROSERVICES PATTERNS
 * -----------------------
 * Q14: What is the Outbox pattern and why use it?
 *      - Solves dual-write: save to DB AND publish to Kafka atomically.
 *      - Write OutboxMessage row in same DB transaction as domain entity.
 *      - Background relay reads OutboxMessage and publishes; marks as processed.
 *      - Guarantees at-least-once; consumers must be idempotent.
 *
 * Q15: Choreography vs Orchestration saga — when to use each?
 *      - Choreography: simpler, more decoupled, harder to trace complex flows.
 *        Use for simple 2-3 step sagas with clear event flows.
 *      - Orchestration: central state machine, easier observability, testable.
 *        Use for complex multi-step sagas where traceability is critical.
 *
 * Q16: How do you handle a "poison pill" message?
 *      - Catch deserialization errors and send to DLQ with original payload + error headers.
 *      - Do NOT block the consumer; commit the offset and move on.
 *      - DLQ consumer re-processes with human intervention or alternate logic.
 *
 * Q17: What is a compacted topic and when do you use it?
 *      - cleanup.policy=compact: Kafka keeps only the latest record per key.
 *      - Use for: state snapshots (user profiles, product catalog, config).
 *      - Tombstone (null value) = delete the key from the compacted log.
 *      - NOT for event history — use delete policy for full event logs.
 *
 * SCHEMA & OPERATIONS
 * --------------------
 * Q18: How do you evolve a Kafka message schema without breaking consumers?
 *      - Use Schema Registry with compatibility mode (BACKWARD preferred).
 *      - BACKWARD: new schema can read old data → consumers upgrade first.
 *      - Add fields with default values; never remove/rename required fields.
 *      - Version the schema; consumers should handle missing optional fields gracefully.
 *
 * Q19: How do you replay all events from a Kafka topic?
 *      - Reset consumer group offset to beginning: AdminClient.AlterConsumerGroupOffsetsAsync().
 *      - Or: new consumer group (different group.id) + AutoOffsetReset.Earliest.
 *      - For time-based replay: offsetsForTimes() to seek to a specific timestamp.
 *      - Ensure downstream is idempotent before replaying (duplicate events will be produced).
 *
 * Q20: What is KRaft and why does it matter?
 *      - Kafka Raft: replaces ZooKeeper for metadata management (Kafka 3.3+ production-ready).
 *      - Benefits: simpler ops (no ZK cluster), faster controller failover, higher partition limits.
 *      - In Kafka 3.x, ZooKeeper mode is deprecated; target KRaft for new deployments.
 *
 * ============================================================================
 * KEY NUMBERS TO REMEMBER
 * ============================================================================
 *
 *   Default retention:            7 days (retention.ms=604800000)
 *   Default partition count:      1 (always override to 12+ in production)
 *   Default replication factor:   1 (always 3 in production)
 *   max.in.flight with idempotence: 5 (not 1 — common misconception)
 *   Minimum partition count rule: partitions >= max consumer instances
 *   Session timeout default:      45s (Kafka ≥ 3.0)
 *   max.poll.interval.ms default: 5 minutes
 *   ZooKeeper deprecated:         Kafka 3.x (KRaft is the future)
 *
 * ============================================================================
 * COMMON MISCONCEPTIONS (INTERVIEW TRAPS)
 * ============================================================================
 *
 *   WRONG: "Kafka guarantees global ordering across all partitions."
 *   RIGHT: Kafka guarantees ordering WITHIN a partition only.
 *
 *   WRONG: "enable.idempotence=true requires max.in.flight=1."
 *   RIGHT: max.in.flight can be up to 5 with idempotence + strict ordering preserved.
 *
 *   WRONG: "acks=1 is safe enough for production."
 *   RIGHT: acks=1 can lose data if the leader crashes before replication. Use acks=all.
 *
 *   WRONG: "Auto-commit is fine for at-least-once delivery."
 *   RIGHT: Auto-commit commits after polling, not after processing → message loss on crash.
 *
 *   WRONG: "Adding more consumers always increases throughput."
 *   RIGHT: Consumers > partitions means extra consumers are idle. Increase partitions first.
 *
 *   WRONG: "Exactly-once means my DB side effects happen exactly once."
 *   RIGHT: Kafka EOS is at the Kafka level. DB side effects need idempotency keys separately.
 *
 *   WRONG: "Kafka is a message queue like RabbitMQ."
 *   RIGHT: Kafka is an immutable, distributed log. Messages persist and can be replayed.
 *          RabbitMQ: messages deleted on ack. Kafka: messages retained until retention expires.
 *
 *   WRONG: "You can reduce partition count on a topic."
 *   RIGHT: You can only ADD partitions, never reduce. Reduction requires topic recreation + migration.
 *
 * ============================================================================
 * Good luck with your interview!
 * ============================================================================
 */

#endregion
