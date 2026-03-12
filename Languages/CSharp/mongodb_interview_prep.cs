// ============================================================================
// MONGODB + .NET INTERVIEW PREPARATION — COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience | Microservices & .NET Focus
// ============================================================================
//
// TOPIC COVERAGE:
//   1.  MongoDB Core Concepts (Documents, Collections, BSON, ObjectId)
//   2.  Driver Setup & Dependency Injection in ASP.NET Core
//   3.  CRUD Operations — Full .NET Driver API
//   4.  Querying — Filters, Projections, Sorting, Pagination
//   5.  Aggregation Pipeline
//   6.  Indexes — Types, Strategy, Explain Plans
//   7.  Schema Design — Embedding vs Referencing, Patterns
//   8.  Transactions & Multi-Document ACID
//   9.  Change Streams — Real-Time Event Watching
//  10.  GridFS — Large File Storage
//  11.  Replica Sets & Write/Read Concerns
//  12.  Sharding — Concepts & Shard Key Strategy
//  13.  Atlas Search & Full-Text Search
//  14.  Repository Pattern & Clean Architecture
//  15.  Performance Tuning & Anti-Patterns
//  16.  Security — Auth, TLS, Field-Level Encryption
//  17.  Interview Q&A Quick Reference
//
// NuGet packages:
//   MongoDB.Driver                 (core driver)
//   MongoDB.Driver.GridFS          (large file storage)
//   MongoDB.Bson                   (BSON serialization)
//   Microsoft.Extensions.Options   (config binding)
//   Microsoft.Extensions.DependencyInjection
// ============================================================================

using MongoDB.Bson;
using MongoDB.Bson.Serialization;
using MongoDB.Bson.Serialization.Attributes;
using MongoDB.Bson.Serialization.Conventions;
using MongoDB.Driver;
using MongoDB.Driver.Core.Events;
using MongoDB.Driver.GridFS;
using MongoDB.Driver.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

// ============================================================================
// SECTION 1: CORE CONCEPTS
// ============================================================================

#region Core Concepts (Theory)

/*
 * MONGODB FUNDAMENTALS
 * =====================
 *
 * DOCUMENT:
 *   - JSON-like (stored as BSON internally).
 *   - Schema-flexible: documents in the same collection CAN have different shapes.
 *   - Max document size: 16MB. Use GridFS for larger data.
 *   - Nested documents (sub-documents) and arrays are first-class citizens.
 *
 * COLLECTION:
 *   - Analogous to a SQL table, but schema-free.
 *   - Created implicitly on first insert (or explicitly with CreateCollection).
 *   - Can enforce schema validation with JSON Schema rules at collection level.
 *
 * BSON (Binary JSON):
 *   - MongoDB's wire + storage format.
 *   - Adds types not in JSON: ObjectId, Date, Decimal128, Binary, Regex, etc.
 *   - Faster to scan than JSON (length-prefixed fields).
 *
 * OBJECTID:
 *   - 12-byte default _id.
 *   - Layout: [4B timestamp][5B random machine+process][3B incrementing counter].
 *   - Monotonically increasing within a second → roughly sortable by insertion time.
 *   - Generated client-side by default (no round-trip for ID generation).
 *
 * BSON vs JSON vs EJSON:
 *   - BSON: binary, internal/wire format.
 *   - JSON: human-readable, no type metadata.
 *   - EJSON (Extended JSON): JSON with type annotations ($oid, $date) for round-tripping BSON types.
 *
 * WiredTiger STORAGE ENGINE (default since 3.2):
 *   - Document-level locking (not collection or database level).
 *   - Compression at rest (snappy by default, zstd available).
 *   - MVCC for read isolation (readers don't block writers).
 *   - Write-ahead log (journal) for durability.
 *
 * INTERVIEW TIP:
 *   "MongoDB is not a replacement for a relational DB — it excels at:
 *    hierarchical/nested data, flexible schemas during rapid iteration,
 *    horizontal scale-out (sharding), and high write throughput.
 *    It struggles with: complex multi-collection joins, strict relational integrity,
 *    and ad-hoc analytical queries (use Atlas Analytics or export to a warehouse)."
 */

#endregion

// ============================================================================
// SECTION 2: DOMAIN MODELS & BSON SERIALIZATION
// ============================================================================

#region Domain Models & BSON Attributes

/*
 * BSON SERIALIZATION ATTRIBUTES:
 *   [BsonId]              → maps property to _id field
 *   [BsonRepresentation]  → controls how a .NET type is stored in BSON
 *   [BsonElement("name")] → override field name in BSON (camelCase convention)
 *   [BsonIgnore]          → exclude field from serialization
 *   [BsonIgnoreIfNull]    → don't write field if null (reduces document size)
 *   [BsonDateTimeOptions] → control timezone handling for DateTime
 *   [BsonRequired]        → throw on deserialization if field is missing
 *   [BsonDefaultValue]    → use this value if field is absent
 *
 * CONVENTION PACKS (preferred over per-property attributes):
 *   Register once at startup to apply rules globally.
 */

// ── Order aggregate root ──────────────────────────────────────────────────────
public class Order
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; } = ObjectId.GenerateNewId().ToString();

    // Embedded customer snapshot — avoids join, preserves state at time of order
    public CustomerSnapshot Customer { get; set; } = default!;

    // Array of embedded sub-documents
    public List<OrderLine> Lines { get; set; } = new();

    public OrderStatus Status { get; set; } = OrderStatus.Pending;

    public MoneyAmount Total { get; set; } = default!;

    [BsonDateTimeOptions(Kind = DateTimeKind.Utc)]
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    [BsonDateTimeOptions(Kind = DateTimeKind.Utc)]
    [BsonIgnoreIfNull]
    public DateTime? ShippedAt { get; set; }

    // Metadata bag for extensibility without schema change
    [BsonIgnoreIfNull]
    public BsonDocument? Metadata { get; set; }

    // Version field for optimistic concurrency
    [BsonElement("__v")]
    public int Version { get; set; }
}

public class CustomerSnapshot
{
    public string CustomerId { get; set; } = default!;
    public string FullName   { get; set; } = default!;
    public string Email      { get; set; } = default!;
    // Snapshot of address at time of order (not a reference)
    public Address ShippingAddress { get; set; } = default!;
}

public class OrderLine
{
    public string ProductId   { get; set; } = default!;
    public string ProductName { get; set; } = default!; // Snapshot
    public int    Quantity    { get; set; }
    public MoneyAmount UnitPrice { get; set; } = default!;
}

public class Address
{
    public string Street  { get; set; } = default!;
    public string City    { get; set; } = default!;
    public string Country { get; set; } = default!;
    public string ZipCode { get; set; } = default!;
}

public record MoneyAmount(decimal Amount, string Currency = "USD");

public enum OrderStatus { Pending, Confirmed, Shipped, Delivered, Cancelled }

// ── Product document ──────────────────────────────────────────────────────────
public class Product
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; } = ObjectId.GenerateNewId().ToString();

    public string   Name        { get; set; } = default!;
    public string   Sku         { get; set; } = default!;
    public decimal  Price       { get; set; }
    public string   Currency    { get; set; } = "USD";
    public int      Stock       { get; set; }
    public string[] Tags        { get; set; } = Array.Empty<string>();
    public string   Category    { get; set; } = default!;
    public bool     IsActive    { get; set; } = true;

    [BsonDateTimeOptions(Kind = DateTimeKind.Utc)]
    public DateTime UpdatedAt   { get; set; } = DateTime.UtcNow;
}

// ── BSON Convention Pack registration (call once in Program.cs) ──────────────
public static class BsonConventions
{
    public static void Register()
    {
        var pack = new ConventionPack
        {
            new CamelCaseElementNameConvention(),    // "MyProp" → "myProp" in BSON
            new IgnoreExtraElementsConvention(true), // Don't throw on unknown fields (schema evolution)
            new IgnoreIfNullConvention(true),        // Don't store null fields
            new EnumRepresentationConvention(BsonType.String), // Enums as strings (readable + safer)
        };
        ConventionRegistry.Register("AppConventions", pack, _ => true);
    }
}

#endregion

// ============================================================================
// SECTION 3: DRIVER SETUP & DEPENDENCY INJECTION
// ============================================================================

#region Driver Setup & DI

/*
 * MONGOCLIENT IS THREAD-SAFE AND SHOULD BE A SINGLETON:
 *   - Maintains an internal connection pool (default: 100 connections per server).
 *   - Creating a new MongoClient per request is a critical anti-pattern —
 *     exhausts connections and destroys performance.
 *
 * MONGOURL vs CONNECTION STRING:
 *   - mongodb://user:pass@host:27017/dbname?authSource=admin
 *   - mongodb+srv://... for Atlas DNS SRV lookup (auto-discovers replica set members).
 *
 * CONNECTION POOL SETTINGS:
 *   - MaxConnectionPoolSize: max open connections (default 100). Tune to workload.
 *   - MinConnectionPoolSize: keep-alive connections (0 by default).
 *   - WaitQueueTimeout: how long a request waits for a pool connection.
 *   - ConnectTimeout, ServerSelectionTimeout: cluster discovery timeouts.
 *
 * COMMAND MONITORING:
 *   - Hook into driver events for logging slow queries or emitting metrics.
 */

public class MongoDbOptions
{
    public string ConnectionString { get; set; } = "mongodb://localhost:27017";
    public string DatabaseName     { get; set; } = "shop";
    public int    MaxPoolSize      { get; set; } = 100;
    public int    MinPoolSize      { get; set; } = 5;
    public int    CommandTimeoutMs { get; set; } = 30_000;
}

public static class MongoDbServiceExtensions
{
    public static IServiceCollection AddMongoDb(
        this IServiceCollection services,
        Action<MongoDbOptions> configure)
    {
        services.Configure<MongoDbOptions>(configure);

        // Singleton MongoClient — one per application lifetime
        services.AddSingleton<IMongoClient>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<MongoDbOptions>>().Value;
            var logger = sp.GetRequiredService<ILogger<IMongoClient>>();

            var settings = MongoClientSettings.FromConnectionString(opts.ConnectionString);

            // Connection pool tuning
            settings.MaxConnectionPoolSize = opts.MaxPoolSize;
            settings.MinConnectionPoolSize = opts.MinPoolSize;
            settings.ServerSelectionTimeout = TimeSpan.FromSeconds(10);
            settings.ConnectTimeout         = TimeSpan.FromSeconds(10);
            settings.SocketTimeout          = TimeSpan.FromMilliseconds(opts.CommandTimeoutMs);

            // Command monitoring for slow query logging
            settings.ClusterConfigurator = cb =>
            {
                cb.Subscribe<CommandSucceededEvent>(e =>
                {
                    if (e.Duration > TimeSpan.FromMilliseconds(200))
                        logger.LogWarning("Slow MongoDB command [{Command}] took {Ms}ms",
                            e.CommandName, e.Duration.TotalMilliseconds);
                });
                cb.Subscribe<CommandFailedEvent>(e =>
                    logger.LogError(e.Failure, "MongoDB command [{Command}] failed", e.CommandName));
            };

            // Uncomment for Atlas with X.509 / TLS:
            // settings.SslSettings = new SslSettings { ... };

            return new MongoClient(settings);
        });

        // Scoped database — cheap, safe to scope per request
        services.AddScoped<IMongoDatabase>(sp =>
        {
            var client = sp.GetRequiredService<IMongoClient>();
            var opts   = sp.GetRequiredService<IOptions<MongoDbOptions>>().Value;
            return client.GetDatabase(opts.DatabaseName);
        });

        // Register repositories
        services.AddScoped<IOrderRepository, MongoOrderRepository>();
        services.AddScoped<IProductRepository, MongoProductRepository>();

        return services;
    }
}

#endregion

// ============================================================================
// SECTION 4: CRUD OPERATIONS
// ============================================================================

#region CRUD Operations

/*
 * WRITE OPERATIONS — KEY CONCEPTS:
 *
 *   InsertOne / InsertMany:
 *     - InsertMany with ordered=false: continues on individual failures (faster for bulk).
 *     - ordered=true (default): stops on first failure.
 *
 *   ReplaceOne:       Replaces the ENTIRE document (except _id). Like SQL UPDATE all columns.
 *   UpdateOne/Many:   Updates specific fields using update operators ($set, $inc, $push, etc.).
 *   FindOneAndUpdate: Atomically finds, updates, and returns the document (old or new).
 *
 *   Upsert:           update with IsUpsert=true → inserts if not found.
 *
 * UPDATE OPERATORS:
 *   $set      → set field value
 *   $unset    → remove a field
 *   $inc      → increment numeric field
 *   $push     → append to array
 *   $addToSet → append to array only if not already present
 *   $pull     → remove from array matching condition
 *   $currentDate → set to current date/time
 *
 * INTERVIEW TRAP:
 *   Q: "What's the difference between UpdateOne and ReplaceOne?"
 *   A: UpdateOne applies an update operator patch ($set, $inc, etc.) to specific fields.
 *      ReplaceOne swaps the entire document body (the _id stays). Accidentally using
 *      ReplaceOne with an update operator will throw an error.
 */

public class MongoCrudExamples
{
    private readonly IMongoCollection<Order> _orders;
    private readonly IMongoCollection<Product> _products;

    public MongoCrudExamples(IMongoDatabase db)
    {
        _orders   = db.GetCollection<Order>("orders");
        _products = db.GetCollection<Product>("products");
    }

    // ── INSERT ────────────────────────────────────────────────────────────────

    public async Task<string> InsertOrderAsync(Order order)
    {
        await _orders.InsertOneAsync(order);
        return order.Id; // Driver populates _id in-place
    }

    public async Task BulkInsertProductsAsync(IEnumerable<Product> products)
    {
        // ordered: false → don't stop on first failure; duplicates are skipped
        await _products.InsertManyAsync(products, new InsertManyOptions { IsOrdered = false });
    }

    // ── READ ──────────────────────────────────────────────────────────────────

    public async Task<Order?> GetOrderByIdAsync(string id)
    {
        // Strongly-typed filter using Builders<T>
        var filter = Builders<Order>.Filter.Eq(o => o.Id, id);
        return await _orders.Find(filter).FirstOrDefaultAsync();
    }

    public async Task<List<Order>> GetOrdersByStatusAsync(OrderStatus status)
    {
        return await _orders
            .Find(o => o.Status == status)  // Lambda filter (compiled to BSON)
            .SortByDescending(o => o.CreatedAt)
            .ToListAsync();
    }

    // ── UPDATE ────────────────────────────────────────────────────────────────

    // UpdateOne with $set — only touches specified fields
    public async Task UpdateOrderStatusAsync(string orderId, OrderStatus newStatus)
    {
        var filter = Builders<Order>.Filter.Eq(o => o.Id, orderId);
        var update = Builders<Order>.Update
            .Set(o => o.Status, newStatus)
            .Set(o => o.ShippedAt, newStatus == OrderStatus.Shipped ? DateTime.UtcNow : (DateTime?)null)
            .Inc(o => o.Version, 1)           // Increment version (optimistic concurrency)
            .CurrentDate("updatedAt");         // Server-side timestamp

        var result = await _orders.UpdateOneAsync(filter, update);
        if (result.ModifiedCount == 0)
            throw new InvalidOperationException($"Order {orderId} not found or already in state {newStatus}");
    }

    // Optimistic concurrency: update only if version matches
    public async Task<bool> UpdateWithOptimisticConcurrencyAsync(Order order, int expectedVersion)
    {
        var filter = Builders<Order>.Filter.And(
            Builders<Order>.Filter.Eq(o => o.Id, orderId: order.Id),
            Builders<Order>.Filter.Eq(o => o.Version, expectedVersion));

        var update = Builders<Order>.Update
            .Set(o => o.Status, order.Status)
            .Set(o => o.Lines, order.Lines)
            .Inc(o => o.Version, 1);

        var result = await _orders.UpdateOneAsync(filter, update);
        return result.ModifiedCount == 1; // false = concurrent modification detected
    }

    // Upsert: insert if not found, update if found
    public async Task UpsertProductAsync(Product product)
    {
        var filter = Builders<Product>.Filter.Eq(p => p.Sku, product.Sku);
        var update = Builders<Product>.Update
            .Set(p => p.Name, product.Name)
            .Set(p => p.Price, product.Price)
            .Set(p => p.Stock, product.Stock)
            .Set(p => p.UpdatedAt, DateTime.UtcNow)
            .SetOnInsert(p => p.Id, ObjectId.GenerateNewId().ToString()); // Only set on insert

        await _products.UpdateOneAsync(filter, update, new UpdateOptions { IsUpsert = true });
    }

    // FindOneAndUpdate: atomic read-modify — returns old or new document
    public async Task<Product?> DecrementStockAtomicAsync(string productId, int quantity)
    {
        var filter = Builders<Product>.Filter.And(
            Builders<Product>.Filter.Eq(p => p.Id, productId),
            Builders<Product>.Filter.Gte(p => p.Stock, quantity)); // Only if enough stock

        var update = Builders<Product>.Update
            .Inc(p => p.Stock, -quantity)
            .Set(p => p.UpdatedAt, DateTime.UtcNow);

        // ReturnDocument.After → returns the document AFTER update (updated stock value)
        var options = new FindOneAndUpdateOptions<Product>
        {
            ReturnDocument = ReturnDocument.After,
        };

        return await _products.FindOneAndUpdateAsync(filter, update, options);
        // Returns null if not found or insufficient stock → caller handles "out of stock"
    }

    // ── DELETE ────────────────────────────────────────────────────────────────

    public async Task<bool> SoftDeleteOrderAsync(string orderId)
    {
        // Prefer soft delete: set a deletedAt timestamp, filter it everywhere
        var filter = Builders<Order>.Filter.Eq(o => o.Id, orderId);
        var update = Builders<Order>.Update
            .Set("deletedAt", DateTime.UtcNow)
            .Set(o => o.Status, OrderStatus.Cancelled);
        var result = await _orders.UpdateOneAsync(filter, update);
        return result.ModifiedCount == 1;
    }

    public async Task HardDeleteAsync(string orderId)
    {
        await _orders.DeleteOneAsync(o => o.Id == orderId);
    }

    // ── BULK WRITE (batch multiple operations) ────────────────────────────────

    public async Task BulkUpdateStockAsync(Dictionary<string, int> stockDeltas)
    {
        var writes = stockDeltas.Select(kvp =>
            new UpdateOneModel<Product>(
                Builders<Product>.Filter.Eq(p => p.Id, kvp.Key),
                Builders<Product>.Update.Inc(p => p.Stock, kvp.Value))
            { IsUpsert = false })
            .Cast<WriteModel<Product>>()
            .ToList();

        // IsOrdered: false → execute in parallel, don't stop on errors
        var result = await _products.BulkWriteAsync(writes, new BulkWriteOptions { IsOrdered = false });
        Console.WriteLine($"Matched: {result.MatchedCount}, Modified: {result.ModifiedCount}");
    }
}

#endregion

// ============================================================================
// SECTION 5: QUERYING — FILTERS, PROJECTIONS, SORTING, PAGINATION
// ============================================================================

#region Querying

/*
 * FILTER BUILDERS — PREFER STRONGLY TYPED:
 *   Builders<T>.Filter.Eq / Ne / Gt / Gte / Lt / Lte
 *   Builders<T>.Filter.In / Nin
 *   Builders<T>.Filter.Regex
 *   Builders<T>.Filter.And / Or / Nor / Not
 *   Builders<T>.Filter.ElemMatch   → match elements within array
 *   Builders<T>.Filter.Size        → match array by length
 *   Builders<T>.Filter.Exists      → field existence check
 *   Builders<T>.Filter.Type        → BSON type check
 *   Builders<T>.Filter.Text        → full-text search (requires text index)
 *   Builders<T>.Filter.Near        → geospatial (requires 2dsphere index)
 *
 * PROJECTION:
 *   Include: Builders<T>.Projection.Include(x => x.Name)
 *   Exclude: Builders<T>.Projection.Exclude(x => x.Metadata)
 *   Can't mix include/exclude (except always can exclude _id).
 *
 * CURSOR vs ToList:
 *   - ToListAsync(): loads all results into memory — risky on large result sets.
 *   - ForEachAsync(): streams via cursor, lower memory.
 *   - ToCursorAsync(): full cursor control.
 *
 * PAGINATION:
 *   Skip/Limit: simple but SLOW for large skip values (MongoDB must scan skipped docs).
 *   Cursor-based (keyset pagination): filter by last seen _id or sorted field — O(log N) always.
 */

public class MongoQueryExamples
{
    private readonly IMongoCollection<Order> _orders;
    private readonly IMongoCollection<Product> _products;

    public MongoQueryExamples(IMongoDatabase db)
    {
        _orders   = db.GetCollection<Order>("orders");
        _products = db.GetCollection<Product>("products");
    }

    // Composing filters with And/Or
    public async Task<List<Order>> FindActiveOrdersForCustomerAsync(string customerId, decimal minAmount)
    {
        var filter = Builders<Order>.Filter.And(
            Builders<Order>.Filter.Eq(o => o.Customer.CustomerId, customerId),
            Builders<Order>.Filter.In(o => o.Status, new[] { OrderStatus.Pending, OrderStatus.Confirmed }),
            Builders<Order>.Filter.Gte(o => o.Total.Amount, minAmount),
            // Only non-deleted documents
            Builders<Order>.Filter.Or(
                Builders<Order>.Filter.Exists("deletedAt", exists: false),
                Builders<Order>.Filter.Eq<DateTime?>("deletedAt", null))
        );

        return await _orders
            .Find(filter)
            .SortByDescending(o => o.CreatedAt)
            .ToListAsync();
    }

    // Projection — only fetch needed fields (reduces network + memory)
    public async Task<List<OrderSummary>> GetOrderSummariesAsync(string customerId)
    {
        var filter     = Builders<Order>.Filter.Eq(o => o.Customer.CustomerId, customerId);
        var projection = Builders<Order>.Projection
            .Include(o => o.Id)
            .Include(o => o.Status)
            .Include(o => o.Total)
            .Include(o => o.CreatedAt)
            .Exclude(o => o.Lines) // Don't fetch line items for summary
            .Exclude(o => o.Metadata);

        return await _orders
            .Find(filter)
            .Project(projection)
            .As<OrderSummary>() // Map to a lighter DTO
            .ToListAsync();
    }

    // Cursor-based pagination (production-grade — no skip penalty)
    public async Task<(List<Order> Items, string? NextCursor)> GetOrdersPageAsync(
        string? afterId,
        int pageSize = 20)
    {
        var filter = afterId is null
            ? Builders<Order>.Filter.Empty
            : Builders<Order>.Filter.Gt(o => o.Id, afterId); // Continue after last seen ID

        var items = await _orders
            .Find(filter)
            .SortBy(o => o.Id) // Must sort on the cursor field
            .Limit(pageSize + 1) // Fetch one extra to detect next page
            .ToListAsync();

        string? nextCursor = null;
        if (items.Count > pageSize)
        {
            nextCursor = items[pageSize - 1].Id;
            items = items.Take(pageSize).ToList();
        }

        return (items, nextCursor);
    }

    // Skip/Limit pagination (simpler but avoid for large offsets)
    public async Task<List<Product>> GetProductsPageAsync(int page, int pageSize = 20)
    {
        // WARN: page 1000 * pageSize 20 = skip 20000 → full scan up to 20000 docs
        return await _products
            .Find(p => p.IsActive)
            .SortBy(p => p.Name)
            .Skip((page - 1) * pageSize)
            .Limit(pageSize)
            .ToListAsync();
    }

    // Array queries: ElemMatch on nested array
    public async Task<List<Order>> FindOrdersContainingProductAsync(string productId)
    {
        var filter = Builders<Order>.Filter.ElemMatch(
            o => o.Lines,
            line => line.ProductId == productId);

        return await _orders.Find(filter).ToListAsync();
    }

    // Text search (requires text index on Name + Tags)
    public async Task<List<Product>> TextSearchProductsAsync(string searchTerm)
    {
        var filter = Builders<Product>.Filter.Text(searchTerm,
            new TextSearchOptions { CaseSensitive = false, DiacriticSensitive = false });

        // Project relevance score
        var projection = Builders<Product>.Projection
            .MetaTextScore("score")
            .Include(p => p.Id)
            .Include(p => p.Name)
            .Include(p => p.Price);

        // Sort by relevance score
        var sort = Builders<Product>.Sort.MetaTextScore("score");

        return await _products
            .Find(filter)
            .Project<Product>(projection)
            .Sort(sort)
            .ToListAsync();
    }

    // Streaming large result sets with cursor (memory-efficient)
    public async Task ProcessAllOrdersAsync(Func<Order, Task> processFunc, CancellationToken ct)
    {
        var cursor = await _orders.FindAsync(
            Builders<Order>.Filter.Empty,
            new FindOptions<Order> { BatchSize = 100 }, // Fetch 100 at a time
            ct);

        await cursor.ForEachAsync(processFunc, ct);
    }
}

public record OrderSummary(string Id, OrderStatus Status, MoneyAmount Total, DateTime CreatedAt);

#endregion

// ============================================================================
// SECTION 6: AGGREGATION PIPELINE
// ============================================================================

#region Aggregation Pipeline

/*
 * AGGREGATION PIPELINE STAGES:
 *   $match     → filter documents (put EARLY to use indexes)
 *   $project   → reshape documents (include/exclude/compute fields)
 *   $group     → group by key + compute accumulators ($sum, $avg, $min, $max, $push)
 *   $sort      → sort (can use index if BEFORE $group/$project)
 *   $limit     → limit output count
 *   $skip      → skip N documents
 *   $lookup    → LEFT OUTER JOIN to another collection
 *   $unwind    → deconstruct array into individual documents
 *   $addFields → add computed fields without changing existing ones
 *   $replaceRoot → replace root with a sub-document
 *   $facet     → run multiple sub-pipelines in parallel (multi-category facets)
 *   $bucket    → group into ranges
 *   $out       → write results to a collection
 *   $merge     → merge results into a collection (upsert semantics)
 *
 * PERFORMANCE RULES:
 *   1. $match and $sort FIRST — use indexes before any transformation.
 *   2. $project early to reduce document size for downstream stages.
 *   3. $lookup is expensive — denormalize hot paths instead.
 *   4. allowDiskUse: true for large aggregations (>100MB working set).
 *
 * INTERVIEW TIP:
 *   Q: "How does $lookup differ from a SQL JOIN?"
 *   A: $lookup is a LEFT OUTER JOIN. It's done in the app tier (mongos or mongod),
 *      not via index merges like SQL. Very expensive at scale — prefer embedding
 *      frequently co-accessed data. For analytics, use $lookup; for OLTP, embed.
 */

public class MongoAggregationExamples
{
    private readonly IMongoCollection<Order> _orders;
    private readonly IMongoCollection<Product> _products;

    public MongoAggregationExamples(IMongoDatabase db)
    {
        _orders   = db.GetCollection<Order>("orders");
        _products = db.GetCollection<Product>("products");
    }

    // Sales summary per customer: group + sum
    public async Task<List<CustomerSalesSummary>> GetSalesByCustomerAsync(
        DateTime from, DateTime to)
    {
        return await _orders.Aggregate()
            // Stage 1: filter (uses index on CreatedAt + Status)
            .Match(o => o.CreatedAt >= from && o.CreatedAt <= to
                     && o.Status == OrderStatus.Delivered)
            // Stage 2: group by customer
            .Group(
                o => o.Customer.CustomerId,
                g => new CustomerSalesSummary
                {
                    CustomerId  = g.Key,
                    CustomerName = g.First().Customer.FullName,
                    OrderCount  = g.Count(),
                    TotalSpent  = g.Sum(o => o.Total.Amount),
                    AvgOrderValue = g.Average(o => o.Total.Amount),
                    LastOrderAt = g.Max(o => o.CreatedAt),
                })
            .SortByDescending(s => s.TotalSpent)
            .Limit(100)
            .ToListAsync();
    }

    // $unwind array + $group: total units sold per product
    public async Task<List<ProductSalesStats>> GetTopSellingProductsAsync(int topN)
    {
        return await _orders.Aggregate()
            .Match(o => o.Status == OrderStatus.Delivered)
            .Unwind(o => o.Lines)                   // Explode Lines array → one doc per line
            .Group(
                doc => doc["lines"]["productId"],   // Group by the unwound line's productId
                g => new ProductSalesStats
                {
                    ProductId    = g.Key.AsString,
                    UnitsSold    = g.Sum(doc => doc["lines"]["quantity"].ToInt32()),
                    Revenue      = g.Sum(doc => doc["lines"]["unitPrice"]["amount"].ToDecimal()),
                })
            .SortByDescending(s => s.Revenue)
            .Limit(topN)
            .ToListAsync();
    }

    // $lookup: join orders with products collection
    public async Task<List<BsonDocument>> GetOrdersWithProductDetailsAsync(string customerId)
    {
        return await _orders.Aggregate()
            .Match(o => o.Customer.CustomerId == customerId)
            .Unwind(o => o.Lines)
            .Lookup(
                foreignCollectionName: "products",
                localField:   "lines.productId",
                foreignField: "_id",
                @as:          "productDetails")
            .Unwind("$productDetails")
            .Project(Builders<BsonDocument>.Projection
                .Include("_id")
                .Include("status")
                .Include("lines.quantity")
                .Include("productDetails.name")
                .Include("productDetails.price"))
            .ToListAsync();
    }

    // $facet: multiple aggregations in one pass (for faceted search UI)
    public async Task<BsonDocument> GetProductFacetsAsync(string category)
    {
        var pipeline = new[]
        {
            new BsonDocument("$match", new BsonDocument
            {
                { "category", category },
                { "isActive", true },
            }),
            new BsonDocument("$facet", new BsonDocument
            {
                // Facet 1: price histogram
                { "priceRanges", new BsonArray
                    {
                        new BsonDocument("$bucket", new BsonDocument
                        {
                            { "groupBy", "$price" },
                            { "boundaries", new BsonArray { 0, 25, 50, 100, 250, 1000 } },
                            { "default", "1000+" },
                            { "output", new BsonDocument { { "count", new BsonDocument("$sum", 1) } } },
                        })
                    }
                },
                // Facet 2: tags frequency
                { "popularTags", new BsonArray
                    {
                        new BsonDocument("$unwind", "$tags"),
                        new BsonDocument("$group", new BsonDocument
                        {
                            { "_id", "$tags" },
                            { "count", new BsonDocument("$sum", 1) },
                        }),
                        new BsonDocument("$sort", new BsonDocument("count", -1)),
                        new BsonDocument("$limit", 10),
                    }
                },
                // Facet 3: total count
                { "totalCount", new BsonArray
                    {
                        new BsonDocument("$count", "count"),
                    }
                },
            }),
        };

        return await _products.Aggregate<BsonDocument>(pipeline).FirstOrDefaultAsync()
            ?? new BsonDocument();
    }

    // $merge: materialize aggregation result into a reporting collection
    public async Task RefreshSalesSummaryAsync()
    {
        await _orders.Aggregate()
            .Match(o => o.Status == OrderStatus.Delivered)
            .Group(
                o => o.Customer.CustomerId,
                g => new { CustomerId = g.Key, TotalSpent = g.Sum(o => o.Total.Amount) })
            .MergeAsync("customer_sales_summary", new MergeStageOptions<BsonDocument>
            {
                WhenMatched  = MergeStageWhenMatched.Replace,
                WhenNotMatched = MergeStageWhenNotMatched.Insert,
            });
    }
}

public record CustomerSalesSummary
{
    public string   CustomerId    { get; init; } = default!;
    public string   CustomerName  { get; init; } = default!;
    public int      OrderCount    { get; init; }
    public decimal  TotalSpent    { get; init; }
    public decimal  AvgOrderValue { get; init; }
    public DateTime LastOrderAt   { get; init; }
}

public record ProductSalesStats
{
    public string  ProductId { get; init; } = default!;
    public int     UnitsSold { get; init; }
    public decimal Revenue   { get; init; }
}

#endregion

// ============================================================================
// SECTION 7: INDEXES — TYPES, STRATEGY & EXPLAIN PLANS
// ============================================================================

#region Indexes

/*
 * INDEX TYPES:
 *   Single field:   { field: 1 } ascending, { field: -1 } descending
 *   Compound:       { a: 1, b: -1 } — ESR rule: Equality → Sort → Range
 *   Multikey:       Auto-created when field is an array (index per array element)
 *   Text:           Full-text search index (one per collection)
 *   Geospatial:     2d (flat), 2dsphere (spherical/GeoJSON)
 *   Hashed:         { field: "hashed" } — for even shard key distribution
 *   Wildcard:       { "$**": 1 } — indexes all fields (flexible schema)
 *   Partial:        Index with filter expression — smaller, more efficient
 *   Sparse:         Only indexes documents where the field exists
 *   TTL:            Auto-delete documents after expiry (logs, sessions, OTPs)
 *   Unique:         Enforce uniqueness (like SQL UNIQUE constraint)
 *
 * ESR RULE (for compound indexes):
 *   1. Equality fields first (exact match conditions)
 *   2. Sort fields second
 *   3. Range fields last (>, <, $in on multiple values)
 *   This order maximizes index usage for queries with all three.
 *
 * COVERED QUERY:
 *   A query is "covered" if all fields in the filter AND projection are in the index.
 *   MongoDB never touches the actual documents → maximum performance.
 *
 * EXPLAIN PLAN:
 *   .explain("executionStats") → IXSCAN (good) vs COLLSCAN (bad — no index used).
 *   Look for: totalDocsExamined (should be close to nReturned).
 *   High totalDocsExamined vs low nReturned = bad selectivity → add/refine index.
 */

public class MongoIndexManager
{
    private readonly IMongoCollection<Order> _orders;
    private readonly IMongoCollection<Product> _products;

    public MongoIndexManager(IMongoDatabase db)
    {
        _orders   = db.GetCollection<Order>("orders");
        _products = db.GetCollection<Product>("products");
    }

    public async Task CreateAllIndexesAsync()
    {
        await CreateOrderIndexesAsync();
        await CreateProductIndexesAsync();
    }

    private async Task CreateOrderIndexesAsync()
    {
        var indexes = new[]
        {
            // ESR compound: customerId (equality) + createdAt (sort) + status (range)
            new CreateIndexModel<Order>(
                Builders<Order>.IndexKeys
                    .Ascending(o => o.Customer.CustomerId)
                    .Descending(o => o.CreatedAt)
                    .Ascending(o => o.Status),
                new CreateIndexOptions { Name = "idx_customer_created_status" }),

            // Status + createdAt for admin dashboard queries
            new CreateIndexModel<Order>(
                Builders<Order>.IndexKeys
                    .Ascending(o => o.Status)
                    .Descending(o => o.CreatedAt),
                new CreateIndexOptions { Name = "idx_status_created" }),

            // TTL index: auto-delete cancelled orders after 90 days
            // Field must be a BSON Date type
            new CreateIndexModel<Order>(
                Builders<Order>.IndexKeys.Ascending("deletedAt"),
                new CreateIndexOptions
                {
                    Name              = "idx_ttl_deleted",
                    ExpireAfter       = TimeSpan.FromDays(90),
                    Sparse            = true, // Only index docs where deletedAt exists
                }),

            // Partial index: only index Pending/Confirmed orders (active workload)
            // Much smaller than a full-status index
            new CreateIndexModel<Order>(
                Builders<Order>.IndexKeys.Ascending(o => o.CreatedAt),
                new CreateIndexOptions
                {
                    Name             = "idx_active_orders_created",
                    PartialFilterExpression = Builders<Order>.Filter.In(
                        o => o.Status, new[] { OrderStatus.Pending, OrderStatus.Confirmed }),
                }),
        };

        await _orders.Indexes.CreateManyAsync(indexes);
    }

    private async Task CreateProductIndexesAsync()
    {
        var indexes = new[]
        {
            // Unique index on SKU — like SQL UNIQUE constraint
            new CreateIndexModel<Product>(
                Builders<Product>.IndexKeys.Ascending(p => p.Sku),
                new CreateIndexOptions { Unique = true, Name = "idx_sku_unique" }),

            // Compound: category (equality) + price (range) — ESR
            new CreateIndexModel<Product>(
                Builders<Product>.IndexKeys
                    .Ascending(p => p.Category)
                    .Ascending(p => p.IsActive)
                    .Ascending(p => p.Price),
                new CreateIndexOptions { Name = "idx_category_active_price" }),

            // Text index for full-text search on name + tags
            new CreateIndexModel<Product>(
                Builders<Product>.IndexKeys
                    .Text(p => p.Name)
                    .Text("tags"), // array field
                new CreateIndexOptions
                {
                    Name = "idx_text_name_tags",
                    // Weights: name matches score 10x higher than tag matches
                    Weights = new BsonDocument { { "name", 10 }, { "tags", 3 } },
                    DefaultLanguage = "english",
                }),

            // Multikey on Tags array (auto-created, but explicit for clarity)
            new CreateIndexModel<Product>(
                Builders<Product>.IndexKeys.Ascending("tags"),
                new CreateIndexOptions { Name = "idx_tags" }),

            // TTL: expire inactive products after 1 year with no update
            new CreateIndexModel<Product>(
                Builders<Product>.IndexKeys.Ascending(p => p.UpdatedAt),
                new CreateIndexOptions
                {
                    Name        = "idx_ttl_updated",
                    ExpireAfter = TimeSpan.FromDays(365),
                    Sparse      = true,
                }),
        };

        await _products.Indexes.CreateManyAsync(indexes);
    }

    // Run explain plan to diagnose query performance
    public async Task<BsonDocument> ExplainQueryAsync(FilterDefinition<Order> filter)
    {
        var command = new BsonDocument
        {
            { "explain", new BsonDocument
                {
                    { "find", "orders" },
                    { "filter", filter.Render(
                        BsonSerializer.SerializerRegistry.GetSerializer<Order>(),
                        BsonSerializer.SerializerRegistry) },
                }
            },
            { "verbosity", "executionStats" },
        };

        var db = _orders.Database;
        var result = await db.RunCommandAsync<BsonDocument>(command);

        // Log key metrics from explain output
        var execStats = result.GetValue("executionStats", BsonNull.Value).AsBsonDocument;
        if (execStats is not null)
        {
            Console.WriteLine($"Docs Examined: {execStats.GetValue("totalDocsExamined", 0)}");
            Console.WriteLine($"Docs Returned: {execStats.GetValue("nReturned", 0)}");
            Console.WriteLine($"Execution ms:  {execStats.GetValue("executionTimeMillis", 0)}");
            // IXSCAN = index used (good), COLLSCAN = full scan (needs index)
            Console.WriteLine($"Stage: {execStats["executionStages"]["stage"]}");
        }

        return result;
    }
}

#endregion

// ============================================================================
// SECTION 8: TRANSACTIONS — MULTI-DOCUMENT ACID
// ============================================================================

#region Transactions

/*
 * MONGODB MULTI-DOCUMENT TRANSACTIONS (since 4.0):
 *   - ACID across multiple documents, collections, and databases.
 *   - Require a REPLICA SET (or sharded cluster with 4.2+).
 *   - Transactions have a 60-second time limit by default (configurable).
 *   - Heavier than single-document operations — use sparingly.
 *
 * WHEN TO USE TRANSACTIONS:
 *   - Reserve inventory + create order atomically.
 *   - Transfer balance between two accounts.
 *   - Any multi-document write that must be all-or-nothing.
 *
 * WHEN NOT TO USE TRANSACTIONS:
 *   - Single document write (atomic by nature in MongoDB — no transaction needed).
 *   - High-throughput writes (transactions add overhead).
 *   - If you can use the Outbox pattern instead (writes to outbox in same transaction).
 *
 * INTERVIEW TIP:
 *   Q: "Are single-document operations atomic in MongoDB?"
 *   A: YES. A single document write (including nested arrays and sub-documents)
 *      is always atomic. This is why embedding related data in one document
 *      often eliminates the need for transactions entirely.
 */

public class OrderCheckoutService
{
    private readonly IMongoClient _client;
    private readonly IMongoDatabase _db;
    private readonly ILogger<OrderCheckoutService> _logger;

    public OrderCheckoutService(IMongoClient client, IMongoDatabase db, ILogger<OrderCheckoutService> logger)
    {
        _client = client;
        _db     = db;
        _logger = logger;
    }

    // Multi-document transaction: reserve stock + create order atomically
    public async Task<string> CheckoutAsync(
        string customerId,
        List<(string ProductId, int Qty)> items,
        CancellationToken ct = default)
    {
        using var session = await _client.StartSessionAsync(cancellationToken: ct);

        var transactionOptions = new TransactionOptions(
            readConcern:    ReadConcern.Snapshot,   // Consistent point-in-time read
            writeConcern:   WriteConcern.WMajority,  // Wait for majority ack before commit
            readPreference: ReadPreference.Primary); // Always read from primary in transactions

        return await session.WithTransactionAsync(
            async (sess, token) =>
            {
                var products = _db.GetCollection<Product>("products");
                var orders   = _db.GetCollection<Order>("orders");

                decimal orderTotal = 0;
                var orderLines     = new List<OrderLine>();

                // Step 1: Reserve inventory for each item
                foreach (var (productId, qty) in items)
                {
                    var filter = Builders<Product>.Filter.And(
                        Builders<Product>.Filter.Eq(p => p.Id, productId),
                        Builders<Product>.Filter.Gte(p => p.Stock, qty),
                        Builders<Product>.Filter.Eq(p => p.IsActive, true));

                    var update = Builders<Product>.Update
                        .Inc(p => p.Stock, -qty)
                        .Set(p => p.UpdatedAt, DateTime.UtcNow);

                    var opts = new FindOneAndUpdateOptions<Product>
                        { ReturnDocument = ReturnDocument.After };

                    var product = await products.FindOneAndUpdateAsync(
                        sess, filter, update, opts, token);

                    if (product is null)
                        throw new InsufficientStockException(productId);

                    orderLines.Add(new OrderLine
                    {
                        ProductId   = product.Id,
                        ProductName = product.Name,
                        Quantity    = qty,
                        UnitPrice   = new MoneyAmount(product.Price),
                    });
                    orderTotal += product.Price * qty;
                }

                // Step 2: Create the order in the same transaction
                var order = new Order
                {
                    Customer = new CustomerSnapshot { CustomerId = customerId },
                    Lines    = orderLines,
                    Total    = new MoneyAmount(orderTotal),
                    Status   = OrderStatus.Confirmed,
                };

                await orders.InsertOneAsync(sess, order, cancellationToken: token);
                _logger.LogInformation("Order {OrderId} created atomically with stock reservation", order.Id);
                return order.Id;
            },
            transactionOptions,
            ct);
        // WithTransactionAsync auto-retries on transient transaction errors
        // (WriteConflict, TransientTransactionError labels)
    }
}

public class InsufficientStockException : Exception
{
    public string ProductId { get; }
    public InsufficientStockException(string productId)
        : base($"Insufficient stock for product {productId}") => ProductId = productId;
}

#endregion

// ============================================================================
// SECTION 9: CHANGE STREAMS — REAL-TIME EVENT WATCHING
// ============================================================================

#region Change Streams

/*
 * CHANGE STREAMS:
 *   - Watch for insert/update/delete/replace events on a collection, database, or cluster.
 *   - Built on MongoDB's oplog (operation log from the replica set).
 *   - Require: replica set or sharded cluster (not standalone).
 *   - Use cases: event-driven microservices, CDC (Change Data Capture), cache invalidation,
 *                live dashboards, audit logging.
 *
 * RESUME TOKENS:
 *   - Each change event has a _id (resume token).
 *   - Persist the resume token to pick up where you left off after a crash.
 *   - Pass ResumeAfter or StartAfterTime on the watch options.
 *
 * vs POLLING:
 *   Change streams: push-based, low-latency (milliseconds), no DB polling load.
 *   Polling: pull-based, simpler, latency = poll interval, extra DB read load.
 *
 * INTERVIEW TIP:
 *   Change streams can replace the Outbox polling relay — instead of a relay service
 *   polling the outbox table, watch the outbox collection with a change stream
 *   and react to insert events in real time (near-zero latency).
 */

public class OrderChangeStreamService : BackgroundService
{
    private readonly IMongoClient _client;
    private readonly IMongoDatabase _db;
    private readonly IEventPublisher _publisher;  // Kafka publisher from earlier
    private readonly ILogger<OrderChangeStreamService> _logger;

    // Persist resume token so we survive restarts
    private readonly IResumeTokenStore _tokenStore;

    public OrderChangeStreamService(
        IMongoClient client, IMongoDatabase db,
        IEventPublisher publisher, IResumeTokenStore tokenStore,
        ILogger<OrderChangeStreamService> logger)
    {
        _client     = client;
        _db         = db;
        _publisher  = publisher;
        _tokenStore = tokenStore;
        _logger     = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var orders = _db.GetCollection<Order>("orders");

        // Pipeline: only watch insert and update to Status field
        var pipeline = new EmptyPipelineDefinition<ChangeStreamDocument<Order>>()
            .Match(change =>
                change.OperationType == ChangeStreamOperationType.Insert ||
                (change.OperationType == ChangeStreamOperationType.Update &&
                 change.UpdateDescription.UpdatedFields.Contains("status")));

        var savedToken = await _tokenStore.GetAsync();

        var watchOptions = new ChangeStreamOptions
        {
            FullDocument    = ChangeStreamFullDocumentOption.UpdateLookup, // Fetch full doc on update
            ResumeAfter     = savedToken, // Resume from last saved token (null = start now)
            MaxAwaitTime    = TimeSpan.FromSeconds(10),
            BatchSize       = 100,
        };

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var cursor = await orders.WatchAsync(pipeline, watchOptions, stoppingToken);

                await cursor.ForEachAsync(async change =>
                {
                    _logger.LogDebug("Change stream event: {OperationType} on Order {Id}",
                        change.OperationType, change.DocumentKey["_id"]);

                    switch (change.OperationType)
                    {
                        case ChangeStreamOperationType.Insert:
                            await _publisher.PublishAsync(
                                "order.order.created",
                                change.FullDocument.Id,
                                new OrderCreatedEvent { OrderId = change.FullDocument.Id });
                            break;

                        case ChangeStreamOperationType.Update:
                            if (change.FullDocument?.Status == OrderStatus.Shipped)
                                await _publisher.PublishAsync(
                                    "order.order.shipped",
                                    change.FullDocument.Id,
                                    new OrderShippedEvent { OrderId = change.FullDocument.Id });
                            break;
                    }

                    // Persist resume token AFTER successful processing
                    await _tokenStore.SaveAsync(change.ResumeToken);

                    // Update watch options for next cursor (in case of reconnect)
                    watchOptions.ResumeAfter = change.ResumeToken;

                }, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { break; }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Change stream error, reconnecting in 5s");
                await Task.Delay(5_000, stoppingToken);
                // Loop restarts watch with the last saved ResumeAfter token
            }
        }
    }
}

// Stub types
public interface IEventPublisher { Task PublishAsync<T>(string topic, string key, T @event, CancellationToken ct = default); }
public interface IResumeTokenStore { Task<BsonDocument?> GetAsync(); Task SaveAsync(BsonDocument token); }
public record OrderCreatedEvent { public string OrderId { get; init; } = ""; }
public record OrderShippedEvent { public string OrderId { get; init; } = ""; }

#endregion

// ============================================================================
// SECTION 10: REPLICA SETS, WRITE CONCERNS & READ PREFERENCES
// ============================================================================

#region Replica Sets & Read/Write Concerns

/*
 * REPLICA SET:
 *   - Minimum 3 members recommended: 1 Primary + 2 Secondaries.
 *   - Primary: handles all writes + default reads.
 *   - Secondaries: replicate primary's oplog asynchronously.
 *   - Election: if primary goes down, secondaries elect a new primary via Raft-like protocol.
 *   - Arbiter: votes in elections but stores no data (use sparingly).
 *
 * WRITE CONCERN:
 *   w:0            → no acknowledgment (fire and forget)
 *   w:1            → primary ack (default)
 *   w:"majority"   → ack from majority of voting members (durable against primary failure)
 *   w:N            → ack from N members
 *   j:true         → wait for journal write (on-disk durability even before replication)
 *   wtimeout:5000  → fail if not acked within 5s
 *
 * READ PREFERENCE:
 *   Primary:            Always read from primary (default, consistent)
 *   PrimaryPreferred:   Primary if available, else secondary
 *   Secondary:          Always read from secondary (possibly stale)
 *   SecondaryPreferred: Secondary if available, else primary (scale reads)
 *   Nearest:            Lowest network latency member
 *
 * CAUSAL CONSISTENCY:
 *   - "Read your own writes" across sessions.
 *   - Use client.StartSession() + pass session to all operations.
 *   - The driver tracks operation timestamps (cluster time + operation time).
 *
 * INTERVIEW TIP:
 *   Q: "Can you lose writes with w:majority?"
 *   A: Extremely unlikely but theoretically yes if more than half the voting
 *      members fail simultaneously before replication. In practice, w:majority
 *      with j:true is considered durable for production.
 */

public class ReadPreferenceExamples
{
    private readonly IMongoClient _client;

    public ReadPreferenceExamples(IMongoClient client) => _client = client;

    // Default: write with majority concern, read from primary
    public IMongoCollection<Order> GetDurableCollection()
    {
        var db = _client.GetDatabase("shop", new MongoDatabaseSettings
        {
            WriteConcern   = WriteConcern.WMajority,
            ReadConcern    = ReadConcern.Majority,   // Only read data committed by majority
            ReadPreference = ReadPreference.Primary, // Consistent reads from primary
        });
        return db.GetCollection<Order>("orders");
    }

    // Scale reads: route read-heavy reporting queries to secondaries
    public IMongoCollection<Order> GetReportingCollection()
    {
        var db = _client.GetDatabase("shop", new MongoDatabaseSettings
        {
            ReadPreference = ReadPreference.Secondary,
            ReadConcern    = ReadConcern.Local, // Slight staleness acceptable for reports
        });
        return db.GetCollection<Order>("orders");
    }

    // Causal consistency: ensure "read your own writes"
    public async Task CausalConsistencyExampleAsync()
    {
        using var session = await _client.StartSessionAsync(new ClientSessionOptions
        {
            DefaultTransactionOptions = new TransactionOptions(
                writeConcern: WriteConcern.WMajority)
        });
        session.StartTransaction();

        var orders = _client.GetDatabase("shop").GetCollection<Order>("orders");

        // Write with session
        var order = new Order { Status = OrderStatus.Confirmed };
        await orders.InsertOneAsync(session, order);

        // Read immediately after write within same session — guaranteed to see the write
        var found = await orders.Find(session, o => o.Id == order.Id).FirstOrDefaultAsync();
        Console.WriteLine($"Causal read: {found?.Status}"); // Confirmed (not stale)

        await session.CommitTransactionAsync();
    }
}

#endregion

// ============================================================================
// SECTION 11: SCHEMA DESIGN PATTERNS
// ============================================================================

#region Schema Design Patterns

/*
 * THE #1 RULE OF MONGODB SCHEMA DESIGN:
 *   "Model data the way your APPLICATION QUERIES it, not the way it 'exists' in reality."
 *   Unlike SQL (normalized for storage efficiency), MongoDB schemas are denormalized
 *   for query performance. Design for your most common access patterns FIRST.
 *
 * EMBEDDING vs REFERENCING:
 *   EMBED when:
 *     - Data is always accessed together ("has-a" relationship, data owned by parent).
 *     - 1:1 or 1:few relationships (address, order lines, comments).
 *     - Child documents don't exceed 16MB limit.
 *     - Child documents don't grow unboundedly (avoid the Unbounded Array anti-pattern).
 *
 *   REFERENCE when:
 *     - Data is accessed independently (products exist without orders).
 *     - 1:many or many:many relationships where "many" is large.
 *     - Child documents are large and not always needed.
 *     - Data is shared across multiple parent documents.
 *
 * KEY SCHEMA PATTERNS:
 *   Attribute Pattern:      Store dynamic key-value pairs in an array of {k, v} objects.
 *   Bucket Pattern:         Group time-series data into time-windowed documents.
 *   Computed Pattern:       Pre-compute and store aggregated values (e.g., order total).
 *   Document Versioning:    Keep a version number + history collection for audit.
 *   Outlier Pattern:        Store most data embedded, overflow to extra documents.
 *   Tree Patterns:          Parent reference, child reference, array of ancestors.
 *
 * ANTI-PATTERNS TO MENTION IN INTERVIEWS:
 *   - Unbounded Arrays:     Appending to an array indefinitely → document grows past 16MB.
 *   - Massive Number of Collections: no benefit in MongoDB; use one collection + discriminator.
 *   - Overly Normalized:    Excessive $lookup = poor performance.
 *   - Using _id as a join key everywhere = you're using MongoDB like a SQL DB.
 */

// Attribute Pattern: flexible product specifications
public class ProductWithSpecs
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id   { get; set; } = ObjectId.GenerateNewId().ToString();
    public string Name { get; set; } = default!;

    // Instead of fixed columns: color, weight, size, voltage, resolution...
    // Store as a searchable array of k/v pairs
    public List<ProductSpec> Specs { get; set; } = new();
}

public record ProductSpec(string Key, string Value, string? Unit = null);
// Query: db.products.find({ "specs": { $elemMatch: { k: "color", v: "red" } } })
// Index:  { "specs.k": 1, "specs.v": 1 }

// Bucket Pattern: time-series IoT readings
public class SensorReadingBucket
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; } = ObjectId.GenerateNewId().ToString();

    public string   SensorId     { get; set; } = default!;
    public DateTime HourStart    { get; set; }   // One doc per sensor per hour
    public int      ReadingCount  { get; set; }  // Pre-counted
    public double   SumTemperature { get; set; } // Pre-summed for avg calculation
    public List<SensorReading> Readings { get; set; } = new(); // Max 60 per bucket
}

public record SensorReading(DateTime Timestamp, double Temperature, double Humidity);
// Benefits: far fewer documents, pre-computed stats, efficient range queries

// Document Versioning Pattern
public class OrderWithHistory
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id      { get; set; } = default!;
    public int    Version { get; set; }
    // Current state fields...
    public OrderStatus Status { get; set; }
}

public class OrderHistoryEntry
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id         { get; set; } = ObjectId.GenerateNewId().ToString();
    public string OrderId    { get; set; } = default!;
    public int    Version    { get; set; }
    public string ChangedBy  { get; set; } = default!;
    public DateTime ChangedAt { get; set; } = DateTime.UtcNow;
    public BsonDocument Snapshot { get; set; } = default!; // Full document at this version
}

#endregion

// ============================================================================
// SECTION 12: REPOSITORY PATTERN & CLEAN ARCHITECTURE
// ============================================================================

#region Repository Pattern

public interface IOrderRepository
{
    Task<Order?> GetByIdAsync(string id, CancellationToken ct = default);
    Task<List<Order>> GetByCustomerAsync(string customerId, CancellationToken ct = default);
    Task<(List<Order> Items, string? Cursor)> GetPageAsync(string? afterId, int size, CancellationToken ct = default);
    Task<string> CreateAsync(Order order, CancellationToken ct = default);
    Task<bool> UpdateStatusAsync(string id, OrderStatus status, int expectedVersion, CancellationToken ct = default);
    Task<bool> DeleteAsync(string id, CancellationToken ct = default);
}

public interface IProductRepository
{
    Task<Product?> GetByIdAsync(string id, CancellationToken ct = default);
    Task<Product?> GetBySkuAsync(string sku, CancellationToken ct = default);
    Task UpsertAsync(Product product, CancellationToken ct = default);
    Task<Product?> TryDecrementStockAsync(string id, int qty, CancellationToken ct = default);
}

public class MongoOrderRepository : IOrderRepository
{
    private readonly IMongoCollection<Order> _collection;

    public MongoOrderRepository(IMongoDatabase db)
        => _collection = db.GetCollection<Order>("orders");

    public async Task<Order?> GetByIdAsync(string id, CancellationToken ct = default)
        => await _collection.Find(o => o.Id == id).FirstOrDefaultAsync(ct);

    public async Task<List<Order>> GetByCustomerAsync(string customerId, CancellationToken ct = default)
        => await _collection
            .Find(o => o.Customer.CustomerId == customerId)
            .SortByDescending(o => o.CreatedAt)
            .ToListAsync(ct);

    public async Task<(List<Order> Items, string? Cursor)> GetPageAsync(
        string? afterId, int size, CancellationToken ct = default)
    {
        var filter = afterId is null
            ? Builders<Order>.Filter.Empty
            : Builders<Order>.Filter.Gt(o => o.Id, afterId);

        var items = await _collection
            .Find(filter)
            .SortBy(o => o.Id)
            .Limit(size + 1)
            .ToListAsync(ct);

        string? cursor = null;
        if (items.Count > size)
        {
            cursor = items[size - 1].Id;
            items  = items.Take(size).ToList();
        }

        return (items, cursor);
    }

    public async Task<string> CreateAsync(Order order, CancellationToken ct = default)
    {
        await _collection.InsertOneAsync(order, cancellationToken: ct);
        return order.Id;
    }

    public async Task<bool> UpdateStatusAsync(string id, OrderStatus status, int expectedVersion, CancellationToken ct = default)
    {
        var filter = Builders<Order>.Filter.And(
            Builders<Order>.Filter.Eq(o => o.Id, id),
            Builders<Order>.Filter.Eq(o => o.Version, expectedVersion));

        var update = Builders<Order>.Update
            .Set(o => o.Status, status)
            .Inc(o => o.Version, 1);

        var result = await _collection.UpdateOneAsync(filter, update, cancellationToken: ct);
        return result.ModifiedCount == 1;
    }

    public async Task<bool> DeleteAsync(string id, CancellationToken ct = default)
    {
        var result = await _collection.UpdateOneAsync(
            o => o.Id == id,
            Builders<Order>.Update.Set("deletedAt", DateTime.UtcNow),
            cancellationToken: ct);
        return result.ModifiedCount == 1;
    }
}

public class MongoProductRepository : IProductRepository
{
    private readonly IMongoCollection<Product> _collection;

    public MongoProductRepository(IMongoDatabase db)
        => _collection = db.GetCollection<Product>("products");

    public async Task<Product?> GetByIdAsync(string id, CancellationToken ct = default)
        => await _collection.Find(p => p.Id == id).FirstOrDefaultAsync(ct);

    public async Task<Product?> GetBySkuAsync(string sku, CancellationToken ct = default)
        => await _collection.Find(p => p.Sku == sku).FirstOrDefaultAsync(ct);

    public async Task UpsertAsync(Product product, CancellationToken ct = default)
    {
        var filter = Builders<Product>.Filter.Eq(p => p.Sku, product.Sku);
        var update = Builders<Product>.Update
            .Set(p => p.Name,      product.Name)
            .Set(p => p.Price,     product.Price)
            .Set(p => p.Stock,     product.Stock)
            .Set(p => p.Tags,      product.Tags)
            .Set(p => p.Category,  product.Category)
            .Set(p => p.UpdatedAt, DateTime.UtcNow)
            .SetOnInsert(p => p.Id, ObjectId.GenerateNewId().ToString());

        await _collection.UpdateOneAsync(filter, update, new UpdateOptions { IsUpsert = true }, ct);
    }

    public async Task<Product?> TryDecrementStockAsync(string id, int qty, CancellationToken ct = default)
    {
        var filter = Builders<Product>.Filter.And(
            Builders<Product>.Filter.Eq(p => p.Id, id),
            Builders<Product>.Filter.Gte(p => p.Stock, qty));

        var update = Builders<Product>.Update
            .Inc(p => p.Stock, -qty)
            .Set(p => p.UpdatedAt, DateTime.UtcNow);

        return await _collection.FindOneAndUpdateAsync(
            filter, update,
            new FindOneAndUpdateOptions<Product> { ReturnDocument = ReturnDocument.After },
            ct);
    }
}

#endregion

// ============================================================================
// SECTION 13: GRIDFS — LARGE FILE STORAGE
// ============================================================================

#region GridFS

/*
 * GRIDFS:
 *   - For files > 16MB (MongoDB's document size limit).
 *   - Splits files into 255KB chunks stored in <bucket>.chunks collection.
 *   - Metadata stored in <bucket>.files collection.
 *   - Use cases: product images, invoices, user uploads, reports.
 *
 *   INTERVIEW TIP:
 *   Q: "Should you use GridFS for all file storage?"
 *   A: No. For most production systems, prefer object storage (S3, Azure Blob, GCS)
 *      for files — cheaper, more scalable, CDN-compatible. Use GridFS when:
 *      you need to store files close to the data, atomic writes with metadata,
 *      or your infrastructure doesn't have object storage.
 */

public class GridFsFileService
{
    private readonly IGridFSBucket _bucket;

    public GridFsFileService(IMongoDatabase db)
    {
        _bucket = new GridFSBucket(db, new GridFSBucketOptions
        {
            BucketName = "uploads",
            ChunkSizeBytes = 255 * 1024, // 255KB chunks (default)
            WriteConcern = WriteConcern.WMajority,
        });
    }

    public async Task<string> UploadFileAsync(
        string filename, Stream fileStream, string contentType, CancellationToken ct = default)
    {
        var metadata = new BsonDocument
        {
            { "contentType", contentType },
            { "uploadedAt",  DateTime.UtcNow },
        };

        var options = new GridFSUploadOptions { Metadata = metadata };
        var fileId  = await _bucket.UploadFromStreamAsync(filename, fileStream, options, ct);
        return fileId.ToString();
    }

    public async Task<(Stream Stream, string FileName)> DownloadFileAsync(
        string fileId, CancellationToken ct = default)
    {
        var objectId = new ObjectId(fileId);
        var info     = await _bucket.Find(Builders<GridFSFileInfo>.Filter.Eq("_id", objectId))
                                    .FirstOrDefaultAsync(ct)
                       ?? throw new FileNotFoundException($"File {fileId} not found");

        var stream   = await _bucket.OpenDownloadStreamAsync(objectId, cancellationToken: ct);
        return (stream, info.Filename);
    }

    public async Task DeleteFileAsync(string fileId, CancellationToken ct = default)
        => await _bucket.DeleteAsync(new ObjectId(fileId), ct);
}

#endregion

// ============================================================================
// SECTION 14: SHARDING — CONCEPTS & SHARD KEY STRATEGY
// ============================================================================

#region Sharding

/*
 * SHARDING:
 *   - Horizontal partitioning of data across multiple shards (each shard is a replica set).
 *   - Managed by mongos routers and config servers.
 *   - Only needed at VERY high scale (tens of millions of documents, high write throughput).
 *
 * SHARD KEY SELECTION (most critical decision):
 *   Criteria for a good shard key:
 *   1. High cardinality: enough distinct values to distribute data evenly.
 *   2. Low frequency: no single value dominates (avoid "hotspot shards").
 *   3. Non-monotonically increasing: avoid ObjectId/_id as sole shard key
 *      (all inserts go to last chunk → insert hotspot).
 *   4. Query isolation: most queries include the shard key → routed to one shard (targeted).
 *      Queries WITHOUT shard key go to ALL shards (scatter-gather → expensive).
 *
 * SHARD KEY PATTERNS:
 *   Hashed sharding:    { field: "hashed" } → even distribution, no range queries.
 *   Range sharding:     { field: 1 } → efficient range queries, risk of hotspots.
 *   Compound shard key: { tenantId: 1, _id: 1 } → isolated per tenant, avoids monotonic problem.
 *   Zone sharding:      Route data to specific shards by range (geo-affinity: EU data to EU shard).
 *
 * INTERVIEW TIP:
 *   Q: "Why is ObjectId a bad sole shard key?"
 *   A: ObjectId is monotonically increasing (timestamp-based). All new inserts map
 *      to the same chunk (the "last" chunk), which lives on one shard → write hotspot.
 *      Solution: use hashed shard key on _id, or a compound key like {tenantId, _id}.
 *
 *   Q: "What is a jumbo chunk?"
 *   A: A chunk that can't be split because all documents share the same shard key value.
 *      The balancer can't move it, causing imbalance. Solution: choose higher-cardinality key.
 */

// Code: selecting shard key strategy in application (compound key example)
public class ShardKeyExample
{
    /*
     * For a multi-tenant SaaS:
     *   Shard key: { tenantId: 1, _id: 1 }
     *   - tenantId: all queries scoped per tenant → targeted queries.
     *   - _id: breaks monotonic inserts within each tenant's range.
     *   - Result: data is colocated per tenant (Zone sharding possible), no insert hotspot.
     *
     * Enabling sharding (run in mongo shell / AdminClient):
     *   sh.enableSharding("shop")
     *   sh.shardCollection("shop.orders", { tenantId: 1, _id: 1 })
     */

    public static async Task EnableShardingAsync(IMongoClient client)
    {
        var adminDb = client.GetDatabase("admin");

        // Enable sharding on database
        await adminDb.RunCommandAsync<BsonDocument>(
            new BsonDocument("enableSharding", "shop"));

        // Shard the orders collection with compound hashed key
        await adminDb.RunCommandAsync<BsonDocument>(new BsonDocument
        {
            { "shardCollection", "shop.orders" },
            { "key", new BsonDocument { { "tenantId", 1 }, { "_id", "hashed" } } },
        });
    }
}

#endregion

// ============================================================================
// SECTION 15: PERFORMANCE TUNING & ANTI-PATTERNS
// ============================================================================

#region Performance Tuning

/*
 * PERFORMANCE CHECKLIST:
 *
 *   ✓ Every query uses an index (check explain plan: no COLLSCAN in production)
 *   ✓ Covered queries where possible (all fields in index)
 *   ✓ Projection to reduce document size transferred over wire
 *   ✓ Cursor-based pagination instead of large skip values
 *   ✓ $match and $sort FIRST in aggregation pipelines
 *   ✓ MongoClient is singleton (connection pool reuse)
 *   ✓ ReadPreference.Secondary for reporting queries (offload primary)
 *   ✓ allowDiskUse: true for large aggregations
 *   ✓ Batch inserts with InsertMany instead of loop of InsertOne
 *   ✓ BulkWrite for mixed write operations
 *   ✓ Avoid large arrays in documents (Unbounded Array anti-pattern)
 *   ✓ Use .CountDocuments() for indexed counts; avoid .Find().Count() (full scan)
 *
 * COMMON ANTI-PATTERNS:
 *   1. Creating a new MongoClient per request → destroys connection pool.
 *   2. Unbounded arrays (ever-growing comments/events on a document).
 *   3. Using $where (JavaScript in query) → cannot use indexes, slow.
 *   4. Large $in arrays (hundreds of values) → consider a different data model.
 *   5. Fetching entire documents when only 2 fields are needed → always project.
 *   6. Using skip with large values for pagination → use cursor-based.
 *   7. Running $lookup on hot OLTP paths → embed or denormalize.
 *   8. Storing ObjectId as string instead of ObjectId BSON type → no index efficiency.
 */

public class PerformanceBestPractices
{
    private readonly IMongoCollection<Order> _orders;

    public PerformanceBestPractices(IMongoDatabase db)
        => _orders = db.GetCollection<Order>("orders");

    // GOOD: CountDocuments uses index
    public async Task<long> CountPendingOrdersAsync()
        => await _orders.CountDocumentsAsync(o => o.Status == OrderStatus.Pending);

    // GOOD: EstimatedDocumentCount — O(1), uses collection metadata (no filter)
    public async Task<long> EstimateTotalOrdersAsync()
        => await _orders.EstimatedDocumentCountAsync();

    // GOOD: BulkWrite for batch upserts
    public async Task BulkUpsertOrdersAsync(IEnumerable<Order> orders)
    {
        var writes = orders.Select(o => new ReplaceOneModel<Order>(
            Builders<Order>.Filter.Eq(x => x.Id, o.Id), o)
            { IsUpsert = true })
            .Cast<WriteModel<Order>>()
            .ToList();

        await _orders.BulkWriteAsync(writes, new BulkWriteOptions { IsOrdered = false });
    }

    // GOOD: hint to force a specific index (when query planner chooses wrong one)
    public async Task<List<Order>> FindWithIndexHintAsync(string customerId)
    {
        return await _orders
            .Find(o => o.Customer.CustomerId == customerId)
            .Hint(Builders<Order>.IndexKeys
                .Ascending(o => o.Customer.CustomerId)
                .Descending(o => o.CreatedAt))
            .ToListAsync();
    }
}

#endregion

// ============================================================================
// SECTION 16: SECURITY — AUTH, TLS & FIELD-LEVEL ENCRYPTION
// ============================================================================

#region Security

/*
 * AUTHENTICATION:
 *   SCRAM-SHA-256 (default, recommended)
 *   X.509 certificates (mTLS — strong for microservices)
 *   LDAP / Kerberos (enterprise)
 *   AWS IAM / GCP / Azure (Atlas cloud providers)
 *
 * AUTHORIZATION:
 *   Built-in roles: read, readWrite, dbAdmin, clusterAdmin, root
 *   Custom roles: fine-grained per collection/action
 *   Principle of least privilege: application user has readWrite on its own DB only
 *
 * ENCRYPTION:
 *   TLS in transit: ssl=true in connection string
 *   Encryption at rest: WiredTiger encryption (MongoDB Enterprise / Atlas)
 *   Client-Side Field Level Encryption (CSFLE):
 *     - Encrypt specific fields BEFORE sending to the server.
 *     - Server never sees plaintext of encrypted fields.
 *     - Use for PII: SSN, credit card, health data.
 *     - MongoDB Atlas has Automatic Encryption with key management.
 *
 * NETWORK:
 *   - Never expose MongoDB port (27017) to the internet.
 *   - Use VPC peering or Private Link (Atlas).
 *   - IP allowlist (Atlas) or firewall rules.
 *   - Disable bindIp: 0.0.0.0 in production — bind to specific interfaces only.
 *
 * AUDIT:
 *   - Enable audit logging in MongoDB Enterprise / Atlas.
 *   - Audit: authentication events, authorization failures, schema changes.
 */

public class SecureMongoClientFactory
{
    // TLS + SCRAM authentication
    public static IMongoClient CreateSecureClient(string connectionString, string tlsCertPath)
    {
        var settings = MongoClientSettings.FromConnectionString(connectionString);

        settings.SslSettings = new SslSettings
        {
            EnabledSslProtocols = System.Security.Authentication.SslProtocols.Tls12
                                | System.Security.Authentication.SslProtocols.Tls13,
            // For mTLS (client certificate authentication):
            // ClientCertificates = new[] { X509Certificate.CreateFromCertFile(tlsCertPath) }
        };
        settings.UseTls = true;

        // Connection string format with TLS:
        // mongodb+srv://user:pass@cluster.mongodb.net/db?tls=true&authSource=admin

        return new MongoClient(settings);
    }

    // CSFLE — Client-Side Field Level Encryption (auto-encryption)
    // Requires MongoDB Enterprise or Atlas + mongocryptd process or shared library
    public static IMongoClient CreateEncryptedClient(string connectionString, byte[] masterKey)
    {
        var keyVaultNamespace = CollectionNamespace.FromFullName("encryption.__keyVault");

        var kmsProviders = new Dictionary<string, IReadOnlyDictionary<string, object>>
        {
            ["local"] = new Dictionary<string, object> { ["key"] = masterKey }
            // Production: use AWS KMS, Azure Key Vault, or GCP KMS instead of local key
        };

        var autoEncryptionOptions = new AutoEncryptionOptions(
            keyVaultNamespace: keyVaultNamespace,
            kmsProviders:      kmsProviders,
            bypassAutoEncryption: false);

        var settings = MongoClientSettings.FromConnectionString(connectionString);
        settings.AutoEncryptionOptions = autoEncryptionOptions;

        return new MongoClient(settings);
        // Encrypted fields are transparent to application code — driver handles encrypt/decrypt
    }
}

#endregion

// ============================================================================
// SECTION 17: INTERVIEW Q&A QUICK REFERENCE
// ============================================================================

#region Interview Q&A

/*
 * ============================================================================
 * MONGODB INTERVIEW QUESTIONS — QUICK REFERENCE (8 YOE LEVEL)
 * ============================================================================
 *
 * CORE CONCEPTS
 * -------------
 * Q1: What is BSON and why does MongoDB use it instead of JSON?
 *     - Binary JSON: faster to parse (length-prefixed), richer type system (Date, ObjectId,
 *       Decimal128, Binary), more compact for numeric data.
 *     - JSON is text; BSON is binary → faster for storage and wire transfer.
 *
 * Q2: Is MongoDB schema-less?
 *     - Semi-schema-less. Collections don't enforce schema by default,
 *       but you CAN add JSON Schema validation at the collection level.
 *     - At scale, treat it as "schema-flexible" not "schema-free" — drifting schemas
 *       are a maintenance nightmare. Use convention packs + class maps for consistency.
 *
 * Q3: What is ObjectId and when is it generated?
 *     - 12-byte _id: 4B timestamp + 5B random + 3B counter.
 *     - Generated CLIENT-SIDE by the driver (no server round-trip needed).
 *     - Monotonically increasing within a second; sortable by insertion time.
 *     - NEVER use as a sole shard key (hotspot). Use hashed or compound key.
 *
 * Q4: Are single-document operations atomic in MongoDB?
 *     - YES. Any write to a single document (including embedded arrays and
 *       sub-documents) is atomic at the document level.
 *     - This is the primary reason for embedding related data — eliminates
 *       the need for transactions in most OLTP cases.
 *
 * INDEXING
 * --------
 * Q5: What is the ESR rule for compound indexes?
 *     - Equality → Sort → Range.
 *     - Put exact-match fields first, then fields used for sort, then range fields last.
 *     - Violating this order can prevent the index from being used for sorting.
 *
 * Q6: What is a covered query?
 *     - A query where ALL fields in filter AND projection exist in the index.
 *     - MongoDB never fetches the actual document → fastest possible query.
 *     - Check with explain("executionStats"): totalDocsExamined = 0.
 *
 * Q7: What is a TTL index?
 *     - Index with ExpireAfter set on a Date field.
 *     - MongoDB background task checks every 60 seconds and deletes expired docs.
 *     - Not precise to the second (TTL monitor runs ~60s intervals).
 *     - Use for: sessions, OTP tokens, audit logs, temporary data.
 *
 * Q8: What is a partial index and why use it?
 *     - Index with a filter expression: only indexes matching documents.
 *     - Smaller index = faster writes + lower memory footprint.
 *     - Example: index only active products, not archived ones.
 *     - Query MUST include the partial filter to use the index.
 *
 * SCHEMA DESIGN
 * -------------
 * Q9: When do you embed vs reference in MongoDB?
 *     - EMBED: data always accessed together, 1:1 or 1:few, bounded size.
 *     - REFERENCE: independently accessed, 1:many where "many" is large,
 *       shared data, documents exceeding 16MB.
 *     - Rule of thumb: embed first; extract to a reference when documents
 *       grow too large or update patterns diverge.
 *
 * Q10: What is the Unbounded Array anti-pattern?
 *      - Appending to an embedded array indefinitely (comments, events, logs).
 *      - Problems: document grows toward 16MB limit, index grows proportionally,
 *        write amplification (MongoDB rewrites the whole doc on modification).
 *      - Solution: Bucket pattern (time-window documents), or reference to a
 *        separate collection with a parent reference.
 *
 * Q11: What is the Bucket Pattern?
 *      - Group time-series records into time-windowed documents (one per hour, day).
 *      - Pre-compute aggregates (count, sum) in the bucket document.
 *      - Fewer documents → fewer index entries → faster range queries.
 *      - Ideal for IoT telemetry, metrics, financial ticks.
 *
 * TRANSACTIONS & CONCURRENCY
 * ---------------------------
 * Q12: When do you use multi-document transactions?
 *      - When atomicity is required across multiple documents/collections
 *        that CAN'T be embedded in a single document.
 *      - Example: stock reservation + order creation.
 *      - Avoid for single-document writes (atomic already) and high-throughput paths
 *        (transactions add ~10-30% overhead + 60s max limit).
 *
 * Q13: How do you implement optimistic concurrency in MongoDB?
 *      - Add a version field (int) to the document.
 *      - Update filter includes: { _id: X, version: expectedVersion }.
 *      - Update includes: { $inc: { version: 1 } }.
 *      - ModifiedCount == 0 → concurrent modification detected → retry or fail.
 *
 * Q14: What read/write concern should you use in production?
 *      - WriteConcern: w:"majority", j:true → durable to disk on majority.
 *      - ReadConcern: "majority" for strong consistency.
 *      - ReadConcern: "local" acceptable for non-critical reads (lower latency).
 *      - ReadPreference: Secondary for read-heavy reporting (scale reads).
 *
 * CHANGE STREAMS & EVENTS
 * ------------------------
 * Q15: What are change streams and how are they different from polling?
 *      - Change streams watch MongoDB's oplog and push change events to the application.
 *      - Push-based (millisecond latency), no polling load on the DB.
 *      - Resume tokens enable crash recovery without replaying from the beginning.
 *      - Requires: replica set or sharded cluster (not standalone).
 *
 * Q16: What is a resume token and why is it critical?
 *      - Each change event has a _id (resume token).
 *      - Persist it (in Redis or MongoDB itself) after processing each event.
 *      - On restart, pass ResumeAfter=savedToken → resumes from last position.
 *      - Without it, you re-process all events from "now" → miss events during downtime.
 *
 * SHARDING
 * --------
 * Q17: Why is ObjectId a bad sole shard key?
 *      - ObjectId is monotonically increasing. All new inserts map to the same chunk
 *        (the highest range) → single shard receives all writes (insert hotspot).
 *      - Fix: hashed shard key on _id ({ _id: "hashed" }) for even distribution,
 *        or compound { tenantId: 1, _id: 1 } for tenant isolation + no hotspot.
 *
 * Q18: What is a jumbo chunk and how do you prevent it?
 *      - A chunk where all documents share the same shard key value (can't be split).
 *      - Balancer can't move it → one shard gets all data for that key value.
 *      - Prevention: choose high-cardinality shard keys. Fix: manually split with sh.splitAt().
 *
 * PERFORMANCE
 * -----------
 * Q19: How do you diagnose a slow MongoDB query?
 *      1. Enable slow query log: db.setProfilingLevel(1, { slowms: 100 }).
 *      2. Query system.profile collection.
 *      3. Run .explain("executionStats") on the slow query.
 *      4. Look for COLLSCAN (no index), high totalDocsExamined vs low nReturned.
 *      5. Add appropriate index or rewrite query to use existing index.
 *      6. Check for large skip values → convert to cursor-based pagination.
 *
 * Q20: What is the difference between countDocuments and estimatedDocumentCount?
 *      - countDocuments(filter): accurate, uses index if available, required for filtered count.
 *      - estimatedDocumentCount(): O(1), reads collection metadata, no filter support.
 *        Use for "how many docs in this collection?" without filter.
 *      - OLD: .count() is deprecated — don't mention it; it had correctness issues with sharding.
 *
 * ============================================================================
 * KEY NUMBERS TO REMEMBER
 * ============================================================================
 *
 *   Max document size:              16 MB
 *   GridFS chunk size (default):    255 KB
 *   TTL monitor interval:           ~60 seconds (not precise)
 *   Replica set min members:        3 (1 Primary + 2 Secondary) for HA
 *   Transactions max duration:      60 seconds (default)
 *   Max indexes per collection:     64
 *   Default connection pool size:   100
 *   ObjectId size:                  12 bytes (4B ts + 5B random + 3B counter)
 *   Aggregation in-memory limit:    100 MB (use allowDiskUse:true beyond this)
 *   WiredTiger default compression: Snappy
 *   max.in.flight (MongoClient):    100 (matches pool size)
 *
 * ============================================================================
 * COMMON MISCONCEPTIONS (INTERVIEW TRAPS)
 * ============================================================================
 *
 *   WRONG: "MongoDB is schema-less, so I don't need to design a schema."
 *   RIGHT: Schema design is MORE critical in MongoDB. Without thought, you get
 *          unpredictable shapes, missing indexes, and unbounded growth.
 *
 *   WRONG: "I should normalize data like in SQL."
 *   RIGHT: MongoDB favors denormalization for read performance. Model for your
 *          access patterns, not for storage purity.
 *
 *   WRONG: "Transactions in MongoDB are just like SQL transactions."
 *   RIGHT: MongoDB transactions exist but are heavier than SQL transactions.
 *          Most OLTP cases avoid them via single-document atomicity (embed data).
 *
 *   WRONG: "Read from secondaries for better performance."
 *   RIGHT: Secondary reads can return stale data. Use only where eventual consistency
 *          is acceptable (reports, analytics). NEVER for writes or consistent reads.
 *
 *   WRONG: "More indexes = better performance."
 *   RIGHT: Each index slows down writes (must update all indexes) and uses RAM.
 *          Only add indexes you need. Remove unused indexes (check $indexStats).
 *
 *   WRONG: "GridFS is the right choice for all file storage."
 *   RIGHT: For most production systems, S3/Blob/GCS is cheaper and more scalable.
 *          GridFS is appropriate when files are tightly coupled to MongoDB data.
 *
 *   WRONG: "MongoDB doesn't support joins."
 *   RIGHT: $lookup provides LEFT OUTER JOIN semantics. But it's expensive at scale —
 *          the design preference is to embed data to AVOID needing joins.
 *
 *   WRONG: "ObjectId is a good shard key because it's unique."
 *   RIGHT: ObjectId's monotonic nature creates write hotspots in sharded clusters.
 *          Use hashed sharding or compound shard keys.
 *
 * ============================================================================
 * Good luck with your interview!
 * ============================================================================
 */

#endregion
