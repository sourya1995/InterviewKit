-- ============================================================================
-- SQL INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
-- For 8+ Years SDE Experience | Microservices Architecture Focus
-- ============================================================================
-- Topics Covered:
--   1.  Core SQL Fundamentals & Query Optimization
--   2.  Indexes (Clustered, Non-Clustered, Covering, Partial, Composite)
--   3.  Joins, Subqueries & CTEs
--   4.  Window Functions
--   5.  Transactions, Isolation Levels & Locking
--   6.  Stored Procedures, Functions & Triggers
--   7.  Normalization & Schema Design
--   8.  Partitioning & Sharding (Microservices Scale)
--   9.  Database per Service Pattern & Cross-Service Querying
--   10. Saga Pattern & Distributed Transactions
--   11. CQRS (Command Query Responsibility Segregation)
--   12. Event Sourcing with SQL
--   13. Outbox Pattern (Reliable Messaging from SQL)
--   14. Optimistic vs Pessimistic Concurrency Control
--   15. Query Plans & EXPLAIN / Execution Plan Analysis
--   16. Deadlocks - Detection & Prevention
--   17. Replication: Master-Slave, Read Replicas
--   18. Connection Pooling in Microservices
--   19. Temporal Tables & Audit Logging
--   20. JSON in SQL (PostgreSQL / SQL Server)
-- ============================================================================


-- ============================================================================
-- SECTION 1: CORE SQL FUNDAMENTALS & QUERY OPTIMIZATION
-- ============================================================================

-- 1a. Basic query anatomy and WHERE clause short-circuiting
-- Filter on indexed columns first; SQL engines short-circuit AND chains.
SELECT
    o.order_id,
    o.customer_id,
    o.total_amount,
    o.status
FROM orders o
WHERE o.status       = 'PENDING'     -- indexed column → evaluated first
  AND o.created_at  >= '2024-01-01'  -- range scan on composite index
  AND o.total_amount >  100;

-- 1b. SELECT only what you need — avoid SELECT *
-- In microservices, DTOs are thin; fetching extra columns wastes I/O.
-- Bad:
-- SELECT * FROM orders;

-- Good:
SELECT order_id, status, total_amount FROM orders WHERE customer_id = 42;

-- 1c. EXISTS vs IN vs JOIN — performance comparison
-- EXISTS stops at first match (efficient for large outer tables)
SELECT customer_id, name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

-- IN works well for small static lists
SELECT order_id FROM orders WHERE status IN ('PENDING', 'PROCESSING');

-- NOT EXISTS is faster than NOT IN when NULLs may exist in subquery
SELECT customer_id FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

-- 1d. UNION vs UNION ALL
-- UNION ALL is faster — no deduplication step; use when duplicates are OK
SELECT order_id, 'WEB'    AS channel FROM web_orders
UNION ALL
SELECT order_id, 'MOBILE' AS channel FROM mobile_orders;

-- Use UNION only when deduplication is required
SELECT product_id FROM order_items
UNION
SELECT product_id FROM wishlist_items;


-- ============================================================================
-- SECTION 2: INDEXES
-- ============================================================================

-- 2a. Clustered Index
-- The physical sort order of the table. One per table (Primary Key by default).
-- SQL Server / MySQL:
CREATE TABLE orders (
    order_id     BIGINT       NOT NULL PRIMARY KEY,   -- clustered index
    customer_id  INT          NOT NULL,
    status       VARCHAR(20)  NOT NULL,
    created_at   DATETIME     NOT NULL,
    total_amount DECIMAL(18,2)NOT NULL
);

-- 2b. Non-Clustered Index
-- Separate structure pointing back to clustered index (bookmark lookup).
CREATE INDEX IX_orders_customer_status
    ON orders (customer_id, status);          -- composite non-clustered

-- 2c. Covering Index (include columns to avoid bookmark lookup)
-- Query: fetch order_id + total_amount for a customer's PENDING orders
-- Without covering index → index seek on (customer_id, status), then bookmark lookup for total_amount
-- With covering index → single index seek, no lookup
CREATE INDEX IX_orders_covering
    ON orders (customer_id, status)
    INCLUDE (order_id, total_amount, created_at);   -- SQL Server syntax

-- PostgreSQL equivalent:
-- CREATE INDEX ix_orders_covering ON orders(customer_id, status)
--     INCLUDE (order_id, total_amount);

-- 2d. Partial / Filtered Index (index only rows matching a predicate)
-- Greatly reduces index size; only PENDING orders are indexed
CREATE INDEX IX_orders_pending
    ON orders (created_at)
    WHERE status = 'PENDING';    -- PostgreSQL / SQL Server 2008+

-- 2e. Composite Index — column order matters!
-- Leading column must appear in WHERE clause for index to be used.
-- (customer_id, status) supports: WHERE customer_id = ?
--                                  WHERE customer_id = ? AND status = ?
-- Does NOT help:                   WHERE status = ?   (skips leading column)
CREATE INDEX IX_composite ON orders (customer_id, status, created_at);

-- 2f. Index on Expression / Function-Based Index (PostgreSQL)
CREATE INDEX IX_orders_lower_status ON orders (LOWER(status));
-- Query can now use: WHERE LOWER(status) = 'pending'

-- 2g. When NOT to index
-- • High-write, low-read tables (indexes slow INSERTs/UPDATEs)
-- • Low-cardinality columns (e.g., boolean flags — full scan often faster)
-- • Small tables (optimizer may choose full scan anyway)

-- 2h. Index Maintenance — fragmentation
-- SQL Server: rebuild if fragmentation > 30%, reorganize if 10–30%
ALTER INDEX IX_orders_customer_status ON orders REBUILD;
ALTER INDEX IX_orders_customer_status ON orders REORGANIZE;

-- PostgreSQL:
-- REINDEX INDEX ix_orders_covering;


-- ============================================================================
-- SECTION 3: JOINS, SUBQUERIES & CTEs
-- ============================================================================

-- 3a. INNER JOIN — only matching rows
SELECT o.order_id, c.name, o.total_amount
FROM   orders    o
JOIN   customers c ON c.customer_id = o.customer_id
WHERE  o.status = 'SHIPPED';

-- 3b. LEFT JOIN — all left rows, NULLs for unmatched right
SELECT c.customer_id, c.name, COUNT(o.order_id) AS order_count
FROM   customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name;

-- 3c. Self JOIN — useful for hierarchies
SELECT e.employee_id, e.name, m.name AS manager_name
FROM   employees e
LEFT JOIN employees m ON m.employee_id = e.manager_id;

-- 3d. CTE (Common Table Expression) — improves readability & reusability
WITH ranked_orders AS (
    SELECT
        order_id,
        customer_id,
        total_amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY total_amount DESC) AS rn
    FROM orders
)
SELECT order_id, customer_id, total_amount
FROM   ranked_orders
WHERE  rn = 1;    -- highest order per customer

-- 3e. Recursive CTE — for hierarchical/graph data
WITH RECURSIVE category_tree AS (
    -- Anchor: root categories
    SELECT category_id, name, parent_id, 0 AS depth
    FROM   categories
    WHERE  parent_id IS NULL

    UNION ALL

    -- Recursive: children
    SELECT c.category_id, c.name, c.parent_id, ct.depth + 1
    FROM   categories    c
    JOIN   category_tree ct ON ct.category_id = c.parent_id
)
SELECT * FROM category_tree ORDER BY depth, name;

-- 3f. Correlated Subquery (runs once per outer row — use carefully)
SELECT order_id, total_amount,
       (SELECT AVG(total_amount) FROM orders o2 WHERE o2.customer_id = o1.customer_id) AS customer_avg
FROM   orders o1;
-- Better alternative: use a window function (Section 4)


-- ============================================================================
-- SECTION 4: WINDOW FUNCTIONS
-- ============================================================================

-- Window functions do NOT collapse rows (unlike GROUP BY).
-- Syntax: function() OVER (PARTITION BY ... ORDER BY ... ROWS/RANGE ...)

-- 4a. ROW_NUMBER, RANK, DENSE_RANK
SELECT
    order_id,
    customer_id,
    total_amount,
    ROW_NUMBER()   OVER (PARTITION BY customer_id ORDER BY total_amount DESC) AS row_num,
    RANK()         OVER (PARTITION BY customer_id ORDER BY total_amount DESC) AS rnk,
    DENSE_RANK()   OVER (PARTITION BY customer_id ORDER BY total_amount DESC) AS dense_rnk
FROM orders;
-- RANK skips numbers after ties (1, 2, 2, 4)
-- DENSE_RANK does not skip (1, 2, 2, 3)

-- 4b. LAG / LEAD — access previous/next row without self-join
SELECT
    order_id,
    created_at,
    total_amount,
    LAG(total_amount,  1, 0) OVER (PARTITION BY customer_id ORDER BY created_at) AS prev_order_amount,
    LEAD(total_amount, 1, 0) OVER (PARTITION BY customer_id ORDER BY created_at) AS next_order_amount
FROM orders;

-- 4c. SUM / AVG as Running Totals
SELECT
    order_id,
    created_at,
    total_amount,
    SUM(total_amount) OVER (PARTITION BY customer_id ORDER BY created_at
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM orders;

-- 4d. NTILE — bucket rows into N groups (percentiles, quartiles)
SELECT
    order_id,
    total_amount,
    NTILE(4) OVER (ORDER BY total_amount) AS quartile
FROM orders;

-- 4e. FIRST_VALUE / LAST_VALUE
SELECT
    order_id,
    customer_id,
    total_amount,
    FIRST_VALUE(total_amount) OVER (PARTITION BY customer_id ORDER BY created_at) AS first_order_amount
FROM orders;


-- ============================================================================
-- SECTION 5: TRANSACTIONS, ISOLATION LEVELS & LOCKING
-- ============================================================================

-- 5a. ACID Properties
-- Atomicity  — all or nothing
-- Consistency — data moves between valid states
-- Isolation  — concurrent transactions don't interfere
-- Durability — committed data survives crashes

-- 5b. Basic Transaction
BEGIN TRANSACTION;

    UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
    UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;

    -- Validate no negative balance
    IF EXISTS (SELECT 1 FROM accounts WHERE balance < 0)
    BEGIN
        ROLLBACK TRANSACTION;
        THROW 50001, 'Insufficient funds', 1;
    END

COMMIT TRANSACTION;

-- 5c. Isolation Levels & Anomalies they prevent
-- ┌────────────────────────┬──────────────┬──────────────┬───────────────────┐
-- │ Isolation Level        │ Dirty Read   │ Non-Repeatable│ Phantom Read      │
-- ├────────────────────────┼──────────────┼──────────────┼───────────────────┤
-- │ READ UNCOMMITTED       │ Possible     │ Possible     │ Possible          │
-- │ READ COMMITTED (def.)  │ Prevented    │ Possible     │ Possible          │
-- │ REPEATABLE READ        │ Prevented    │ Prevented    │ Possible (MySQL)  │
-- │ SERIALIZABLE           │ Prevented    │ Prevented    │ Prevented         │
-- └────────────────────────┴──────────────┴──────────────┴───────────────────┘

SET TRANSACTION ISOLATION LEVEL READ COMMITTED;   -- SQL Server / PostgreSQL
BEGIN TRANSACTION;
    SELECT balance FROM accounts WHERE account_id = 1;
    -- ... business logic ...
COMMIT;

-- PostgreSQL SNAPSHOT ISOLATION (default in PG):
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
    -- All reads see snapshot from BEGIN; no non-repeatable reads
COMMIT;

-- 5d. Row-Level Locking
-- SELECT FOR UPDATE — acquire exclusive row lock (pessimistic)
BEGIN;
    SELECT * FROM inventory WHERE product_id = 10 FOR UPDATE;
    UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 10;
COMMIT;

-- SELECT FOR SHARE — shared read lock; block writers, allow readers
BEGIN;
    SELECT * FROM inventory WHERE product_id = 10 FOR SHARE;
COMMIT;

-- SKIP LOCKED — useful in microservices job queues
BEGIN;
    SELECT job_id, payload
    FROM   job_queue
    WHERE  status = 'PENDING'
    ORDER  BY created_at
    LIMIT  1
    FOR UPDATE SKIP LOCKED;   -- skip rows locked by other workers
COMMIT;


-- ============================================================================
-- SECTION 6: STORED PROCEDURES, FUNCTIONS & TRIGGERS
-- ============================================================================

-- 6a. Stored Procedure (SQL Server T-SQL)
CREATE OR ALTER PROCEDURE usp_PlaceOrder
    @customer_id  INT,
    @product_id   INT,
    @quantity     INT,
    @order_id     BIGINT OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        BEGIN TRANSACTION;

            DECLARE @price DECIMAL(18,2);
            SELECT @price = unit_price FROM products WHERE product_id = @product_id;

            INSERT INTO orders (customer_id, status, total_amount, created_at)
            VALUES (@customer_id, 'PENDING', @price * @quantity, GETUTCDATE());

            SET @order_id = SCOPE_IDENTITY();

            UPDATE inventory
            SET    quantity = quantity - @quantity
            WHERE  product_id = @product_id
              AND  quantity   >= @quantity;

            IF @@ROWCOUNT = 0
                THROW 50002, 'Insufficient stock', 1;

        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        THROW;
    END CATCH
END;

-- 6b. User-Defined Function (PostgreSQL)
CREATE OR REPLACE FUNCTION fn_customer_lifetime_value(p_customer_id INT)
RETURNS DECIMAL(18,2)
LANGUAGE SQL STABLE AS $$
    SELECT COALESCE(SUM(total_amount), 0)
    FROM   orders
    WHERE  customer_id = p_customer_id
      AND  status      = 'COMPLETED';
$$;

-- Usage:
SELECT customer_id, name, fn_customer_lifetime_value(customer_id) AS ltv
FROM   customers
ORDER  BY ltv DESC
LIMIT  10;

-- 6c. Trigger — auto-update audit log on order status change
CREATE OR REPLACE FUNCTION trg_order_status_audit()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status <> NEW.status THEN
        INSERT INTO order_audit_log (order_id, old_status, new_status, changed_at, changed_by)
        VALUES (NEW.order_id, OLD.status, NEW.status, NOW(), current_user);
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER order_status_change
AFTER UPDATE ON orders
FOR EACH ROW EXECUTE FUNCTION trg_order_status_audit();


-- ============================================================================
-- SECTION 7: NORMALIZATION & SCHEMA DESIGN
-- ============================================================================

-- 7a. Normalization Forms
-- 1NF: Atomic values, no repeating groups
-- 2NF: 1NF + no partial dependency on composite PK
-- 3NF: 2NF + no transitive dependencies (non-key → non-key)
-- BCNF: 3NF + every determinant is a candidate key

-- Denormalized (1NF violation — tags is a multi-value column):
-- products(product_id, name, tags)  -- tags = 'electronics,mobile,5G'

-- Normalized:
CREATE TABLE products (
    product_id   INT          PRIMARY KEY,
    name         VARCHAR(200) NOT NULL,
    category_id  INT          REFERENCES categories(category_id),
    unit_price   DECIMAL(18,2)NOT NULL
);

CREATE TABLE product_tags (
    product_id   INT REFERENCES products(product_id),
    tag          VARCHAR(50),
    PRIMARY KEY (product_id, tag)
);

-- 7b. Surrogate vs Natural Keys
-- Surrogate: system-generated (INT IDENTITY / BIGSERIAL / UUID)
-- Natural:   business meaning (SSN, email) — risky (can change, privacy issues)

-- 7c. UUID vs BIGINT as PK in Microservices
-- UUID: globally unique across services, no coordination needed, but
--       random UUIDs fragment clustered index (use UUIDv7 or ULID for sequential)
-- BIGINT IDENTITY: sequential, compact, but needs coordination across services

-- 7d. Soft Delete — common in microservices (preserve audit trail)
ALTER TABLE orders ADD COLUMN deleted_at TIMESTAMP NULL;

-- Query always filters deleted rows
CREATE VIEW v_active_orders AS
    SELECT * FROM orders WHERE deleted_at IS NULL;


-- ============================================================================
-- SECTION 8: PARTITIONING & SHARDING (MICROSERVICES SCALE)
-- ============================================================================

-- 8a. Table Partitioning (PostgreSQL range partitioning)
-- Splits one logical table into physical partitions; transparent to queries.
CREATE TABLE orders (
    order_id    BIGSERIAL,
    customer_id INT           NOT NULL,
    status      VARCHAR(20)   NOT NULL,
    created_at  TIMESTAMP     NOT NULL,
    total_amount DECIMAL(18,2)NOT NULL
) PARTITION BY RANGE (created_at);

-- Monthly partitions
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Partition pruning: WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31'
-- only scans orders_2024_01 — avoids scanning all partitions.

-- 8b. Hash Partitioning (distribute rows evenly by customer_id)
CREATE TABLE events (
    event_id    BIGSERIAL,
    customer_id INT NOT NULL,
    payload     JSONB
) PARTITION BY HASH (customer_id);

CREATE TABLE events_p0 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE events_p1 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE events_p2 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE events_p3 PARTITION OF events FOR VALUES FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- 8c. Sharding Strategy (application-level — microservices responsibility)
-- Shard key selection criteria:
--   - High cardinality (customer_id, tenant_id)
--   - Even distribution (avoid hot shards)
--   - Aligned with access patterns (don't shard across join boundaries)

-- Example: shard routing table (stored in a config service)
CREATE TABLE shard_map (
    tenant_id   INT PRIMARY KEY,
    shard_id    SMALLINT    NOT NULL,
    db_host     VARCHAR(100)NOT NULL,
    db_name     VARCHAR(100)NOT NULL
);

-- 8d. Cross-shard aggregation anti-pattern
-- Avoid: SELECT COUNT(*) FROM orders; (requires scatter-gather across all shards)
-- Better: maintain pre-aggregated counters per shard, aggregate in application layer


-- ============================================================================
-- SECTION 9: DATABASE PER SERVICE PATTERN & CROSS-SERVICE QUERYING
-- ============================================================================

-- 9a. Each microservice owns its own schema/database — no shared tables.
-- Order Service DB:   orders, order_items, order_status_history
-- Customer Service DB: customers, addresses, preferences
-- Inventory Service DB: products, inventory, reservations

-- 9b. Cross-service JOIN problem — handled via:
--   Option 1: API Composition (aggregate in application layer)
--   Option 2: Materialized/Replicated Read Model (CQRS — see Section 11)
--   Option 3: Event-driven data replication

-- 9c. Read-only replica of Customer data in Order Service (event-driven sync)
-- Order Service maintains a local denormalized snapshot of customers:
CREATE TABLE customer_snapshot (
    customer_id     INT          PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    email           VARCHAR(200) NOT NULL,
    tier            VARCHAR(20),
    synced_at       TIMESTAMP    NOT NULL DEFAULT NOW()
);
-- Populated/updated by consuming CustomerUpdated domain events from a message broker.

-- 9d. Avoiding distributed JOINs — embed essential foreign data
-- Instead of joining across services, store a snapshot at write time:
CREATE TABLE orders_v2 (
    order_id         BIGSERIAL    PRIMARY KEY,
    customer_id      INT          NOT NULL,
    customer_name    VARCHAR(200) NOT NULL,   -- denormalized snapshot
    customer_email   VARCHAR(200) NOT NULL,   -- denormalized snapshot
    status           VARCHAR(20)  NOT NULL,
    total_amount     DECIMAL(18,2)NOT NULL,
    created_at       TIMESTAMP    NOT NULL DEFAULT NOW()
);


-- ============================================================================
-- SECTION 10: SAGA PATTERN & DISTRIBUTED TRANSACTIONS
-- ============================================================================

-- 10a. Problem: No 2PC across microservices (distributed transactions are fragile)
-- Solution: Saga = sequence of local transactions with compensating transactions

-- Saga State Machine table (Choreography or Orchestration)
CREATE TABLE saga_instance (
    saga_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type       VARCHAR(100) NOT NULL,   -- 'PlaceOrder'
    current_step    VARCHAR(100) NOT NULL,
    status          VARCHAR(20)  NOT NULL,   -- RUNNING, COMPLETED, COMPENSATING, FAILED
    payload         JSONB        NOT NULL,
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE saga_step_log (
    step_log_id     BIGSERIAL    PRIMARY KEY,
    saga_id         UUID         NOT NULL REFERENCES saga_instance(saga_id),
    step_name       VARCHAR(100) NOT NULL,
    step_status     VARCHAR(20)  NOT NULL,   -- PENDING, COMPLETED, COMPENSATED, FAILED
    executed_at     TIMESTAMP    NOT NULL DEFAULT NOW(),
    error_detail    TEXT
);

-- Steps for PlaceOrder Saga:
-- 1. ReserveInventory    → compensate: ReleaseInventory
-- 2. ChargePayment       → compensate: RefundPayment
-- 3. ConfirmOrder        → compensate: CancelOrder
-- 4. SendNotification    → compensate: (none — best effort)

-- 10b. Update saga step atomically with local DB operation (Outbox pattern combined)
BEGIN;
    -- Local DB work
    UPDATE inventory SET reserved = reserved + 1 WHERE product_id = 10;

    -- Advance saga state
    UPDATE saga_instance
    SET    current_step = 'ChargePayment', updated_at = NOW()
    WHERE  saga_id = '...';

    INSERT INTO saga_step_log (saga_id, step_name, step_status)
    VALUES ('...', 'ReserveInventory', 'COMPLETED');
COMMIT;


-- ============================================================================
-- SECTION 11: CQRS (COMMAND QUERY RESPONSIBILITY SEGREGATION)
-- ============================================================================

-- 11a. Command side — normalized write model (master DB)
-- Receives writes: PlaceOrder, CancelOrder, UpdateStatus

-- 11b. Query side — denormalized read model (replica / separate DB / Redis)
-- Orders Dashboard Read Model: pre-joined, pre-aggregated
CREATE TABLE order_summary_view (
    order_id         BIGINT       PRIMARY KEY,
    customer_id      INT          NOT NULL,
    customer_name    VARCHAR(200) NOT NULL,
    customer_tier    VARCHAR(20),
    status           VARCHAR(20)  NOT NULL,
    item_count       INT          NOT NULL,
    total_amount     DECIMAL(18,2)NOT NULL,
    created_at       TIMESTAMP    NOT NULL,
    last_updated_at  TIMESTAMP    NOT NULL
);
-- Rebuilt/updated by consuming domain events (OrderPlaced, OrderStatusChanged, etc.)

-- 11c. Materialized View (DB-native CQRS, within same DB)
CREATE MATERIALIZED VIEW mv_customer_order_summary AS
    SELECT
        c.customer_id,
        c.name,
        COUNT(o.order_id)       AS total_orders,
        SUM(o.total_amount)     AS lifetime_value,
        MAX(o.created_at)       AS last_order_date
    FROM customers c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'COMPLETED' OR o.status IS NULL
    GROUP BY c.customer_id, c.name;

-- Refresh strategy: CONCURRENTLY avoids locking the view during refresh
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_customer_order_summary;


-- ============================================================================
-- SECTION 12: EVENT SOURCING WITH SQL
-- ============================================================================

-- 12a. Event Store table — immutable append-only log of domain events
CREATE TABLE event_store (
    event_id        BIGSERIAL    PRIMARY KEY,
    aggregate_type  VARCHAR(100) NOT NULL,   -- 'Order'
    aggregate_id    VARCHAR(100) NOT NULL,   -- order UUID
    event_type      VARCHAR(100) NOT NULL,   -- 'OrderPlaced', 'OrderShipped'
    event_version   INT          NOT NULL,
    payload         JSONB        NOT NULL,
    metadata        JSONB,                   -- correlation_id, user_id, etc.
    occurred_at     TIMESTAMP    NOT NULL DEFAULT NOW(),
    UNIQUE (aggregate_id, event_version)     -- optimistic concurrency guard
);

CREATE INDEX IX_event_store_aggregate ON event_store (aggregate_type, aggregate_id, event_version);

-- 12b. Append event with optimistic concurrency check
INSERT INTO event_store (aggregate_type, aggregate_id, event_type, event_version, payload)
VALUES ('Order', 'order-abc-123', 'OrderPlaced', 1,
        '{"customer_id": 42, "total_amount": 299.99}');
-- If event_version = 1 already exists → UNIQUE violation → concurrency conflict

-- 12c. Reconstruct aggregate state by replaying events
SELECT event_type, payload, occurred_at
FROM   event_store
WHERE  aggregate_type = 'Order'
  AND  aggregate_id   = 'order-abc-123'
ORDER  BY event_version;
-- Application layer folds events into current state

-- 12d. Snapshot optimization (avoid replaying 10,000 events)
CREATE TABLE aggregate_snapshot (
    aggregate_type  VARCHAR(100) NOT NULL,
    aggregate_id    VARCHAR(100) NOT NULL,
    snapshot_version INT         NOT NULL,
    state           JSONB        NOT NULL,
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (aggregate_type, aggregate_id)
);
-- Load latest snapshot, then replay only events after snapshot_version


-- ============================================================================
-- SECTION 13: OUTBOX PATTERN (RELIABLE MESSAGING FROM SQL)
-- ============================================================================

-- 13a. Problem: After committing a DB transaction, publishing to message broker
--             can fail → inconsistency (DB updated, event not published)
-- Solution: Transactional Outbox — write event to outbox table in SAME transaction

CREATE TABLE outbox_events (
    outbox_id       BIGSERIAL    PRIMARY KEY,
    aggregate_type  VARCHAR(100) NOT NULL,
    aggregate_id    VARCHAR(100) NOT NULL,
    event_type      VARCHAR(100) NOT NULL,
    payload         JSONB        NOT NULL,
    topic           VARCHAR(200) NOT NULL,   -- Kafka topic / RabbitMQ exchange
    status          VARCHAR(20)  NOT NULL DEFAULT 'PENDING',  -- PENDING, PUBLISHED, FAILED
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    published_at    TIMESTAMP
);

CREATE INDEX IX_outbox_pending ON outbox_events (status, created_at)
    WHERE status = 'PENDING';

-- 13b. Write to outbox atomically with business operation
BEGIN;
    -- Business operation
    INSERT INTO orders (customer_id, status, total_amount, created_at)
    VALUES (42, 'PENDING', 299.99, NOW())
    RETURNING order_id;  -- use in next statement

    -- Outbox event (same transaction)
    INSERT INTO outbox_events (aggregate_type, aggregate_id, event_type, payload, topic)
    VALUES ('Order', '12345', 'OrderPlaced',
            '{"order_id": 12345, "customer_id": 42, "total": 299.99}',
            'orders.events');
COMMIT;
-- If commit fails → both roll back; if commit succeeds → both persisted

-- 13c. Outbox Relay / Polling Publisher (background job)
-- SELECT * FROM outbox_events WHERE status = 'PENDING' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 100;
-- Publish to broker → UPDATE outbox_events SET status = 'PUBLISHED', published_at = NOW() WHERE outbox_id IN (...);

-- 13d. CDC (Change Data Capture) alternative
-- Tools like Debezium read Postgres WAL / MySQL binlog → stream outbox events to Kafka
-- No polling needed; near real-time; minimal DB overhead


-- ============================================================================
-- SECTION 14: OPTIMISTIC vs PESSIMISTIC CONCURRENCY CONTROL
-- ============================================================================

-- 14a. Optimistic Concurrency — no locks; use version column to detect conflicts
CREATE TABLE products (
    product_id   INT          PRIMARY KEY,
    name         VARCHAR(200) NOT NULL,
    unit_price   DECIMAL(18,2)NOT NULL,
    row_version  INT          NOT NULL DEFAULT 0   -- incremented on every UPDATE
);

-- Read
SELECT product_id, name, unit_price, row_version FROM products WHERE product_id = 10;
-- (returns row_version = 5)

-- Update — fails if another process updated in between
UPDATE products
SET    unit_price  = 99.99,
       row_version = row_version + 1
WHERE  product_id  = 10
  AND  row_version = 5;   -- optimistic lock check

-- @@ROWCOUNT = 0 (SQL Server) or affected_rows = 0 (PostgreSQL) → conflict detected
-- Application retries or returns 409 Conflict to caller

-- SQL Server native: rowversion / timestamp column
-- ALTER TABLE products ADD row_ts ROWVERSION; -- auto-updated by DB engine

-- 14b. Pessimistic Concurrency — lock first, then update
BEGIN;
    SELECT * FROM products WHERE product_id = 10 FOR UPDATE;  -- lock acquired
    -- No other transaction can UPDATE this row until COMMIT/ROLLBACK
    UPDATE products SET unit_price = 99.99 WHERE product_id = 10;
COMMIT;
-- Higher contention; suitable for high-conflict scenarios (bank transfers)


-- ============================================================================
-- SECTION 15: QUERY PLANS & EXECUTION PLAN ANALYSIS
-- ============================================================================

-- 15a. EXPLAIN / EXPLAIN ANALYZE (PostgreSQL)
EXPLAIN ANALYZE
SELECT o.order_id, c.name, o.total_amount
FROM   orders o
JOIN   customers c ON c.customer_id = o.customer_id
WHERE  o.status    = 'PENDING'
  AND  o.created_at > NOW() - INTERVAL '7 days';

-- Key plan nodes to understand:
-- Seq Scan       — full table scan (may be acceptable for small tables)
-- Index Scan     — seeks index, fetches heap row
-- Index Only Scan— covering index; no heap access (fastest)
-- Nested Loop    — good for small inner tables
-- Hash Join      — good for large unsorted tables
-- Merge Join     — good for pre-sorted inputs

-- 15b. SQL Server Execution Plan operators:
-- Table Scan       → missing index (bad)
-- Clustered Index Seek → good
-- Key Lookup       → index used but needs extra columns → add INCLUDE columns
-- Hash Match       → large joins/aggregations (memory spill risk)
-- Sort             → ORDER BY without supporting index (expensive)
-- Spool            → intermediate result stored (often indicates bad plan)

-- 15c. Statistics — optimizer relies on statistics to choose plan
UPDATE STATISTICS orders;                    -- SQL Server
ANALYZE orders;                              -- PostgreSQL
-- Stale statistics → bad cardinality estimates → wrong join order / plan

-- 15d. Query Hints (last resort — use sparingly)
-- SQL Server:
SELECT * FROM orders WITH (NOLOCK);          -- dirty read; avoids lock contention in reporting
SELECT * FROM orders WITH (INDEX(IX_orders_customer_status));  -- force index

-- PostgreSQL:
SET enable_seqscan = OFF;                    -- force planner to use index (session-level)
SET enable_hashjoin = OFF;


-- ============================================================================
-- SECTION 16: DEADLOCKS — DETECTION & PREVENTION
-- ============================================================================

-- 16a. Deadlock scenario
-- Transaction A: locks orders row 1, wants inventory row 1
-- Transaction B: locks inventory row 1, wants orders row 1
-- → Deadlock! DB kills one transaction (victim)

-- 16b. Prevention strategies:
-- 1. Always acquire locks in the same order (canonical ordering)
-- 2. Keep transactions short — minimize lock hold time
-- 3. Use optimistic concurrency to avoid locks altogether
-- 4. Use READ COMMITTED with row versioning (MVCC in PostgreSQL)

-- 16c. Deadlock detection (SQL Server trace / PostgreSQL log)
-- PostgreSQL: SET deadlock_timeout = '1s'; -- default 1s; adjust in postgresql.conf
-- SQL Server: Use SQL Server Profiler or Extended Events to capture deadlock graphs

-- 16d. Retry logic on deadlock (application layer pattern)
-- On catching deadlock error (SQL Server error 1205 / PostgreSQL P0001):
--   1. Wait brief random back-off (50–200ms)
--   2. Retry transaction up to N times
--   3. Log and alert if retries exhausted

-- 16e. Viewing current locks (PostgreSQL)
SELECT pid, relation::regclass, mode, granted
FROM   pg_locks
WHERE  NOT granted;   -- show waiting locks

-- SQL Server:
-- SELECT * FROM sys.dm_exec_requests WHERE blocking_session_id > 0;


-- ============================================================================
-- SECTION 17: REPLICATION — MASTER-SLAVE / READ REPLICAS
-- ============================================================================

-- 17a. Replication Lag Considerations in Microservices
-- After a write to primary, replica may lag 10ms–1s+
-- Read-your-own-writes problem: user writes → immediately reads → hits stale replica

-- Solutions:
-- 1. Route reads that need consistency to primary (e.g., just after write)
-- 2. Use session-sticky routing for read-after-write (same replica per user)
-- 3. Sync replication (PostgreSQL synchronous_standby_names) — latency cost
-- 4. Wait for replica to catch up (check replication lag before reading)

-- 17b. PostgreSQL replication lag check
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    (sent_lsn - replay_lsn) AS replication_lag_bytes
FROM pg_stat_replication;

-- 17c. Read Replica routing in microservices
-- Commands (writes) → primary connection string
-- Queries (reads)   → read replica connection string
-- Services inject the correct connection string based on operation type

-- 17d. Failover
-- Automatic failover tools: Patroni (PostgreSQL), MHA (MySQL), AWS RDS Multi-AZ
-- Application must handle brief unavailability via retry + circuit breaker


-- ============================================================================
-- SECTION 18: CONNECTION POOLING IN MICROSERVICES
-- ============================================================================

-- 18a. Problem: Each microservice pod opens DB connections.
-- 100 pods × 10 connections = 1000 connections → DB overwhelmed

-- 18b. PgBouncer (PostgreSQL connection pooler) — sits between service and DB
-- Transaction mode pooling: connection released after each transaction (most efficient)
-- Session mode: connection held for entire client session
-- Statement mode: released after each statement (limited compatibility)

-- 18c. Connection pool settings (per service)
-- min_connections: 2–5 (idle connections to keep warm)
-- max_connections: 10–20 per pod (balance throughput vs DB capacity)
-- max_idle_time: 10 min (release idle connections)
-- connection_timeout: 5s (fail fast if pool exhausted)

-- 18d. Monitoring connection health
-- PostgreSQL: active connection count
SELECT count(*), state
FROM   pg_stat_activity
WHERE  datname = 'orders_db'
GROUP  BY state;

-- 18e. Pgpool-II vs PgBouncer
-- PgBouncer: lightweight, fast, transaction pooling, no query routing
-- Pgpool-II:  load balancing, replication, query caching, heavier

-- 18f. HikariCP (Java / Spring Boot) — fastest JDBC connection pool
-- hikari.maximum-pool-size=20
-- hikari.minimum-idle=5
-- hikari.connection-timeout=30000
-- hikari.idle-timeout=600000
-- hikari.max-lifetime=1800000


-- ============================================================================
-- SECTION 19: TEMPORAL TABLES & AUDIT LOGGING
-- ============================================================================

-- 19a. System-Versioned Temporal Tables (SQL Server 2016+)
CREATE TABLE products (
    product_id   INT          PRIMARY KEY,
    name         VARCHAR(200) NOT NULL,
    unit_price   DECIMAL(18,2)NOT NULL,
    valid_from   DATETIME2    GENERATED ALWAYS AS ROW START,
    valid_to     DATETIME2    GENERATED ALWAYS AS ROW END,
    PERIOD FOR SYSTEM_TIME (valid_from, valid_to)
)
WITH (SYSTEM_VERSIONING = ON (HISTORY_TABLE = dbo.products_history));

-- Query as-of a point in time
SELECT * FROM products
FOR SYSTEM_TIME AS OF '2024-06-01T00:00:00';

-- Query what changed between two times
SELECT * FROM products
FOR SYSTEM_TIME BETWEEN '2024-01-01' AND '2024-06-01';

-- 19b. Manual Audit Log (database agnostic — works in all microservices)
CREATE TABLE audit_log (
    audit_id        BIGSERIAL    PRIMARY KEY,
    table_name      VARCHAR(100) NOT NULL,
    record_id       VARCHAR(100) NOT NULL,
    operation       CHAR(1)      NOT NULL,   -- I(nsert), U(pdate), D(elete)
    old_values      JSONB,
    new_values      JSONB,
    changed_by      VARCHAR(100),
    changed_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    correlation_id  UUID                    -- trace microservice request chain
);

-- 19c. Audit trigger (PostgreSQL)
CREATE OR REPLACE FUNCTION fn_audit_trigger()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO audit_log (table_name, record_id, operation, old_values, new_values, changed_by, correlation_id)
    VALUES (
        TG_TABLE_NAME,
        COALESCE(NEW.order_id::TEXT, OLD.order_id::TEXT),
        LEFT(TG_OP, 1),
        CASE WHEN TG_OP <> 'INSERT' THEN row_to_json(OLD)::JSONB END,
        CASE WHEN TG_OP <> 'DELETE' THEN row_to_json(NEW)::JSONB END,
        current_setting('app.current_user', TRUE),
        current_setting('app.correlation_id', TRUE)::UUID
    );
    RETURN COALESCE(NEW, OLD);
END;
$$;

CREATE TRIGGER orders_audit
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION fn_audit_trigger();

-- 19d. Set session-level context for audit (app sets before each operation)
-- PostgreSQL:
SET LOCAL app.current_user      = 'user-service:user_id=42';
SET LOCAL app.correlation_id    = 'a3f1d2c4-...-uuid';


-- ============================================================================
-- SECTION 20: JSON IN SQL (PostgreSQL & SQL Server)
-- ============================================================================

-- 20a. JSONB in PostgreSQL — binary JSON, indexed, efficient
CREATE TABLE service_events (
    event_id   BIGSERIAL PRIMARY KEY,
    service    VARCHAR(50) NOT NULL,
    payload    JSONB       NOT NULL,
    created_at TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- 20b. Insert JSON
INSERT INTO service_events (service, payload)
VALUES ('order-service', '{"order_id": 123, "items": [{"sku": "ABC", "qty": 2}], "amount": 49.99}');

-- 20c. Query JSON fields
SELECT
    payload->>'order_id'      AS order_id,
    payload->>'amount'        AS amount,
    payload->'items'->0->>'sku' AS first_sku
FROM   service_events
WHERE  service = 'order-service'
  AND  (payload->>'amount')::NUMERIC > 30;

-- 20d. GIN Index on JSONB — fast containment / key existence queries
CREATE INDEX IX_events_payload ON service_events USING GIN (payload);

-- Containment query — uses GIN index
SELECT * FROM service_events WHERE payload @> '{"service": "order-service"}';

-- Key exists operator
SELECT * FROM service_events WHERE payload ? 'order_id';

-- 20e. Update nested JSON (PostgreSQL)
UPDATE service_events
SET    payload = jsonb_set(payload, '{amount}', '59.99')
WHERE  event_id = 1;

-- 20f. JSON in SQL Server
SELECT
    JSON_VALUE(payload, '$.order_id')   AS order_id,
    JSON_VALUE(payload, '$.amount')     AS amount,
    JSON_QUERY(payload, '$.items')      AS items_array
FROM   service_events
WHERE  JSON_VALUE(payload, '$.amount') > 30;

-- 20g. Aggregate rows into JSON array
SELECT
    customer_id,
    JSON_AGG(
        JSON_BUILD_OBJECT('order_id', order_id, 'amount', total_amount)
        ORDER BY created_at
    ) AS orders_json
FROM   orders
GROUP  BY customer_id;


-- ============================================================================
-- BONUS: COMMON INTERVIEW PROBLEM PATTERNS
-- ============================================================================

-- B1. Find Nth highest salary / order amount (classic)
-- Using DENSE_RANK:
WITH ranked AS (
    SELECT total_amount, DENSE_RANK() OVER (ORDER BY total_amount DESC) AS dr
    FROM   orders
)
SELECT DISTINCT total_amount FROM ranked WHERE dr = 3;   -- 3rd highest

-- B2. Running total per group
SELECT
    customer_id,
    order_id,
    total_amount,
    SUM(total_amount) OVER (PARTITION BY customer_id ORDER BY created_at) AS running_total
FROM orders;

-- B3. Find duplicate orders (same customer + amount + day)
SELECT customer_id, CAST(created_at AS DATE) AS order_date, total_amount, COUNT(*) AS cnt
FROM   orders
GROUP  BY customer_id, CAST(created_at AS DATE), total_amount
HAVING COUNT(*) > 1;

-- B4. Delete duplicates, keep latest
WITH cte AS (
    SELECT order_id,
           ROW_NUMBER() OVER (PARTITION BY customer_id, total_amount, CAST(created_at AS DATE)
                              ORDER BY created_at DESC) AS rn
    FROM   orders
)
DELETE FROM orders WHERE order_id IN (SELECT order_id FROM cte WHERE rn > 1);

-- B5. Customers who placed orders every month in 2024
SELECT customer_id
FROM   orders
WHERE  created_at >= '2024-01-01' AND created_at < '2025-01-01'
GROUP  BY customer_id
HAVING COUNT(DISTINCT DATE_TRUNC('month', created_at)) = 12;

-- B6. Pivot — monthly revenue per year (conditional aggregation)
SELECT
    EXTRACT(YEAR FROM created_at) AS yr,
    SUM(CASE WHEN EXTRACT(MONTH FROM created_at) =  1 THEN total_amount ELSE 0 END) AS jan,
    SUM(CASE WHEN EXTRACT(MONTH FROM created_at) =  2 THEN total_amount ELSE 0 END) AS feb,
    SUM(CASE WHEN EXTRACT(MONTH FROM created_at) = 12 THEN total_amount ELSE 0 END) AS dec
FROM   orders
GROUP  BY yr
ORDER  BY yr;

-- B7. Gap detection — find missing order IDs in a sequence
SELECT t1.order_id + 1 AS gap_start
FROM   orders t1
WHERE  NOT EXISTS (SELECT 1 FROM orders t2 WHERE t2.order_id = t1.order_id + 1)
  AND  t1.order_id < (SELECT MAX(order_id) FROM orders);

-- B8. Median (no built-in in SQL Server < 2022)
SELECT DISTINCT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount) OVER () AS median_order_amount
FROM orders;

-- B9. Identify sessions (session-ization / event grouping with gaps)
-- Events more than 30 minutes apart = new session
WITH session_flags AS (
    SELECT
        customer_id,
        event_time,
        CASE WHEN event_time - LAG(event_time) OVER (PARTITION BY customer_id ORDER BY event_time)
                  > INTERVAL '30 minutes'
             THEN 1 ELSE 0 END AS new_session_flag
    FROM user_events
),
sessions AS (
    SELECT *,
           SUM(new_session_flag) OVER (PARTITION BY customer_id ORDER BY event_time) AS session_id
    FROM session_flags
)
SELECT customer_id, session_id, MIN(event_time) AS start, MAX(event_time) AS end
FROM   sessions
GROUP  BY customer_id, session_id;

-- B10. Upsert — INSERT or UPDATE atomically (common in microservices data sync)
-- PostgreSQL:
INSERT INTO customer_snapshot (customer_id, name, email, tier, synced_at)
VALUES (42, 'Alice', 'alice@example.com', 'GOLD', NOW())
ON CONFLICT (customer_id) DO UPDATE
    SET name      = EXCLUDED.name,
        email     = EXCLUDED.email,
        tier      = EXCLUDED.tier,
        synced_at = EXCLUDED.synced_at;

-- SQL Server:
MERGE customer_snapshot AS target
USING (SELECT 42 AS customer_id, 'Alice' AS name, 'alice@example.com' AS email, 'GOLD' AS tier) AS source
ON    target.customer_id = source.customer_id
WHEN MATCHED THEN
    UPDATE SET name = source.name, email = source.email, tier = source.tier
WHEN NOT MATCHED THEN
    INSERT (customer_id, name, email, tier, synced_at)
    VALUES (source.customer_id, source.name, source.email, source.tier, GETUTCDATE());

-- ============================================================================
-- SECTION 21: CLASSIC EMPLOYEE / DEPARTMENT PROBLEMS
-- ============================================================================
-- Schema used throughout this section:
--
--   employees(emp_id, name, dept_id, manager_id, salary, hire_date, job_title)
--   departments(dept_id, dept_name, location)
--   projects(project_id, project_name, budget)
--   emp_projects(emp_id, project_id, role, joined_date)
--
-- ============================================================================

-- Setup (run once to follow along)
CREATE TABLE departments (
    dept_id    INT          PRIMARY KEY,
    dept_name  VARCHAR(100) NOT NULL,
    location   VARCHAR(100)
);

CREATE TABLE employees (
    emp_id      INT          PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    dept_id     INT          REFERENCES departments(dept_id),
    manager_id  INT          REFERENCES employees(emp_id),   -- self-referencing
    salary      DECIMAL(12,2)NOT NULL,
    hire_date   DATE         NOT NULL,
    job_title   VARCHAR(100)
);

CREATE TABLE projects (
    project_id   INT          PRIMARY KEY,
    project_name VARCHAR(200) NOT NULL,
    budget       DECIMAL(14,2)
);

CREATE TABLE emp_projects (
    emp_id      INT REFERENCES employees(emp_id),
    project_id  INT REFERENCES projects(project_id),
    role        VARCHAR(100),
    joined_date DATE,
    PRIMARY KEY (emp_id, project_id)
);

-- ── C1. Second (Nth) Highest Salary ─────────────────────────────────────────
-- Method A: DENSE_RANK (handles ties correctly)
WITH ranked AS (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
    FROM   employees
)
SELECT DISTINCT salary FROM ranked WHERE dr = 2;

-- Method B: Subquery (classic, works everywhere)
SELECT MAX(salary) FROM employees
WHERE  salary < (SELECT MAX(salary) FROM employees);

-- Method C: OFFSET/FETCH (PostgreSQL / SQL Server 2012+)
SELECT DISTINCT salary FROM employees
ORDER BY salary DESC
OFFSET 1 ROW FETCH NEXT 1 ROW ONLY;   -- change OFFSET for Nth (0-based)

-- ── C2. Highest Salary Per Department ───────────────────────────────────────
SELECT dept_id, MAX(salary) AS max_salary
FROM   employees
GROUP  BY dept_id;

-- With employee name (avoid correlated subquery — use window function):
WITH ranked AS (
    SELECT e.name, e.dept_id, d.dept_name, e.salary,
           RANK() OVER (PARTITION BY e.dept_id ORDER BY e.salary DESC) AS rnk
    FROM   employees e
    JOIN   departments d ON d.dept_id = e.dept_id
)
SELECT name, dept_name, salary FROM ranked WHERE rnk = 1;

-- ── C3. Employees Earning More Than Their Manager ────────────────────────────
SELECT e.name AS employee, e.salary AS emp_salary,
       m.name AS manager,  m.salary AS mgr_salary
FROM   employees e
JOIN   employees m ON m.emp_id = e.manager_id
WHERE  e.salary > m.salary;

-- ── C4. Employees With No Manager (Top-Level / CEO) ─────────────────────────
SELECT emp_id, name, job_title FROM employees WHERE manager_id IS NULL;

-- ── C5. Full Management Hierarchy (Recursive CTE) ───────────────────────────
WITH RECURSIVE org_chart AS (
    -- Anchor: top-level employees
    SELECT emp_id, name, manager_id, 0 AS depth,
           name::TEXT AS hierarchy_path
    FROM   employees
    WHERE  manager_id IS NULL

    UNION ALL

    SELECT e.emp_id, e.name, e.manager_id, oc.depth + 1,
           oc.hierarchy_path || ' → ' || e.name
    FROM   employees e
    JOIN   org_chart oc ON oc.emp_id = e.manager_id
)
SELECT depth, hierarchy_path, name FROM org_chart ORDER BY hierarchy_path;

-- ── C6. Count of Employees & Avg Salary Per Department ──────────────────────
SELECT
    d.dept_name,
    COUNT(e.emp_id)     AS headcount,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    MIN(e.salary)       AS min_salary,
    MAX(e.salary)       AS max_salary
FROM   departments d
LEFT JOIN employees e ON e.dept_id = d.dept_id
GROUP  BY d.dept_id, d.dept_name
ORDER  BY avg_salary DESC;

-- ── C7. Departments With No Employees ───────────────────────────────────────
SELECT d.dept_id, d.dept_name
FROM   departments d
LEFT JOIN employees e ON e.dept_id = d.dept_id
WHERE  e.emp_id IS NULL;

-- ── C8. Employees Working on More Than One Project ───────────────────────────
SELECT ep.emp_id, e.name, COUNT(ep.project_id) AS project_count
FROM   emp_projects ep
JOIN   employees    e ON e.emp_id = ep.emp_id
GROUP  BY ep.emp_id, e.name
HAVING COUNT(ep.project_id) > 1
ORDER  BY project_count DESC;

-- ── C9. Employees NOT Assigned to Any Project ────────────────────────────────
SELECT e.emp_id, e.name
FROM   employees e
WHERE  NOT EXISTS (SELECT 1 FROM emp_projects ep WHERE ep.emp_id = e.emp_id);

-- ── C10. Top 3 Highest Paid Employees Per Department ────────────────────────
WITH ranked AS (
    SELECT e.name, d.dept_name, e.salary,
           DENSE_RANK() OVER (PARTITION BY e.dept_id ORDER BY e.salary DESC) AS dr
    FROM   employees e
    JOIN   departments d ON d.dept_id = e.dept_id
)
SELECT dept_name, name, salary FROM ranked WHERE dr <= 3 ORDER BY dept_name, dr;

-- ── C11. Salary Difference From Department Average ───────────────────────────
SELECT
    name,
    salary,
    dept_id,
    AVG(salary) OVER (PARTITION BY dept_id)             AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY dept_id)    AS diff_from_avg
FROM employees
ORDER BY dept_id, diff_from_avg DESC;

-- ── C12. Employees Hired in the Last 90 Days ────────────────────────────────
SELECT emp_id, name, hire_date
FROM   employees
WHERE  hire_date >= CURRENT_DATE - INTERVAL '90 days';

-- ── C13. Cumulative Salary Spend Per Department (running total) ──────────────
SELECT
    dept_id,
    emp_id,
    name,
    salary,
    SUM(salary) OVER (PARTITION BY dept_id ORDER BY hire_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_spend
FROM employees;

-- ── C14. Find All Subordinates of a Given Manager (recursive) ───────────────
WITH RECURSIVE subordinates AS (
    SELECT emp_id, name, manager_id, 0 AS level
    FROM   employees
    WHERE  emp_id = 5   -- <-- replace with target manager emp_id

    UNION ALL

    SELECT e.emp_id, e.name, e.manager_id, s.level + 1
    FROM   employees e
    JOIN   subordinates s ON s.emp_id = e.manager_id
)
SELECT level, emp_id, name FROM subordinates WHERE emp_id <> 5 ORDER BY level, name;

-- ── C15. Salary Percentile Rank Across Company ──────────────────────────────
SELECT
    name,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary)    AS pct_rank,   -- 0.0 to 1.0
    CUME_DIST()    OVER (ORDER BY salary)    AS cume_dist    -- fraction at or below
FROM employees;


-- ============================================================================
-- SECTION 22: STRING & DATE MANIPULATION PATTERNS
-- ============================================================================

-- ── S1. String Functions ────────────────────────────────────────────────────

-- Length, case, trim
SELECT
    name,
    LENGTH(name)          AS name_length,
    UPPER(name)           AS upper_name,
    LOWER(name)           AS lower_name,
    TRIM('  Alice  ')     AS trimmed,
    LTRIM(' Alice')       AS left_trimmed,
    RTRIM('Alice ')       AS right_trimmed
FROM employees;

-- Substring extraction
SELECT
    email,
    SUBSTRING(email FROM 1 FOR POSITION('@' IN email) - 1) AS username,   -- PostgreSQL
    SUBSTRING(email, CHARINDEX('@', email) + 1)             AS domain      -- SQL Server
FROM customers;

-- String concatenation
SELECT name || ' (' || job_title || ')' AS display_name FROM employees;   -- PostgreSQL
-- SQL Server: SELECT name + ' (' + job_title + ')' AS display_name
-- ANSI:       SELECT CONCAT(name, ' (', job_title, ')') AS display_name   -- both

-- Pattern matching
SELECT name FROM employees WHERE name LIKE 'A%';        -- starts with A
SELECT name FROM employees WHERE name LIKE '%son';      -- ends with son
SELECT name FROM employees WHERE name LIKE '%ar%';      -- contains ar
SELECT name FROM employees WHERE name SIMILAR TO '(A|B)%';  -- PostgreSQL regex-like

-- ILIKE (PostgreSQL case-insensitive LIKE)
SELECT name FROM employees WHERE name ILIKE 'alice%';

-- REPLACE and REGEXP_REPLACE
SELECT REPLACE(phone, '-', '')                              AS cleaned_phone FROM customers;
SELECT REGEXP_REPLACE(phone, '[^0-9]', '', 'g')             AS digits_only   FROM customers;

-- SPLIT_PART (PostgreSQL) — split on delimiter
SELECT SPLIT_PART('alice@example.com', '@', 2) AS domain;   -- 'example.com'

-- STRING_AGG — aggregate strings (like GROUP_CONCAT in MySQL)
SELECT dept_id, STRING_AGG(name, ', ' ORDER BY name) AS employee_list
FROM   employees
GROUP  BY dept_id;

-- SQL Server equivalent:
-- SELECT dept_id, STRING_AGG(name, ', ') WITHIN GROUP (ORDER BY name) AS employee_list

-- COALESCE and NULLIF — handle NULLs in strings
SELECT COALESCE(middle_name, 'N/A') AS middle_name FROM employees;
SELECT NULLIF(status, '')           AS status      FROM orders;      -- '' → NULL

-- LPAD / RPAD — pad strings
SELECT LPAD(emp_id::TEXT, 6, '0') AS padded_id FROM employees;  -- '000042'

-- ── S2. Date & Time Functions ────────────────────────────────────────────────

-- Current date/time
SELECT
    CURRENT_DATE        AS today,              -- date only
    CURRENT_TIMESTAMP   AS now_with_tz,        -- timestamp with timezone
    NOW()               AS now_pg,             -- PostgreSQL alias
    GETUTCDATE()        AS now_sqlserver;       -- SQL Server

-- Extract parts of a date
SELECT
    hire_date,
    EXTRACT(YEAR  FROM hire_date)  AS hire_year,
    EXTRACT(MONTH FROM hire_date)  AS hire_month,
    EXTRACT(DOW   FROM hire_date)  AS day_of_week,  -- 0=Sunday in PG
    DATE_PART('quarter', hire_date)AS quarter,
    TO_CHAR(hire_date, 'YYYY-MM')  AS year_month    -- PostgreSQL format
FROM employees;

-- SQL Server equivalents:
-- YEAR(hire_date), MONTH(hire_date), DAY(hire_date)
-- DATEPART(QUARTER, hire_date)
-- FORMAT(hire_date, 'yyyy-MM')

-- Date arithmetic
SELECT
    hire_date,
    hire_date + INTERVAL '90 days'              AS probation_end,    -- PG
    hire_date - INTERVAL '1 year'               AS one_year_ago,
    AGE(CURRENT_DATE, hire_date)                AS tenure,           -- PG: '3 years 2 months'
    DATEDIFF(DAY, hire_date, GETDATE())         AS days_employed     -- SQL Server
FROM employees;

-- Truncate to period boundary
SELECT
    DATE_TRUNC('month',   created_at)  AS month_start,   -- PG
    DATE_TRUNC('quarter', created_at)  AS quarter_start,
    DATE_TRUNC('week',    created_at)  AS week_start
FROM orders;

-- SQL Server:
-- DATEADD(MONTH, DATEDIFF(MONTH, 0, created_at), 0) AS month_start

-- ── S3. Classic Date-Based Interview Problems ────────────────────────────────

-- Employees hired in Q1 of any year
SELECT name, hire_date
FROM   employees
WHERE  EXTRACT(MONTH FROM hire_date) BETWEEN 1 AND 3;

-- Orders placed on weekends
SELECT order_id, created_at
FROM   orders
WHERE  EXTRACT(DOW FROM created_at) IN (0, 6);   -- 0=Sunday, 6=Saturday in PG

-- Monthly order counts for the past 12 months
SELECT
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*)                         AS order_count,
    SUM(total_amount)                AS revenue
FROM   orders
WHERE  created_at >= NOW() - INTERVAL '12 months'
GROUP  BY 1
ORDER  BY 1;

-- Day-over-day revenue change (LAG)
WITH daily AS (
    SELECT DATE_TRUNC('day', created_at) AS day, SUM(total_amount) AS revenue
    FROM   orders
    GROUP  BY 1
)
SELECT
    day,
    revenue,
    LAG(revenue) OVER (ORDER BY day)                              AS prev_day_revenue,
    revenue - LAG(revenue) OVER (ORDER BY day)                    AS absolute_change,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY day))
                / NULLIF(LAG(revenue) OVER (ORDER BY day), 0), 2) AS pct_change
FROM daily ORDER BY day;

-- Longest consecutive streak of daily orders per customer
WITH daily_orders AS (
    SELECT customer_id, DATE_TRUNC('day', created_at)::DATE AS order_day
    FROM   orders
    GROUP  BY 1, 2
),
grp AS (
    SELECT customer_id, order_day,
           order_day - ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_day)::INT AS grp_key
    FROM   daily_orders
)
SELECT customer_id, MIN(order_day) AS streak_start, MAX(order_day) AS streak_end,
       COUNT(*) AS streak_days
FROM   grp
GROUP  BY customer_id, grp_key
ORDER  BY streak_days DESC;

-- Format a duration in human-readable form
SELECT
    emp_id,
    name,
    hire_date,
    FLOOR(EXTRACT(YEAR FROM AGE(CURRENT_DATE, hire_date)))  AS years_of_service
FROM employees
ORDER BY years_of_service DESC;

-- ── S4. String Cleaning (common in microservices data ingestion) ─────────────

-- Normalize inconsistent phone numbers ingested from external services
SELECT
    raw_phone,
    REGEXP_REPLACE(TRIM(raw_phone), '[^0-9+]', '', 'g') AS normalized_phone
FROM   raw_customer_import;

-- Detect rows with non-ASCII / corrupted characters
SELECT * FROM raw_customer_import
WHERE  name ~ '[^\x00-\x7F]';   -- PostgreSQL POSIX regex

-- Case-insensitive deduplication (emails from different services)
SELECT LOWER(TRIM(email)) AS canonical_email, COUNT(*) AS cnt
FROM   customers
GROUP  BY 1
HAVING COUNT(*) > 1;


-- ============================================================================
-- SECTION 23: PERFORMANCE TUNING WALKTHROUGHS (SLOW → FIXED)
-- ============================================================================

-- ── P1. Function on Indexed Column Breaks Index Usage ───────────────────────
-- SLOW: wrapping a column in a function prevents index seek
SELECT * FROM orders WHERE YEAR(created_at) = 2024;          -- SQL Server
SELECT * FROM orders WHERE DATE_PART('year', created_at) = 2024; -- PostgreSQL
-- Optimizer cannot use index on created_at → full table scan

-- FIXED: use range predicate on the raw column
SELECT * FROM orders
WHERE  created_at >= '2024-01-01' AND created_at < '2025-01-01';
-- Optimizer uses index seek on created_at ✓

-- ── P2. Implicit Type Conversion Kills Index ─────────────────────────────────
-- Assume customer_id is INT, but passed as VARCHAR from application
-- SLOW: implicit cast → index unusable
SELECT * FROM orders WHERE customer_id = '42';   -- '42' is VARCHAR

-- FIXED: pass correct type
SELECT * FROM orders WHERE customer_id = 42;
-- Or cast to matching type explicitly if schema change isn't possible

-- ── P3. SELECT * With Many Columns Causes Unnecessary I/O ───────────────────
-- SLOW:
SELECT * FROM orders WHERE status = 'PENDING';
-- Fetches all columns → wide rows → more pages → slower

-- FIXED: project only needed columns
SELECT order_id, customer_id, total_amount, created_at
FROM   orders
WHERE  status = 'PENDING';
-- Add those 4 columns to a covering index to avoid heap lookup entirely:
CREATE INDEX IX_orders_pending_covering ON orders(status)
    INCLUDE (order_id, customer_id, total_amount, created_at)
    WHERE status = 'PENDING';   -- filtered/partial index

-- ── P4. N+1 Query Problem (Microservices ORM Anti-Pattern) ───────────────────
-- Application loop: for each customer → execute a separate query for orders
-- 100 customers → 101 queries!

-- SLOW (N+1 in application pseudo-code):
-- SELECT * FROM customers LIMIT 100;
-- for each customer: SELECT * FROM orders WHERE customer_id = ?

-- FIXED: single JOIN or batch IN query
SELECT c.customer_id, c.name, COUNT(o.order_id) AS order_count, SUM(o.total_amount) AS total_spent
FROM   customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP  BY c.customer_id, c.name
LIMIT  100;

-- Or batch: load customers first, then fetch orders for all IDs in one shot
SELECT order_id, customer_id, status, total_amount
FROM   orders
WHERE  customer_id IN (1, 2, 3, ..., 100);   -- pass all IDs

-- ── P5. Missing Index on Foreign Key — Join Performance ──────────────────────
-- SLOW: joining orders to customers without index on orders.customer_id
-- Optimizer does a nested loop with a full scan of orders for each customer row

EXPLAIN ANALYZE
SELECT c.name, o.total_amount FROM customers c JOIN orders o ON o.customer_id = c.customer_id;
-- Plan shows: Seq Scan on orders (bad)

-- FIXED: add index on the FK column
CREATE INDEX IX_orders_customer_id ON orders(customer_id);
-- Plan now: Index Scan on orders using IX_orders_customer_id ✓

-- ── P6. Correlated Subquery vs Window Function ────────────────────────────────
-- SLOW: correlated subquery executes once per row → O(n²) effectively
SELECT
    emp_id,
    salary,
    (SELECT AVG(salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id) AS dept_avg
FROM employees e1;

-- FIXED: window function — single pass over data
SELECT
    emp_id,
    salary,
    AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;

-- ── P7. OR in WHERE Clause Prevents Index Use ────────────────────────────────
-- SLOW: OR on different columns → optimizer may do a full scan
SELECT * FROM orders WHERE customer_id = 42 OR status = 'PENDING';

-- FIXED: UNION ALL (each branch uses its own index)
SELECT * FROM orders WHERE customer_id = 42
UNION ALL
SELECT * FROM orders WHERE status = 'PENDING' AND customer_id <> 42;
-- Each query independently uses its index ✓

-- Or use a composite index that covers both access patterns separately

-- ── P8. COUNT(*) vs COUNT(column) ────────────────────────────────────────────
-- COUNT(*): counts all rows including NULLs — fastest, uses any index
-- COUNT(col): counts non-NULL values — slightly slower (NULL check per row)
-- COUNT(DISTINCT col): requires deduplication — most expensive

SELECT
    COUNT(*)                AS total_rows,
    COUNT(manager_id)       AS rows_with_manager,     -- excludes NULLs
    COUNT(DISTINCT dept_id) AS distinct_departments
FROM employees;
-- Use COUNT(*) when you just need total rows — it's optimized by all engines

-- ── P9. Pagination — OFFSET/FETCH vs Keyset (Cursor) Pagination ───────────────
-- SLOW: OFFSET becomes expensive as offset grows (scans & discards rows)
SELECT order_id, created_at, total_amount
FROM   orders
ORDER  BY created_at DESC
OFFSET 10000 LIMIT 20;   -- scans 10,020 rows, returns 20

-- FIXED: Keyset (seek) pagination — O(log n), constant time regardless of page
-- Client sends the last seen (created_at, order_id) from previous page
SELECT order_id, created_at, total_amount
FROM   orders
WHERE  (created_at, order_id) < ('2024-06-01 12:00:00', 5500)   -- last page cursor
ORDER  BY created_at DESC, order_id DESC
LIMIT  20;
-- Uses composite index on (created_at, order_id) → pure index seek ✓

-- Index to support keyset pagination:
CREATE INDEX IX_orders_keyset ON orders(created_at DESC, order_id DESC);

-- ── P10. Locking / Blocking — Long-Running Transactions ──────────────────────
-- SLOW / Dangerous: holding a transaction open during external API calls
-- BEGIN;
--   SELECT * FROM orders WHERE order_id = 1 FOR UPDATE;   -- lock acquired
--   <--- call external payment API (500ms–5s) --->          -- lock held!
--   UPDATE orders SET status = 'PAID' WHERE order_id = 1;
-- COMMIT;
-- Other transactions needing this row are blocked for the entire API duration

-- FIXED: minimize lock scope — do all external work BEFORE acquiring lock
-- 1. Call payment API (no lock held)
-- 2. Only then open transaction and quickly update
BEGIN;
    UPDATE orders
    SET    status = 'PAID', payment_ref = 'PAY-XYZ'
    WHERE  order_id = 1
      AND  status   = 'PENDING';   -- idempotency guard
    IF @@ROWCOUNT = 0 THEN ROLLBACK; END IF;
COMMIT;
-- Lock held for microseconds, not seconds ✓

-- ── P11. Aggregate Without Index — GROUP BY Performance ──────────────────────
-- SLOW: aggregating on a non-indexed column forces full scan + sort
SELECT dept_id, SUM(salary) FROM employees GROUP BY dept_id;
-- Execution plan: Seq Scan → Sort → HashAggregate (memory pressure risk)

-- FIXED: index on the GROUP BY column
CREATE INDEX IX_employees_dept ON employees(dept_id);
-- Now optimizer can use index scan and avoid sorting ✓

-- For frequently-run aggregations: use a materialized view
CREATE MATERIALIZED VIEW mv_dept_salary_summary AS
SELECT dept_id, COUNT(*) AS headcount, SUM(salary) AS total_salary, AVG(salary) AS avg_salary
FROM   employees
GROUP  BY dept_id;

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dept_salary_summary;

-- ── P12. Identifying Slow Queries in Production ───────────────────────────────
-- PostgreSQL: pg_stat_statements (requires pg_stat_statements extension)
SELECT
    query,
    calls,
    ROUND(total_exec_time::NUMERIC / calls, 2) AS avg_ms,
    rows / calls                                AS avg_rows,
    stddev_exec_time
FROM   pg_stat_statements
ORDER  BY avg_ms DESC
LIMIT  20;

-- PostgreSQL: find currently running slow queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM   pg_stat_activity
WHERE  state = 'active'
  AND  now() - query_start > INTERVAL '5 seconds'
ORDER  BY duration DESC;

-- SQL Server: find top queries by CPU / elapsed time
SELECT TOP 20
    qs.total_elapsed_time / qs.execution_count     AS avg_elapsed_us,
    qs.total_worker_time  / qs.execution_count     AS avg_cpu_us,
    qs.execution_count,
    SUBSTRING(st.text, (qs.statement_start_offset / 2) + 1,
              ((CASE qs.statement_end_offset WHEN -1 THEN DATALENGTH(st.text)
                ELSE qs.statement_end_offset END - qs.statement_start_offset) / 2) + 1) AS query_text
FROM   sys.dm_exec_query_stats qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
ORDER  BY avg_elapsed_us DESC;

-- ── P13. Temp Tables vs CTEs vs Subqueries — When to Use What ────────────────
-- Subquery: inline, single use, no reuse, can be hard to read for complex logic
-- CTE:      named, readable, logically separated steps; NOT materialized by default
--           (optimizer may inline them — no guaranteed performance gain over subquery)
-- Temp Table: physically materialized, reusable, indexable, good for large intermediate
--             results used multiple times

-- CTE (readable, but optimizer may re-evaluate each reference)
WITH expensive_cte AS (
    SELECT customer_id, SUM(total_amount) AS ltv FROM orders GROUP BY customer_id
)
SELECT c.name, ec.ltv FROM customers c JOIN expensive_cte ec ON ec.customer_id = c.customer_id;

-- Temp table (materializes result once — better when CTE is referenced multiple times)
CREATE TEMP TABLE tmp_customer_ltv AS
SELECT customer_id, SUM(total_amount) AS ltv FROM orders GROUP BY customer_id;

CREATE INDEX IX_tmp_ltv ON tmp_customer_ltv(customer_id);   -- indexable!

SELECT c.name, t.ltv FROM customers c JOIN tmp_customer_ltv t ON t.customer_id = c.customer_id;
-- ...reuse tmp_customer_ltv in other queries...
DROP TABLE tmp_customer_ltv;

-- ============================================================================
-- END OF SQL INTERVIEW PREPARATION GUIDE
-- ============================================================================
