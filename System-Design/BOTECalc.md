Back of the Envelope

Here are the core back-of-envelope formulas:

---

## Given Inputs

- **DAU** = Daily Active Users
- **Peak Users** = typically 2–3× average concurrent users
- **Avg request size** = payload in KB/MB (read or write)
- **Read:Write ratio** = e.g., 10:1, 100:1

---

## 1. Throughput (Requests Per Second)

```
Avg RPS = DAU × requests_per_user_per_day / 86,400

Peak RPS = Avg RPS × peak_multiplier   # peak_multiplier ≈ 2–3×
```

---

## 2. Bandwidth

```
Ingress (incoming) = Peak RPS × avg_write_request_size
Egress  (outgoing) = Peak RPS × avg_read_request_size × read_ratio / (read_ratio + write_ratio)
```

> **Rule of thumb:** If read:write = 10:1, then ~90% of traffic is reads (egress), ~10% is writes (ingress).
> 

---

## 3. Daily Storage

```
Daily Storage = DAU × write_requests_per_user_per_day × avg_object_size
              + metadata overhead (≈ 10–20% of raw data)
```

With replication:

```
Daily Storage (with replication) = Daily Storage × replication_factor   # typically 3×
```

---

## 4. Annual Storage

```
Annual Storage = Daily Storage × 365 × growth_buffer   # growth_buffer ≈ 1.2–1.5×
```

For retention policies:

```
Annual Storage = Daily Storage × retention_days   # if data expires
```

---

## 5. Number of Servers

**By RPS / CPU capacity:**

```
Servers = Peak RPS / RPS_per_server   # a typical app server handles 1K–10K RPS
```

**By memory:**

```
Servers = (DAU × memory_per_active_session) / RAM_per_server
```

**By storage:**

```
Servers = Total Storage / usable_disk_per_server   # e.g., 4TB usable per node
```

> Take the **max** of all three — that's your bottleneck.
> 

---

## Quick Reference Constants

| Thing | Value |
| --- | --- |
| Seconds in a day | 86,400 |
| KB → MB → GB | × 1,024 each |
| Replication factor | 3× |
| Peak multiplier | 2–3× |
| Annual growth buffer | 1.2–1.5× |
| Typical app server RPS | 1K–10K |
| Typical server RAM | 64–256 GB |
| Typical disk per node | 2–10 TB usable |

---

## Example (plug-and-play)

> 10M DAU, 5 requests/user/day, 50KB avg payload, 10:1 read:write
> 

```
Avg RPS     = 10M × 5 / 86,400       ≈ 578 RPS
Peak RPS    = 578 × 3                 ≈ 1,750 RPS

Ingress     = 1,750 × 0.1 × 50KB     ≈ 8.75 MB/s
Egress      = 1,750 × 0.9 × 50KB     ≈ 78.8 MB/s

Daily write = 10M × 0.5 × 50KB       = 250 GB/day
With 3× rep = 750 GB/day

Annual      = 750GB × 365 × 1.2      ≈ 328 TB/year

Servers(RPS)= 1,750 / 1,000          ≈ 2 app servers (+ redundancy → 4–6)
```

---

The key insight is: **storage is dominated by write volume × retention**, **bandwidth by read volume × payload size**, and **servers by your peak load ÷ per-server capacity**.
