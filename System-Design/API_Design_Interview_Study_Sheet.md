# 🏆 API Design Interview Study Sheet

> All 17 APIs + Best Practices Reference. Each API includes: endpoints, auth, idempotency, special considerations, and error scenarios.

---

## Table of Contents

1. [Best Practices Reference](#best-practices-reference)
2. [Search Service](#search-service)
3. [File Service](#file-service)
4. [Comment Service](#comment-service)
5. [Pub-Sub Service](#pub-sub-service)
6. [YouTube Streaming API](#youtube-streaming-api)
7. [Facebook Messenger API](#facebook-messenger-api)
8. [Google Maps API](#google-maps-api)
9. [Chess API](#chess-api)
10. [Zoom API](#zoom-api)
11. [LeetCode API](#leetcode-api)
12. [Payment Gateway API](#payment-gateway-api)
13. [Twitter API](#twitter-api)
14. [Uber API](#uber-api)
15. [CamelCamelCamel API](#camelcamelcamel-api)
16. [Gaming API](#gaming-api)
17. [Evernote API](#evernote-api)
18. [Amazon IAM API](#amazon-iam-api)

---

# Best Practices Reference

## 🔒 Security

### Authentication
- OAuth 2.0 / OpenID Connect for delegated auth
- JWT (short-lived access tokens + refresh tokens)
- API Keys (server-to-server, never in client-side JS)
- mTLS for internal microservices

```
POST /auth/token  →  { access_token, expires_in, refresh_token }
Authorization: Bearer <JWT>
API-Key: sk-xxxx  ← in header, never query param
```

### Authorization
- RBAC (Role-Based) or ABAC (Attribute-Based)
- Least-privilege principle
- Scope claims in JWT (e.g. `read:files write:files`)
- Resource-level ownership checks (`userId == resource.ownerId`)

```
403 Forbidden   ← authenticated but not authorized
401 Unauthorized ← not authenticated
JWT payload: { sub, roles:["admin"], scope:"read:files" }
```

### Transport Security
- TLS 1.2+ (enforce, reject downgrade)
- HSTS header
- Certificate pinning for mobile clients
- Cipher suite allow-list

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
HTTP → 301 redirect to HTTPS
```

### Input Validation
- Validate all inputs (type, length, format, range)
- Allowlist not blocklist
- Parameterized queries (prevent SQLi)
- Sanitize HTML (prevent XSS)
- Validate Content-Type header

```json
{
  "error": "VALIDATION_ERROR",
  "field": "email",
  "message": "Invalid email format"
}
```

### Rate Limiting & DDoS
- Per-user, per-IP, per-endpoint limits
- Token bucket / sliding window algorithm
- Exponential backoff guidance in 429 response
- WAF + CDN for layer 3/4 protection

```
HTTP 429 Too Many Requests
Retry-After: 60
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1700000000
```

### Secrets Management
- Never log tokens, passwords, or PII
- Secrets in Vault / AWS Secrets Manager — not env vars in git
- Rotate keys automatically
- Mask PII in logs

```
❌ Bad:  GET /api?api_key=secret123
✅ Good: Authorization: Bearer <token>
```

---

## 🔁 Idempotency

### HTTP Method Semantics

| Method | Safe | Idempotent | Notes |
|--------|------|------------|-------|
| GET | ✅ | ✅ | No side effects |
| HEAD | ✅ | ✅ | No side effects |
| OPTIONS | ✅ | ✅ | No side effects |
| PUT | ❌ | ✅ | Same result if called N times |
| DELETE | ❌ | ✅ | Same result if called N times |
| POST | ❌ | ❌ | Use Idempotency-Key header |
| PATCH | ❌ | ⚠️ | Idempotent if designed carefully |

### Idempotency Keys
- Client generates UUID v4 per logical operation
- Server stores `(key → result)` for TTL (e.g. 24 hrs)
- Re-use returns cached response — no side effects
- Return `409` if key reused with different payload

```
POST /payments
Idempotency-Key: a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx
{ "amount": 100, "currency": "USD" }

→ 201 Created  (first call)
→ 200 OK       (replay — same response, no new charge)
```

### Conditional Requests (ETags)
- `ETag`: hash of resource version
- `If-Match`: only update if version matches (optimistic locking)
- `If-None-Match`: cache validation
- Prevents lost-update problem in concurrent edits

```
GET /files/123         →  ETag: "v3"
PUT /files/123
If-Match: "v3"
→ 200 OK                  (if v3 still current)
→ 412 Precondition Failed (if stale)
```

---

## 📄 Pagination & Filtering

### Cursor-Based Pagination
Preferred for large or live datasets. Cursor is an opaque encoded pointer (base64 of last_id + timestamp). Stable under concurrent inserts unlike offset.

```
GET /posts?limit=20&after=eyJpZCI6MTAwfQ==
→ {
    "data": [...],
    "next_cursor": "eyJpZCI6MTIwfQ==",
    "has_more": true
  }
```

### Offset Pagination
Simple; use for static or small datasets. Include `total_count` for UI pagination controls.

```
GET /users?page=2&per_page=50
→ { "data": [...], "total": 1200, "page": 2, "per_page": 50, "total_pages": 24 }
```

### Filtering & Sorting

```
GET /orders
  ?status=shipped
  &created_after=2024-01-01
  &sort_by=amount
  &order=desc
  &fields=id,amount,status     ← field selection reduces payload
```

---

## ❌ Error Handling

### HTTP Status Codes

| Range | Codes |
|-------|-------|
| 2xx Success | 200 OK, 201 Created, 202 Accepted, 204 No Content |
| 4xx Client Error | 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 422 Unprocessable, 429 Too Many Requests |
| 5xx Server Error | 500 Internal, 502 Bad Gateway, 503 Unavailable, 504 Timeout |

> ⚠️ Never return `200` with `{ success: false }` in the body — use correct HTTP semantics.

### Error Response Body Schema

```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Account balance too low",
    "request_id": "req_abc123",
    "details": [
      { "field": "amount", "issue": "exceeds balance" }
    ]
  }
}
```

---

## 🔢 Versioning & Compatibility

### Versioning Strategies

| Strategy | Example | Notes |
|----------|---------|-------|
| URI versioning | `/v1/users` | Most common, easy to route |
| Header versioning | `API-Version: 2024-01-01` | Clean URLs |
| Content-Type | `application/vnd.myapi.v2+json` | REST purist approach |
| Query param | `?version=2` | Least preferred |

```
Deprecation-Notice: sunset=2025-06-01
Sunset: Sat, 01 Jun 2025 00:00:00 GMT
```

### Backward Compatibility Rules
- ✅ Safe: add new optional fields
- ❌ Breaking: rename field, change type, remove field, change status code semantics
- Use feature flags before hard version bumps
- Deprecate with headers before removing

---

## ⚡ Performance & Caching

### HTTP Caching

```
Cache-Control: public, max-age=3600
ETag: "abc123"

Client sends:  If-None-Match: "abc123"
Server returns: 304 Not Modified  (no body — saves bandwidth)
```

### Async Long-Running Jobs

```
POST /exports     →  202 Accepted
{
  "job_id": "job_xyz",
  "status": "queued",
  "poll_url": "/jobs/job_xyz"
}

GET /jobs/job_xyz →  { "status": "completed", "result_url": "..." }
```

---

## 🏗️ REST Design Conventions

### Resource Naming

```
✅ POST   /users
✅ GET    /users/123
✅ PUT    /users/123
✅ DELETE /users/123
✅ GET    /users/123/orders
✅ POST   /users/123/activate   ← actions as sub-resources
❌ GET    /getUser?id=123
```

- Nouns not verbs: `/users` not `/getUsers`
- Plural collections: `/users`, `/orders`
- Lowercase kebab-case: `/payment-methods`

### Request / Response Design
- `Accept` / `Content-Type: application/json`
- Consistent field naming: camelCase or snake_case (pick one)
- ISO 8601 dates: `2024-01-15T10:30:00Z`
- IDs as strings (avoid JS integer overflow for large IDs)
- Enveloped list responses: `{ data: [], meta: {} }`

```json
{
  "data": {
    "id": "usr_123",
    "created_at": "2024-01-15T10:30:00Z",
    "status": "active"
  },
  "meta": { "request_id": "req_abc" }
}
```

---

# Search Service

> Core service for full-text and faceted search across entities. Used internally by most services.

**Entities:** Index, Document, Query, SearchResult, Suggestion

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/indices` | Create a new search index | 201 |
| POST | `/v1/indices/{index_id}/documents` | Bulk index documents | 202 |
| GET | `/v1/search` | Full-text + faceted search | 200 |
| GET | `/v1/suggest` | Autocomplete suggestions | 200 |
| DELETE | `/v1/indices/{index_id}/documents/{doc_id}` | Delete document | 204 |
| PUT | `/v1/indices/{index_id}/documents/{doc_id}` | Upsert document | 200 |

### Key Request / Response Examples

```
POST /v1/indices
Body: { index_id, name, schema }
Response: { index_id, name, status, created_at }

GET /v1/search?q=foo&index=products&filters=category:books&sort=price:asc&limit=20&cursor=xxx
Response: { results:[{id,score,fields}], facets:{}, next_cursor, total }

GET /v1/suggest?q=appl&index=products&limit=5
Response: { suggestions:[{text, score}] }

PUT /v1/indices/{index_id}/documents/{doc_id}
Body: { fields }
Response: { id, version }
```

## 🔒 Auth
API Key (internal) / JWT Bearer (external). Scopes: `search:read`, `search:write`, `search:admin`

## 🔁 Idempotency
PUT upsert is idempotent. POST bulk-index: use `Idempotency-Key` header → async job dedup.

## ⚙️ Special Considerations
- Cursor pagination (live index)
- ETag on index schema
- Async indexing returns `202` + `job_id`
- Rate-limit per index
- Circuit breaker on backend engine (Elasticsearch/OpenSearch)

## ❌ Error Scenarios
`400` invalid query syntax, `404` index not found, `413` document too large, `429` rate limit, `503` engine unavailable

---

# File Service

> Upload, store, retrieve, and manage files of any type. Supports multipart, resumable uploads, and signed URLs.

**Entities:** File, Folder, ShareLink, Version, UploadSession

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/files/upload-session` | Initiate resumable upload | 201 |
| PUT | `/v1/files/upload-session/{id}/parts/{n}` | Upload part | 200 |
| POST | `/v1/files/upload-session/{id}/complete` | Complete upload | 201 |
| GET | `/v1/files/{file_id}` | Get file metadata | 200 |
| GET | `/v1/files/{file_id}/download` | Download / signed URL | 200 |
| DELETE | `/v1/files/{file_id}` | Soft-delete file | 204 |
| POST | `/v1/files/{file_id}/share` | Create share link | 201 |
| GET | `/v1/folders/{folder_id}/files` | List files in folder | 200 |

### Key Request / Response Examples

```
POST /v1/files/upload-session
Body: { filename, size, mime_type, folder_id }
Response: { upload_id, upload_url, part_size, expires_at }

PUT /v1/files/upload-session/{id}/parts/1
Body: Binary chunk (octet-stream)
Response: { etag, part_number }

POST /v1/files/upload-session/{id}/complete
Body: { parts:[{n, etag}] }
Response: { file_id, name, size, url, version }

GET /v1/files/{file_id}/download?expires=3600
Response: { signed_url, expires_at }  OR  302 redirect

POST /v1/files/{file_id}/share
Body: { expires_at, permission }
Response: { link_id, url, expires_at }
```

## 🔒 Auth
JWT Bearer. Scopes: `files:read`, `files:write`, `files:delete`, `files:share`

## 🔁 Idempotency
Upload parts are idempotent by part number. Complete upload: `Idempotency-Key`. DELETE is idempotent.

## ⚙️ Special Considerations
- Resumable upload with part ETags (S3 multipart style)
- Signed download URLs (presigned, short TTL)
- Virus scan async after upload
- `Content-Disposition`, `Content-Type` on download headers
- Version history preserved
- Soft delete → purge after 30 days
- Max file size enforced (`413`)
- CRC32/SHA256 checksum in response for integrity

## ❌ Error Scenarios
`400` bad mime/size, `404` file not found, `409` upload session expired, `413` too large, `415` unsupported type, `423` locked (being written)

---

# Comment Service

> Threaded commenting on any entity type (posts, files, code). Supports reactions, mentions, and moderation.

**Entities:** Comment, Thread, Reaction, Mention, Report

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/comments` | Create comment | 201 |
| GET | `/v1/comments` | List comments on entity | 200 |
| GET | `/v1/comments/{comment_id}/replies` | Get thread replies | 200 |
| PUT | `/v1/comments/{comment_id}` | Edit comment | 200 |
| DELETE | `/v1/comments/{comment_id}` | Soft-delete comment | 204 |
| POST | `/v1/comments/{comment_id}/reactions` | Add reaction | 200 |
| DELETE | `/v1/comments/{comment_id}/reactions/{emoji}` | Remove reaction | 204 |
| POST | `/v1/comments/{comment_id}/report` | Report comment | 202 |

### Key Request / Response Examples

```
POST /v1/comments
Body: { entity_type, entity_id, body, parent_id? }
Response: { comment_id, author, body, created_at, thread_id }

GET /v1/comments?entity_type=post&entity_id=123&limit=20&cursor=xxx
Response: { data:[comments], next_cursor }

PUT /v1/comments/{id}
Headers: If-Match: "v2"
Body: { body }
Response: { comment_id, body, edited_at, version }

POST /v1/comments/{id}/reactions
Body: { emoji: "👍" }
Response: { reactions: {"👍":10, "❤️":3} }
```

## 🔒 Auth
JWT Bearer. Scopes: `comments:read` (public), `comments:write` (authenticated), `comments:moderate`

## 🔁 Idempotency
Reactions are idempotent (add same emoji twice = no-op). POST comment: `Idempotency-Key` to prevent double-submit.

## ⚙️ Special Considerations
- Optimistic concurrency on edit (ETag / `If-Match`)
- Soft delete preserves thread integrity (body replaced with `[deleted]`, replies stay)
- Rate-limit per user per entity (spam prevention)
- Mention parsing (`@username` → notification event)
- Profanity / ML moderation pre-save
- Cursor pagination (real-time stream)
- WebSocket push for live comment updates

## ❌ Error Scenarios
`400` empty body, `403` edit another user's comment, `404` comment not found, `409` edit conflict (stale ETag), `422` body too long

---

# Pub-Sub Service

> Asynchronous message broker. Producers publish to topics; consumers subscribe via push (webhook) or pull.

**Entities:** Topic, Subscription, Message, Snapshot, DeadLetterQueue

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/topics` | Create topic | 201 |
| POST | `/v1/topics/{topic_id}/publish` | Publish messages | 200 |
| POST | `/v1/subscriptions` | Create subscription | 201 |
| POST | `/v1/subscriptions/{sub_id}/pull` | Pull messages | 200 |
| POST | `/v1/subscriptions/{sub_id}/acknowledge` | Ack messages | 200 |
| POST | `/v1/subscriptions/{sub_id}/modifyAckDeadline` | Extend ack deadline | 200 |
| GET | `/v1/topics/{topic_id}/subscriptions` | List subscriptions | 200 |
| DELETE | `/v1/subscriptions/{sub_id}` | Delete subscription | 204 |

### Key Request / Response Examples

```
POST /v1/topics/{topic_id}/publish
Body: { messages:[{ data, attributes, ordering_key? }] }
Response: { message_ids:[] }

POST /v1/subscriptions
Body: { topic_id, name, delivery:"push"|"pull", endpoint?, ack_deadline_seconds }
Response: { sub_id, ... }

POST /v1/subscriptions/{sub_id}/pull
Body: { max_messages: 10 }
Response: { messages:[{ ack_id, data, attributes, publish_time }] }

POST /v1/subscriptions/{sub_id}/acknowledge
Body: { ack_ids:[] }
Response: {}
```

## 🔒 Auth
OAuth 2.0 service accounts. Scopes: `pubsub:publish`, `pubsub:subscribe`, `pubsub:admin`. Topic-level IAM.

## 🔁 Idempotency
Publish: include `message_id` (dedup within retention window). Ack is idempotent. Unacked messages redelivered after `ack_deadline`.

## ⚙️ Special Considerations
- At-least-once delivery guarantee
- Exactly-once via `message_id` dedup
- Ordering keys for ordered delivery within partition
- Dead-letter queue after N delivery attempts
- Push delivery with HMAC-signed payloads
- Exponential backoff on push failures
- Message retention: 7–30 days
- Snapshot for replay
- Schema registry for payload validation

## ❌ Error Scenarios
`400` invalid schema, `403` no publish permission, `404` topic/sub not found, `413` message too large (10MB limit), `429` publish quota exceeded

---

# YouTube Streaming API

> Video upload, transcoding, streaming, and metadata management. Adaptive bitrate (HLS/DASH). Live streaming support.

**Entities:** Video, Channel, Playlist, LiveStream, Caption, Analytics

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/videos/upload-session` | Initiate video upload | 201 |
| PUT | `/v1/videos/upload-session/{id}` | Stream upload bytes | 200 |
| GET | `/v1/videos/{video_id}` | Get video metadata | 200 |
| GET | `/v1/videos/{video_id}/stream` | Get adaptive manifests | 200 |
| POST | `/v1/live-streams` | Create live stream | 201 |
| POST | `/v1/live-streams/{id}/transition` | Start/end broadcast | 200 |
| GET | `/v1/search` | Search videos | 200 |
| GET | `/v1/videos/{video_id}/analytics` | Video analytics | 200 |

### Key Request / Response Examples

```
POST /v1/videos/upload-session
Body: { title, description, privacy, category }
Response: { upload_id, upload_url }

PUT /v1/videos/upload-session/{id}
Headers: Content-Range: bytes 0-999999/5000000
Body: Binary chunk
Response: { offset, status }

GET /v1/videos/{video_id}/stream
Response: { hls_url, dash_url, qualities:[1080,720,480] }

POST /v1/live-streams
Body: { title, scheduled_start, latency_mode }
Response: { stream_id, rtmp_url, stream_key, playback_url }

POST /v1/live-streams/{id}/transition
Body: { status: "testing"|"live"|"complete" }
Response: { status, viewer_count }
```

## 🔒 Auth
OAuth 2.0. Scopes: `youtube.readonly`, `youtube.upload`, `youtube.force-ssl`. API quota units per operation.

## 🔁 Idempotency
Resumable upload: idempotent by offset/Content-Range. PUT is idempotent. Live stream transition uses status machine (idempotent if same status).

## ⚙️ Special Considerations
- Resumable upload with `Content-Range` header
- Async transcoding → webhook on completion
- Adaptive bitrate: HLS/DASH manifests
- CDN-backed stream URLs (signed, short TTL for private videos)
- RTMP ingest for live streaming
- Chapter markers, captions (VTT/SRT)
- DRM (Widevine/FairPlay)
- Age-gate on content
- Copyright ContentID webhook
- Quota budgeting per project

## ❌ Error Scenarios
`400` bad aspect ratio/codec, `403` quota exceeded, `404` video not found, `409` upload offset mismatch, `451` geo-restricted content

---

# Facebook Messenger API

> Send/receive messages for chatbots and business accounts via the Messenger Platform (Meta Graph API style).

**Entities:** Conversation, Message, Attachment, Template, Webhook, User

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/messages` | Send message | 200 |
| GET | `/v1/conversations` | List conversations | 200 |
| GET | `/v1/conversations/{conv_id}/messages` | Get messages | 200 |
| POST | `/v1/messages/{msg_id}/reactions` | React to message | 200 |
| POST | `/v1/webhooks` | Subscribe to events | 201 |
| DELETE | `/v1/messages/{msg_id}` | Unsend message | 200 |
| POST | `/v1/attachments` | Upload reusable asset | 200 |
| GET | `/v1/users/{user_id}` | Get user profile | 200 |

### Key Request / Response Examples

```
POST /v1/messages
Body: { recipient:{id}, message:{text|attachment|template} }
Response: { message_id, recipient_id }

POST /v1/webhooks
Body: { url, events:["messages","reads","deliveries"], secret }
Response: { webhook_id, verify_token }
  ↓
GET /v1/webhooks?hub.mode=subscribe&hub.challenge=xxx&hub.verify_token=xxx
← must echo hub.challenge back (verification handshake)

POST /v1/attachments
Body: Multipart form (image/video/file)
Response: { attachment_id }   ← reuse in future messages
```

## 🔒 Auth
Page Access Token (OAuth 2.0). App Secret for webhook signature. User-level tokens for user-initiated flows.

## 🔁 Idempotency
Messages: include `client_message_id` to deduplicate sends. Webhook delivery: replay with same `event_id`; be idempotent in handler.

## ⚙️ Special Considerations
- Webhook GET challenge handshake for verification
- HMAC-SHA256 `X-Hub-Signature-256` on all webhook payloads
- Quick replies, carousel templates, buttons
- 24-hour messaging window (standard) + Message Tags for outside window
- Typing indicators (`POST /actions`)
- Read receipts
- Handover Protocol for multi-bot
- Rate limit: 200 msgs/sec per page

## ❌ Error Scenarios
`400` invalid recipient/message, `403` page not opted-in, `404` user not found, `429` rate limit, `200+error_code` (Graph API quirk — always check body)

---

# Google Maps API

> Geocoding, directions, places search, static/dynamic maps, distance matrix, and elevation services.

**Entities:** Place, Route, Waypoint, Geocode, DistanceMatrix, Elevation

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/v1/geocode` | Address → coordinates | 200 |
| GET | `/v1/geocode/reverse` | Coordinates → address | 200 |
| GET | `/v1/directions` | Get route/directions | 200 |
| GET | `/v1/places/search` | Nearby places search | 200 |
| GET | `/v1/places/{place_id}` | Place details | 200 |
| GET | `/v1/distance-matrix` | Multi-origin distances | 200 |
| GET | `/v1/elevation` | Elevation data | 200 |
| GET | `/v1/maps/static` | Static map image | 200 |

### Key Request / Response Examples

```
GET /v1/geocode?address=1600+Amphitheatre+Pkwy&components=country:US
Response: { results:[{ place_id, geometry:{lat,lng}, formatted_address }] }

GET /v1/directions?origin=A&destination=B&mode=driving|walking|transit&waypoints=C|D
Response: { routes:[{ legs, duration, distance, polyline }] }

GET /v1/places/search?location=37.4,-122.1&radius=1000&type=restaurant
Response: { results:[{ place_id,name,rating,vicinity,open_now }], next_page_token }

GET /v1/distance-matrix?origins=A|B&destinations=C|D&mode=driving
Response: { rows:[{ elements:[{ distance, duration, status }] }] }
```

## 🔒 Auth
API Key (`?key=` query param for server; restricted by IP/referrer). OAuth 2.0 for user-facing APIs.

## 🔁 Idempotency
All GET endpoints — safe + idempotent by definition. Cacheable with `Cache-Control`.

## ⚙️ Special Considerations
- API Key restrictions: HTTP referrer, IP, mobile app bundle ID
- Per-method quotas (queries per day / per second)
- `next_page_token` for places pagination (not cursor — token expires in 60s)
- Polyline encoding for routes
- Distance matrix: O(n×m) cost per call
- Autocomplete debounce (min 3 chars, 300ms)
- Maps JS API loads asynchronously with callback

## ❌ Error Scenarios
`400` invalid params, `ZERO_RESULTS` status in body, `OVER_QUERY_LIMIT` (429 equiv), `REQUEST_DENIED` (bad key), `INVALID_REQUEST`

---

# Chess API

> Create and manage chess games, validate moves, engine analysis, puzzles, and tournament pairings.

**Entities:** Game, Move, Position, Player, Puzzle, Tournament, Analysis

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/games` | Create game | 201 |
| POST | `/v1/games/{game_id}/moves` | Submit move | 200 |
| GET | `/v1/games/{game_id}` | Get game state | 200 |
| GET | `/v1/games/{game_id}/analysis` | Computer analysis | 200 |
| POST | `/v1/games/{game_id}/resign` | Resign game | 200 |
| POST | `/v1/games/{game_id}/draw-offer` | Offer draw | 200 |
| GET | `/v1/puzzles/daily` | Daily puzzle | 200 |
| POST | `/v1/puzzles/{puzzle_id}/attempt` | Submit puzzle solution | 200 |

### Key Request / Response Examples

```
POST /v1/games
Body: { white_player_id, black_player_id, time_control:{base_secs,increment_secs}, variant }
Response: { game_id, fen, pgn_headers }

POST /v1/games/{game_id}/moves
Headers: Idempotency-Key: move-uuid
Body: { move: "e2e4" }   ← UCI notation, or "e4" SAN
Response: { fen, pgn, last_move, status, clock:{white_ms,black_ms} }

GET /v1/games/{game_id}?format=json
Response: { game_id, fen, moves:[], status, players, clocks }

GET /v1/games/{game_id}/analysis?depth=20&multipv=3
Response: { moves:[{ san, score_cp, depth, best_move }] }
  ↑ returns 202 if analysis not ready yet, poll again
```

## 🔒 Auth
OAuth 2.0 (player actions). API Key (read-only engine/puzzle access). JWT for real-time move auth.

## 🔁 Idempotency
Move submission: `Idempotency-Key` per move (clock already ticked — server ignores duplicate with same key). Game creation: `Idempotency-Key` to prevent double-game.

## ⚙️ Special Considerations
- Real-time move delivery via WebSocket (`/v1/games/{id}/stream`)
- Illegal move → `422` with reason
- Clock managed server-side (anti-cheat)
- FEN and PGN both returned in response
- Engine analysis async (Stockfish): `202` → poll or webhook
- ELO rating update after game
- Move validation: castling, en passant, promotion (include promotion piece: `e7e8q`)
- Time-scramble: server enforces flag (loss on time)

## ❌ Error Scenarios
`400` invalid FEN/SAN, `403` not your turn, `409` game already ended, `410` game expired, `422` illegal move (with explanation)

---

# Zoom API

> Create meetings, webinars, manage participants, recordings, and real-time streaming for conferencing.

**Entities:** Meeting, Webinar, Participant, Recording, Breakout, Webhook

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/meetings` | Create meeting | 201 |
| GET | `/v1/meetings/{meeting_id}` | Get meeting details | 200 |
| PATCH | `/v1/meetings/{meeting_id}` | Update meeting | 204 |
| DELETE | `/v1/meetings/{meeting_id}` | Cancel meeting | 204 |
| GET | `/v1/meetings/{meeting_id}/participants` | List participants | 200 |
| GET | `/v1/meetings/{meeting_id}/recordings` | Get recordings | 200 |
| POST | `/v1/webinars` | Create webinar | 201 |
| POST | `/v1/meetings/{meeting_id}/live-stream` | Start live stream | 204 |

### Key Request / Response Examples

```
POST /v1/meetings
Body: {
  topic, type:1|2|3|8, start_time, duration_min, password,
  settings:{ waiting_room, mute_on_entry }
}
Response: { meeting_id, join_url, start_url, password, uuid }

GET /v1/meetings/{id}/participants?limit=100&next_page_token=xxx
Response: {
  participants:[{ name, email, join_time, leave_time, duration_sec }],
  next_page_token
}

GET /v1/meetings/{id}/recordings
Response: { recording_files:[{ type, play_url, download_url, file_size }] }
```

## 🔒 Auth
OAuth 2.0 (Authorization Code or Server-to-Server). JWT deprecated. Scopes: `meeting:read`, `meeting:write`, `recording:read`, `webinar:write`.

## 🔁 Idempotency
Meeting creation: `Idempotency-Key`. PATCH/DELETE idempotent. Webhook events contain `uuid` — be idempotent in handler (events may repeat).

## ⚙️ Special Considerations
- Server-to-Server OAuth for backend automation (no user login required)
- Webhook signature verification (`X-Zm-Signature` + secret)
- Meeting password / waiting room for security
- Rate limits: 100 req/day for some endpoints (per-user and per-account)
- Recording download requires separate token with expiry
- Dashboard API for admin-level metrics
- Recurrence meetings: use `occurrence_id` for individual instances

## ❌ Error Scenarios
`300` invalid meeting ID format, `400` bad params, `404` meeting not found, `429` rate limit (`Retry-After` header), `3000` series app-specific error codes

---

# LeetCode API

> Problem management, code submission, judge execution, contest management, and user progress tracking.

**Entities:** Problem, Submission, TestCase, Contest, Leaderboard, User

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/v1/problems` | List problems | 200 |
| GET | `/v1/problems/{slug}` | Get problem detail | 200 |
| POST | `/v1/submissions` | Submit solution | 202 |
| GET | `/v1/submissions/{submission_id}` | Get judge result | 200 |
| POST | `/v1/submissions/run` | Run against custom test | 202 |
| GET | `/v1/contests` | List contests | 200 |
| POST | `/v1/contests/{contest_id}/register` | Register for contest | 201 |
| GET | `/v1/users/{username}/stats` | User stats | 200 |

### Key Request / Response Examples

```
GET /v1/problems?difficulty=medium&tags=dp,graph&status=todo&limit=20&cursor=xxx
Response: { data:[{ id,slug,title,difficulty,acceptance_rate,tags }], next_cursor }

POST /v1/submissions
Headers: Idempotency-Key: sub-uuid
Body: { problem_id, language:"python3", code:"class Solution:..." }
Response: { submission_id, status:"queued", poll_url:"/v1/submissions/sub_xyz" }

GET /v1/submissions/{id}
Response: {
  status: "accepted"|"wrong_answer"|"TLE"|"MLE"|"CE"|"RE",
  runtime_ms, memory_mb, test_case_result
}

GET /v1/users/{username}/stats
Response: { solved:{easy,medium,hard,total}, acceptance_rate, streak, ranking }
```

## 🔒 Auth
JWT Bearer (user sessions). API Key for CI/CD integrations. Rate-limited by user tier.

## 🔁 Idempotency
Submit: `Idempotency-Key` prevents duplicate submissions (e.g. double-click). Judge is async — poll `/submissions/{id}`. Run is non-idempotent (different test inputs allowed).

## ⚙️ Special Considerations
- Async judge: `202` → poll for result (long-poll or webhook)
- Judge sandboxing (cgroups, seccomp)
- Language-specific time/memory limits stored per problem
- Contest: submissions locked after `end_time` (server-side timestamp validation)
- Anti-cheat: plagiarism detection async post-contest
- Rate-limit: N submissions per problem per hour
- Problem tags used for recommendation engine

## ❌ Error Scenarios
`400` unsupported language, `403` contest not started yet, `404` problem not found, `409` already registered for contest, `429` submission rate limit

---

# Payment Gateway API

> Charge customers, manage subscriptions, refunds, payouts, and fraud detection. PCI-DSS compliant tokenization.

**Entities:** PaymentMethod, PaymentIntent, Charge, Refund, Subscription, Payout, Webhook

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/payment-methods` | Tokenize card | 201 |
| POST | `/v1/payment-intents` | Create payment intent | 201 |
| POST | `/v1/payment-intents/{id}/confirm` | Confirm payment | 200 |
| POST | `/v1/payment-intents/{id}/capture` | Capture authorized amount | 200 |
| POST | `/v1/refunds` | Issue refund | 201 |
| POST | `/v1/subscriptions` | Create subscription | 201 |
| DELETE | `/v1/subscriptions/{sub_id}` | Cancel subscription | 200 |
| POST | `/v1/webhooks` | Register webhook | 201 |

### Key Request / Response Examples

```
POST /v1/payment-intents
Headers: Idempotency-Key: pi-uuid
Body: { amount:1000, currency:"usd", payment_method_id:"pm_xxx", confirm:false }
Response: { intent_id, status:"requires_confirmation", client_secret }

POST /v1/payment-intents/{id}/confirm
Headers: Idempotency-Key: confirm-uuid
Response: {
  status: "succeeded"|"requires_action",
  next_action:{ type:"redirect", redirect_url }   ← 3DS2 flow
}

POST /v1/refunds
Headers: Idempotency-Key: refund-uuid
Body: { charge_id, amount:500, reason:"customer_request" }
Response: { refund_id, amount, status:"pending"|"succeeded"|"failed" }

POST /v1/subscriptions
Body: { customer_id, price_id, trial_days:14 }
Response: { sub_id, status, current_period_end, latest_invoice }
```

## 🔒 Auth
Secret Key (server-side only, never client). Publishable Key (client-side tokenization only). HTTPS-only. `Authorization: Bearer sk_live_xxx`

## 🔁 Idempotency
**CRITICAL: ALL POST mutations MUST use `Idempotency-Key`.** Payment retry without new charge. Refund dedup. 7-day key retention. `409` if same key with different payload.

## ⚙️ Special Considerations
- **PCI-DSS**: card numbers NEVER touch your server — use JS SDK / mobile SDK tokenization
- 3DS2 / SCA: `next_action` flow for bank authentication
- Webhook HMAC-SHA256 signature (`Stripe-Signature` style)
- Idempotency keys mandatory on all state-changing calls
- Soft descriptor for bank statement
- Fraud: ML risk score, AVS (address verification), CVV check
- Payout schedule configurable
- Test mode with `sk_test_` prefix
- Separate `authorize` (hold funds) from `capture` (settle)

## ❌ Error Scenarios
`400` invalid card, `402` card declined (with `decline_code`), `402` insufficient funds, `409` idempotency conflict, `429` rate limit, `charge.failed` webhook event

---

# Twitter API

> Post/read tweets, manage followers, search tweets, upload media, and stream real-time content.

**Entities:** Tweet, User, Media, List, Space, Stream, DirectMessage

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v2/tweets` | Create tweet | 201 |
| GET | `/v2/tweets/{tweet_id}` | Get tweet | 200 |
| DELETE | `/v2/tweets/{tweet_id}` | Delete tweet | 200 |
| GET | `/v2/tweets/search/recent` | Search tweets | 200 |
| GET | `/v2/tweets/search/stream` | Filtered real-time stream | 200 |
| POST | `/v2/tweets/search/stream/rules` | Add stream rules | 201 |
| POST | `/v1/media/upload` | Upload media | 200 |
| GET | `/v2/users/{user_id}/followers` | Get followers | 200 |

### Key Request / Response Examples

```
POST /v2/tweets
Headers: Idempotency-Key: tweet-uuid
Body: { text, media?:{media_ids:["123"]}, reply?:{in_reply_to_tweet_id:"456"} }
Response: { data:{ id, text, created_at } }

GET /v2/tweets/search/recent
  ?query=from:user -is:retweet
  &tweet.fields=public_metrics,author_id
  &expansions=author_id
  &max_results=10
  &next_token=xxx
Response: { data:[tweets], includes:{users}, meta:{next_token,result_count} }

# Media upload (3-step)
POST /v1/media/upload  INIT   → { media_id, expires_after_secs }
POST /v1/media/upload  APPEND → (chunk, segment_index)
POST /v1/media/upload  FINALIZE → { media_id, processing_info:{state,check_after_secs} }
```

## 🔒 Auth
OAuth 2.0 PKCE (user context). Bearer Token (app-only, read). Access levels: Free, Basic, Pro, Enterprise — different rate limits per tier.

## 🔁 Idempotency
Tweet creation: `Idempotency-Key` in header. Media upload INIT is idempotent (same command/total_bytes → same `media_id`). DELETE idempotent.

## ⚙️ Special Considerations
- **Expansions pattern**: request related objects in single call to reduce round-trips
- Tweet fields / User fields selection reduces payload
- Filtered stream: rule-based, persistent SSE connection
- Chunked media upload: INIT → APPEND (with `segment_index`) → FINALIZE → poll processing
- Rate limits vary by endpoint and access tier
- Quoted tweets via `quoted_tweet_id`, thread replies via `in_reply_to_tweet_id`

## ❌ Error Scenarios
`400` invalid query, `403` no write permission, `404` tweet not found, `429` rate limit (`x-rate-limit-reset` header), `453` endpoint requires higher access level

---

# Uber API

> Request rides, get price estimates, track drivers, manage trip lifecycle, and earnings for drivers.

**Entities:** Trip, Estimate, Driver, Vehicle, Route, Surge, Receipt

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/v1/estimates/price` | Get price estimates | 200 |
| GET | `/v1/estimates/time` | ETA estimates | 200 |
| POST | `/v1/requests` | Request ride | 202 |
| GET | `/v1/requests/{request_id}` | Get trip status | 200 |
| DELETE | `/v1/requests/{request_id}` | Cancel ride | 204 |
| GET | `/v1/requests/{request_id}/map` | Trip map | 200 |
| GET | `/v1/requests/{request_id}/receipt` | Trip receipt | 200 |
| POST | `/v1/requests/{request_id}/rating` | Rate driver | 200 |

### Key Request / Response Examples

```
GET /v1/estimates/price
  ?start_latitude=37.3&start_longitude=-122.0
  &end_latitude=37.7&end_longitude=-122.4
Response: {
  prices:[{ product_id, display_name:"UberX", estimate:"$12-15",
            surge_multiplier:1.0, duration_secs:900, distance }]
}

POST /v1/requests
Headers: Idempotency-Key: req-uuid
Body: { product_id, start_lat, start_lng, end_lat, end_lng, fare_id }
  ↑ fare_id from estimate — locks price, expires in ~2 min
Response: { request_id, status:"processing", surge_multiplier, eta }

GET /v1/requests/{id}
Response: {
  status: "accepted"|"arriving"|"in_progress"|"completed",
  driver:{ name, rating, photo },
  vehicle:{ make, model, plate },
  location:{ lat, lng },
  eta_secs
}
```

## 🔒 Auth
OAuth 2.0 Authorization Code (user trips). Server Token (estimate endpoints, no user context). Scopes: `request`, `history`, `profile`.

## 🔁 Idempotency
Ride request: `Idempotency-Key` (avoid double-booking). `fare_id` from estimate locks price and must be used within TTL (~2 min). DELETE (cancel) is idempotent.

## ⚙️ Special Considerations
- Price lock via `fare_id` (from estimate, has TTL — prevents surge bait-and-switch)
- Surge pricing prominently displayed in estimate
- Async request: `202` → poll `/requests/{id}` or listen to webhook
- Real-time driver location via WebSocket or polling
- Upfront pricing with cancellation fee logic
- Geographic availability check before request
- Driver matching is server-side (client cannot choose driver)
- Product catalog varies by city

## ❌ Error Scenarios
`400` invalid coordinates, `404` product unavailable in area, `409` already have active request, `422` no drivers available (surge too high / zone), `429` rate limit

---

# CamelCamelCamel API

> Amazon product price history tracking, price drop alerts, and product data lookup.

**Entities:** Product, PriceHistory, Alert, Category, Deal

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/v1/products/{asin}` | Get product info | 200 |
| GET | `/v1/products/{asin}/history` | Get price history | 200 |
| POST | `/v1/alerts` | Create price alert | 201 |
| GET | `/v1/alerts` | List user alerts | 200 |
| DELETE | `/v1/alerts/{alert_id}` | Delete alert | 204 |
| GET | `/v1/search` | Search products | 200 |
| GET | `/v1/deals` | Top price drops today | 200 |
| GET | `/v1/categories` | List categories | 200 |

### Key Request / Response Examples

```
GET /v1/products/{asin}?locale=us
Response: {
  asin, title, url, image, category,
  current_prices:{ amazon:29.99, new:27.50, used:15.00 }
}

GET /v1/products/{asin}/history?type=amazon&from=2023-01-01&to=2024-01-01
Response: {
  asin,
  history:[{ date:"2024-01-15", price:32.99 }],
  stats:{ min:19.99, max:49.99, avg:34.50, current:29.99 }
}

POST /v1/alerts
Headers: Idempotency-Key: alert-uuid
Body: { asin, target_price:25.00, type:"amazon", email:"user@example.com" }
Response: { alert_id, asin, target_price, status:"active" }

GET /v1/deals?category=electronics&min_discount_pct=20&limit=20
Response: { data:[{ asin, title, current_price, lowest_price, drop_pct }] }
```

## 🔒 Auth
API Key (header `X-API-Key`). Free tier: limited calls/day. Pro: higher limits + webhooks for alerts.

## 🔁 Idempotency
Alert creation: `Idempotency-Key` (avoid duplicate alerts for same ASIN). DELETE idempotent. GET endpoints safe + idempotent + cacheable.

## ⚙️ Special Considerations
- Price history is time-series — use `?from/to` date range to limit payload
- Cache aggressively (history data is immutable once written)
- Push alerts via email or webhook (POST to configured URL with HMAC signature)
- `locale` param for regional pricing (US/UK/DE/etc.)
- `null` price means out-of-stock
- ASIN validation: 10-character alphanumeric

## ❌ Error Scenarios
`400` invalid ASIN format, `404` product not found / not tracked, `422` target price must be below current price, `429` quota exceeded

---

# Gaming API

> Multiplayer matchmaking, leaderboards, achievements, inventory, in-game purchases, and game sessions.

**Entities:** Player, Session, Match, Leaderboard, Achievement, Inventory, Item

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/sessions` | Create game session | 201 |
| POST | `/v1/matchmaking/tickets` | Enter matchmaking | 202 |
| GET | `/v1/matchmaking/tickets/{ticket_id}` | Check match status | 200 |
| GET | `/v1/leaderboards/{board_id}` | Get leaderboard | 200 |
| POST | `/v1/leaderboards/{board_id}/scores` | Submit score | 200 |
| GET | `/v1/players/{player_id}/achievements` | Get achievements | 200 |
| POST | `/v1/players/{player_id}/inventory` | Add item to inventory | 201 |
| POST | `/v1/store/purchase` | Buy item (IAP) | 201 |

### Key Request / Response Examples

```
POST /v1/sessions
Headers: Idempotency-Key: session-uuid
Body: { game_id, mode:"battle_royale", max_players:100, map:"verdansk" }
Response: { session_id, server_ip, server_port, join_token, expires_at }

POST /v1/matchmaking/tickets
Body: { player_id, game_mode:"ranked", skill_rating:1450, preferences:{region:"us-west"} }
Response: { ticket_id, status:"queued", estimated_wait_secs:30, poll_url }

POST /v1/leaderboards/{board_id}/scores
Headers: Idempotency-Key: score-uuid
Body: { player_id, score:98500, metadata:{kills:15,time_secs:1200} }
Response: { rank:42, score:98500, personal_best:true }

POST /v1/store/purchase
Headers: Idempotency-Key: purchase-uuid
Body: { player_id, item_id:"skin_001", receipt:"<apple/google receipt>" }
Response: { transaction_id, items_granted:[{item_id,quantity}], currency_balance:500 }
```

## 🔒 Auth
JWT (player sessions from game client). Server-to-server: HMAC-signed requests. Server authoritative — never trust client for scores or loot drops.

## 🔁 Idempotency
Score submission: `Idempotency-Key` (client retry safe). Match creation: dedup. IAP purchase: `Idempotency-Key` + receipt validation (idempotent on same receipt). Inventory add: idempotent key per drop event.

## ⚙️ Special Considerations
- **Server-authoritative game state** (anti-cheat) — client never dictates outcomes
- Score validation: server-side plausibility check (score vs session duration)
- Leaderboard: sorted set (Redis ZSET internally)
- Real-time: WebSocket / UDP for in-session events
- Matchmaking: ELO/skill-based, region-aware, async poll or webhook
- IAP receipt validation: forward to Apple/Google/Stripe server-side
- Soft currency vs hard currency split
- Battle pass / season progression tracking

## ❌ Error Scenarios
`400` invalid score (too high for session duration), `403` player banned, `404` session expired, `409` duplicate score submission (non-idempotent attempt), `422` IAP receipt invalid

---

# Evernote API

> Create, read, update, and organize notes, notebooks, tags, and attachments. Full-text search across notes.

**Entities:** Note, Notebook, Tag, Resource (Attachment), SharedNote, SavedSearch

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/notebooks` | Create notebook | 201 |
| POST | `/v1/notes` | Create note | 201 |
| GET | `/v1/notes/{note_id}` | Get note | 200 |
| PUT | `/v1/notes/{note_id}` | Update note | 200 |
| DELETE | `/v1/notes/{note_id}` | Move to trash | 204 |
| GET | `/v1/notes` | Search notes | 200 |
| POST | `/v1/notes/{note_id}/resources` | Attach resource | 201 |
| POST | `/v1/notes/{note_id}/share` | Share note | 201 |

### Key Request / Response Examples

```
POST /v1/notes
Headers: Idempotency-Key: note-uuid
Body: {
  title:"Meeting Notes",
  content_enml:"<?xml version...><!DOCTYPE en-note...><en-note>...</en-note>",
  notebook_id:"nb_123",
  tag_names:["work","q4"]
}
Response: { note_id, title, created_at, update_sequence_num:42 }

PUT /v1/notes/{id}
Headers: If-Match: "42"   ← USN acts as ETag
Body: { title:"Updated Title", content_enml:"..." }
Response: { note_id, update_sequence_num:43, updated_at }
↑ 409 if USN mismatch (stale update)

GET /v1/notes?q=meeting&notebook_id=nb_123&tags=work&limit=20&offset=0
Response: { notes:[{ note_id,title,snippet,updated_at }], total_count:87 }

POST /v1/notes/{id}/share
Body: { permission:"read"|"edit", expires_at:"2024-12-31T00:00:00Z" }
Response: { share_url, share_key }
```

## 🔒 Auth
OAuth 2.0 (user notes). Developer tokens (personal scripts). Scopes: `note.read`, `note.write`, `note.delete`, `notebook.manage`.

## 🔁 Idempotency
Note creation: `Idempotency-Key`. Note update: USN (Update Sequence Number) = optimistic locking via `If-Match`. Resource upload idempotent by content hash.

## ⚙️ Special Considerations
- **ENML** (Evernote Markup Language) for rich content — validated server-side on write
- **Update Sequence Number (USN)**: incrementing integer acts as version/ETag for conflict detection
- Sync protocol: get delta changes since last USN
- Note size limit: 100MB with resources
- Shared note URL with access token embedded
- Tags are global (not per-notebook)
- Saved searches (stored query objects)
- Offline sync: track dirty notes by USN delta
- Content hash stored per resource for deduplication

## ❌ Error Scenarios
`400` invalid ENML, `404` note not found, `409` note update conflict (USN mismatch), `413` note too large, `429` rate limit

---

# Amazon IAM API

> Manage users, groups, roles, policies, and credentials. Grant/revoke fine-grained AWS resource permissions.

**Entities:** User, Group, Role, Policy, Permission, Credential, TemporaryCredential

## Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/v1/users` | Create IAM user | 201 |
| POST | `/v1/users/{username}/access-keys` | Create access key | 201 |
| PUT | `/v1/users/{username}/access-keys/{key_id}/status` | Activate/deactivate key | 200 |
| POST | `/v1/roles` | Create role | 201 |
| POST | `/v1/roles/{role_name}/assume` | Assume role (STS) | 200 |
| POST | `/v1/policies` | Create policy | 201 |
| PUT | `/v1/users/{username}/policies/{policy_arn}` | Attach policy to user | 200 |
| GET | `/v1/users/{username}/permissions` | Simulate permissions | 200 |

### Key Request / Response Examples

```
POST /v1/users/{username}/access-keys
Headers: Idempotency-Key: key-uuid
Response: {
  access_key_id: "AKIAIOSFODNN7EXAMPLE",
  secret_access_key: "wJalrXUtnFEMI/K7MDENG..."  ← shown ONCE only, store immediately
  status: "Active"
}

POST /v1/roles/{role_name}/assume
Body: {
  role_session_name:"MySession",
  duration_secs:3600,
  external_id:"abc123",   ← required for cross-account
  mfa_serial?:"arn:..."
}
Response: {
  credentials:{
    access_key, secret_key, session_token,
    expiration:"2024-01-15T11:30:00Z"
  },
  assumed_role_user:{ arn }
}

POST /v1/policies
Body: {
  policy_name:"S3ReadOnly",
  document:{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Action":["s3:GetObject","s3:ListBucket"],
      "Resource":["arn:aws:s3:::my-bucket/*"],
      "Condition":{ "IpAddress":{ "aws:SourceIp":"192.168.1.0/24" } }
    }]
  }
}

GET /v1/users/{username}/permissions
  ?actions=s3:GetObject,ec2:StartInstances
  &resources=arn:aws:s3:::my-bucket/*
Response: { results:[{ action, resource, decision:"Allow"|"Deny", matched_policies }] }
```

## 🔒 Auth
**AWS Signature V4** — HMAC-SHA256 of canonical request signed with secret key. Root account = emergency only. Use IAM roles + temporary credentials (STS AssumeRole) for services. MFA for privileged operations.

## 🔁 Idempotency
User/key creation: `Idempotency-Key`. Policy attach/detach idempotent. STS AssumeRole: not idempotent (each call = new temp creds). DELETE operations idempotent.

## ⚙️ Special Considerations
- **Secret access key shown ONCE** — store in Secrets Manager immediately at creation
- **Principle of least privilege**: attach minimal policies, use conditions
- Policy conditions: IP, MFA required, time windows, resource tags
- Permission boundaries for delegation without granting excess
- Service Control Policies (org-level guardrails that override everything)
- Access Advisor: last-used timestamp per permission (identify unused perms)
- Credential rotation automation (Lambda + EventBridge)
- Cross-account: `ExternalId` in AssumeRole trust policy (confused deputy protection)
- AWS managed vs customer managed policies
- Policy versioning (up to 5 versions; set default)
- IAM Access Analyzer for automated policy validation and finding overly-permissive policies

## ❌ Error Scenarios
`400` invalid policy JSON, `403` access denied (with policy evaluation context), `404` entity not found, `409` entity already exists (`LimitExceededException`), `429` throttling (retry with exponential backoff)

---

*Study tip: For each API in your interview, address these 5 areas unprompted: (1) Core entities & data model, (2) Key endpoints with HTTP methods, (3) Auth strategy + scopes, (4) Idempotency design, (5) At least 2 special design considerations (async, pagination, webhooks, etc.).*
