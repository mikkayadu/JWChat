---
source: Context7 API
library: Celery + Redis
package: celery-redis
topic: parallel web crawling with deduplication
fetched: 2026-04-14T00:00:00Z
official_docs: https://docs.celeryq.dev/en/stable/
---

## Relevant, current guidance (Celery + Redis)

### 1) Atomic URL dedupe using Redis sets
- Use Redis `SADD` as your atomic dedupe gate: enqueue only if `SADD crawl:seen <url_hash>` returns `1`.
- `SISMEMBER`/`SMISMEMBER` are useful for read checks, but prefer a single write-path gate (`SADD`) to avoid check-then-act races.
- Normalize URLs before hashing (scheme/host casing, default ports, fragments, tracking params) so dedupe matches crawler intent.
- For memory control, rotate keys by crawl run/domain (`crawl:seen:{run_id}`) and expire with `EXPIRE` when crawl completes.

### 2) Queueing patterns for high fan-out crawling
- Split queues by workload class (e.g., `seed`, `fetch`, `parse`, `persist`) using `task_routes`/`task_queues`.
- Route IO-heavy fetchers to dedicated queues and workers; avoid mixing with CPU-heavy parsing on same worker pool.
- Keep prefetch conservative for fairness and crash recovery:
  - `worker_prefetch_multiplier = 1`
  - (Celery >=5.6) consider `worker_disable_prefetch = True` for pull-when-slot-free behavior.

### 3) Idempotent tasks (required with late-ack/retries)
- Design `fetch_url(url)` so duplicate delivery is harmless:
  - deterministic output keys (`content:{url_hash}`),
  - upsert writes,
  - side effects guarded by idempotency keys.
- Never assume “exactly once” with distributed workers; build for at-least-once.

### 4) Retries and backoff
- Prefer declarative retries on transient network failures:
  - `autoretry_for`, `max_retries`, `retry_backoff`, `retry_backoff_max`, `retry_jitter`.
- Use jitter to avoid synchronized retry storms.
- Classify non-retryable errors (e.g., malformed URL, blocked domain policy) to fail fast.

### 5) `acks_late` / `reject_on_worker_lost` semantics
- `task_acks_late=True`: ack after execution, improving crash safety for in-flight tasks.
- Pair with idempotency because worker crash can cause redelivery.
- For crash redelivery intent, use `task_reject_on_worker_lost=True` so unacked tasks are re-queued when worker process is lost.
- Keep Redis broker `visibility_timeout` greater than worst-case task runtime to reduce premature redelivery loops.

### 6) Rate limiting and politeness
- Apply per-task `rate_limit` for global worker-side shaping.
- Add per-domain politeness in Redis (token-bucket/counter with `INCR` + `EXPIRE`, optionally Lua for atomicity).
- Respect robots.txt, crawl-delay, and per-host connection limits; separate queues for “slow hosts” helps fairness.

### 7) Safe worker concurrency tuning
- For network-bound crawling, prefork/gevent/eventlet can work; validate library compatibility before non-prefork pools.
- Start conservative and load-test upward:
  - concurrency roughly near available cores for CPU-bound parse workers,
  - higher for IO-bound fetch workers (bounded by outbound sockets, DNS, target-site tolerance).
- Use `max_tasks_per_child` to mitigate memory leaks in long-running crawlers.
- Monitor queue latency, retry rate, duplicate-execution rate, and per-domain error spikes before raising concurrency.

## Minimal Celery config baseline (production-oriented)

```python
task_acks_late = True
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1

# Redis broker safety for long fetches
broker_transport_options = {"visibility_timeout": 3600}

# Retry defaults (override per task as needed)
task_default_retry_delay = 1
```

```python
from celery import Task

class CrawlTask(Task):
    autoretry_for = (TimeoutError, ConnectionError)
    max_retries = 7
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
```

```python
# Atomic dedupe gate before enqueue
if redis.sadd("crawl:seen", url_hash):
    fetch_url.apply_async(args=[url])
```

## Sources used
- Celery docs (routing, tasks, optimizing, configuration, workers, redis transport): https://docs.celeryq.dev/en/stable/
- Redis docs (sets, SISMEMBER/SMISMEMBER, atomic scripting patterns): https://redis.io/docs/latest/
