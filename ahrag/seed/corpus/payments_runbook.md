# Payments Service Runbook (payments-api)

Owner: Platform Engineering · On-call rota: `#oncall-payments`

## 1. Service overview

`payments-api` accepts authorisation and capture requests from the checkout
front end and forwards them to the acquirer gateway. It is a stateless Go
service behind an internal load balancer, backed by a Postgres ledger and a
Redis idempotency cache.

## 2. Error code reference

### ERR-5041 — Acquirer timeout on capture

The acquirer gateway did not respond to a capture request within the 8-second
timeout. The transaction is left in `PENDING_CAPTURE` and is safe to retry:
capture requests are idempotent on the `idempotency_key` header.

Remediation for ERR-5041:

1. Check the acquirer status page and the `acquirer_latency_p99` dashboard.
2. If p99 latency exceeds 6000 ms, raise severity to SEV-2 and page the
   acquirer liaison.
3. Replay the pending captures with `payctl replay --state PENDING_CAPTURE
   --since 30m`. The replay tool is idempotent and safe to run twice.
4. If the replay itself returns ERR-5041 for more than 5% of transactions,
   fail over to the secondary acquirer with `payctl acquirer failover
   --target secondary`.

Do not manually mark transactions as captured in the ledger. Manual ledger
edits break the daily reconciliation job and require a finance-side correction.

### ERR-4012 — Idempotency key collision

Two distinct request bodies were submitted with the same `idempotency_key`.
The second request is rejected. This is almost always a client bug: the caller
is reusing a key across different carts.

Remediation for ERR-4012: identify the calling service from the
`x-client-id` header, and open a ticket against that team. No platform-side
action resolves this; do not widen the key namespace to suppress the error.

### ERR-5503 — Ledger write rejected

The Postgres ledger rejected a write, usually because the connection pool is
saturated or a long-running migration holds a lock.

Remediation for ERR-5503:

1. Check `pg_stat_activity` for queries running longer than 60 seconds.
2. If a migration is in progress, wait; do not kill migration transactions.
3. If the pool is saturated with application queries, scale the read replicas
   and restart the service with `payctl rollout restart payments-api`.

## 3. Standard restart procedure

A rolling restart is performed with `payctl rollout restart payments-api`. The
rollout drains connections over 30 seconds per pod. Never restart more than one
pod at a time during business hours.

## 4. Escalation

SEV-1 and SEV-2 incidents page the on-call engineer through PagerDuty service
`PD-PAY-01`. The escalation policy is: on-call engineer (0 min), payments tech
lead (15 min), head of platform (30 min).
