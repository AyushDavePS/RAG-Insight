# Demo Operations

## Cache failure

If Redis is unavailable, the API reads from PostgreSQL and skips cache writes. Requests continue with higher latency.

## Recovery

PostgreSQL backups run daily at 02:00 UTC. Backups are retained for 14 days. A restore drill runs every month.

## Ingestion retries

A failed ingestion job is retried twice with exponential backoff. After the final failure, the job is marked failed and requires manual resubmission.

## Monitoring

An alert fires when the API error rate exceeds 5 percent for 10 minutes. The ingestion queue has a separate alert when its oldest job exceeds 15 minutes.
