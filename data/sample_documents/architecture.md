# RAG Insight Demo Service

This is a fictional service used only for reproducible retrieval experiments.

## Cache decision

Redis was selected for shared caching across API replicas and native key expiration. The cache TTL is 300 seconds. Cache keys use the prefix insight:v1:.

## Storage

PostgreSQL stores accounts and document metadata. Original document files are stored in object storage. Redis is not the source of truth.

## Upload processing

Uploads are limited to 10 MB per file. A worker extracts text and builds the search index asynchronously. The API returns a job identifier so clients can poll ingestion status.

## Deployment

The API runs as two replicas. Workers run separately from the API. The health endpoint is /healthz.
