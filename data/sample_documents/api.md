# Demo API

## Authentication

Clients authenticate with an API key in the X-API-Key header. Missing or invalid keys receive HTTP 401.

## Upload endpoint

POST /documents accepts a multipart file upload and returns HTTP 202 with a job_id. GET /jobs/{job_id} returns the ingestion status.

## Query endpoint

POST /questions accepts a JSON object with a question field. Answers include a sources array containing document names and passage identifiers.

```json
{"question": "Why was Redis selected?"}
```

## Rate limits

Each API key is limited to 60 requests per minute. Exceeding the limit returns HTTP 429 with a Retry-After header.
