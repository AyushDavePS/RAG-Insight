# Service Overview

The service stores durable records in PostgreSQL.

## Authentication

Clients send the `X-API-Key` header on every request.

```python
# this is a code comment, not a heading
headers = {"X-API-Key": token}
```

### Retries

When retries fail, the worker records the final error for investigation.

This final paragraph is deliberately long enough to require splitting when a small token budget is used. It preserves a useful source location across chunk boundaries without changing its Unicode meaning: café नमस्ते.
