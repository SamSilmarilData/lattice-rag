---
status: accepted
---

# The deployed database is a Seeded Database on a single instance

Cloud Run runs one instance (`--max-instances 1`) with the LatticeDB file on ephemeral disk, pre-seeded into the image. Runtime `POST /ingest` is supported but its documents are not expected to survive a restart or redeploy; durable content goes through `seed` at build time. We chose this over a persistent volume or a hosted store because LatticeDB is a single-writer embedded file, and extra instances would each hold a divergent copy while the duplicate-ingest lock is in-process only.

## Consequences

- Do not raise `--max-instances` without first moving to shared storage and replacing the in-process ingest lock.
- Ingest responses and docs should say that runtime documents are ephemeral.
