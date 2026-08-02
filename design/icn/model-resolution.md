---
applies_to:
  - inference/crates/icn-catalog/**
  - inference/crates/icn-models/**
  - inference/catalog/**
---

# ICN model resolution

Repository revisions are optimistic retrieval addresses. The shipped `ModelPackage` is authoritative:
its paths, sizes, SHA-256 digests, roles, and relationships define the only package ICN may install.

## Catalog lock advancement

```text
resolve each repository's current main to an immutable commit
                              |
                              v
resolve every declared format through the production catalog path
                              |
                 +------------+------------+
                 |                         |
              complete                 any failure
                 |                         |
                 v                         v
       atomically publish lock    keep previous lock unchanged
                                   report entry + root failure
```

Release generation uses only the published lock. It never follows `main`; any unresolved entry fails
the release.

## Runtime acquisition

Catalog loading, recommendation, and assessment use the installed planner bundle and make no upstream
request. Only acquisition resolves a repository:

```text
download exact paths from pinned commit
                 |
       +---------+------------------------------+
       |                                        |
    available                         definitive revision/file 404
       |                                        |
       v                                        v
download + verify                 resolve current main to a commit
                                                   |
                                      compare every required file
                                      - same relative path
                                      - same byte size
                                      - same SHA-256
                                                   |
                                      +------------+------------+
                                      |                         |
                                  all match                 any mismatch
                                      |                         |
                                      v                         v
                          download from resolved commit     fail acquisition
                          and verify downloaded bytes       install nothing
```

| Pinned-revision outcome | Behavior |
| --- | --- |
| Revision or required file is definitively absent | Attempt verified `main` fallback. |
| Authentication, authorization, rate limit, timeout, or server failure | Report the original failure; do not fall back. |

The fallback changes only the retrieval commit. Package and catalog identities, file roles and
relationships, planner inputs, capabilities, and recommendation evidence remain those shipped in the
release. Downloaded bytes must still pass ordinary SHA-256 validation before publication.

If `main` differs, the download attempt reports `package_unavailable`, the catalog entry remains
present, and installed copies remain valid. Structured diagnostics identify the repository, pinned
and observed commits, and first missing or mismatched path without logging credentials.

## Required guarantees

- Publish a lock only after every catalog entry resolves through the production generation path.
- Generate releases from the lock, never from a mutable branch.
- Fall back only for definitive absence, never for transient or access failures.
- Accept a replacement commit only when every required path, size, and SHA-256 matches.
- Publish downloaded content only after validating it against the shipped package.
- Never alter or remove an installed package because its upstream revision disappeared.
