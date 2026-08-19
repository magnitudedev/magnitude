---
applies_to:
  - packages/acn-protocol/src/**
  - packages/acn/src/attachments/**
  - packages/acn/src/session-*.ts
  - packages/client-common/src/**
  - cli/src/features/composer/**
  - web/src/components/composer.tsx
  - web/src/lib/message-uploads.ts
  - web/src/app.tsx
  - web/src/components/messages/**
  - packages/agent/src/display/timeline-projection.ts
---

# Message uploads

## Model and ownership

A message upload is a client-host file snapshot submitted with a user message. It is distinct from
a file mention: an upload transfers bytes that the agent host may not otherwise access, while a
mention refers to an existing agent-host path.

The client owns unsent upload drafts and their presentation. The ACN owns authoritative validation,
session-scoped materialization, and admission with the message. A client path is never submitted as
an agent-host path.

Uploads are part of `SendMessage` or an initial `CreateSession` message. They are not pre-uploaded by
a separate client workflow. The message is accepted only after every upload has materialized; a
failure before event admission leaves no user-message event and the client restores the submitted
draft without overwriting input authored after submission. An acknowledgement lost after possible
admission is not rejection: the exact optimistic message remains until authoritative display with
the same message identity reconciles it.

## Materialization

Image uploads use the existing durable image-capture path and are limited to 10 MiB of submitted
bytes. Text-file uploads must be strict UTF-8,
contain no NUL byte, and contain at most 500 KiB. ACN copies an accepted text file into the session
scratchpad and appends one trailing file-mention occurrence for that captured path. The existing
mention resolver remains the authority for turning the file into model context.

One message may contain at most 20 uploads and at most 25 MiB of decoded upload bytes in aggregate.
The protocol bounds the collection and individual encoded values; ACN validates canonical base64,
decoded individual sizes, and the decoded aggregate before writing anything. Scratchpad filenames
are claimed with exclusive creation so concurrent same-name submissions cannot overwrite one
another.

The durable user-message event retains images as image attachments and uploaded text as trailing
mentions. Display projection exposes images and trailing mentions as message attachments; inline
mentions remain represented by their authored spans and are not duplicated as attachment rows.

## Client behavior

Composer uploads survive ordinary component remounts within one client connection. Selecting,
dropping, pasting, or removing a file is a user-event action, not reactive synchronization. Picker,
drop, and paste ingestion share one validation path in each client environment.

The composer presents uploads in one bounded horizontal row above its text input. Submission clears
the visible draft immediately and projects the exact pending message. Definite rejection removes
that projection. It restores the submitted text and its positional mentions only while the text
draft is still empty, and prepends the rejected uploads to any files attached since submission.
Authoritative display with the same message identity replaces the pending projection after
acceptance.

## Guarantees

- Client-local paths never cross into agent-host path APIs.
- Picker filtering is advisory; ACN validates every upload authoritatively.
- Binary, invalid UTF-8, NUL-containing, oversized, and unsupported image uploads are rejected.
- Text, image, mixed, multiple, and attachment-only messages are supported.
- Uploaded text reaches the agent through the existing trailing-mention lifecycle.
- Replay reconstructs the same visible attachments and resolved model context.
- Failed submission does not silently discard the user's upload draft.
