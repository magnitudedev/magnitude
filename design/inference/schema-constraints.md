---
applies_to:
  - inference/engine/templates/native/**
  - inference/engine/chat/src/schema.rs
  - inference/engine/chat/src/conformance.rs
  - inference/engine/chat/src/request.rs
  - inference/engine/src/chat/session.rs
---

# Schema constraints

Tool argument schemas and JSON output schemas constrain generation best
effort: exactly where a grammar can express the schema, loosened and reported
where it cannot. A schema never fails a request unless it is not a valid JSON
Schema.

## Admission

- A schema is admitted when it is valid under its draft and every reference
  resolves within the document. The draft is the one `$schema` names; a
  document naming none, or a dialect the engine does not know, is read as
  2020-12. Remote references are never fetched and are invalid.
- Admission is the only schema-caused request error. Nothing downstream of
  admission rejects a schema.
- Formats are annotations, as JSON Schema 2019-09 and later define them.
  Recognized formats may still shape the grammar.
- `strict` is accepted on every protocol and changes nothing: every request is
  constrained as tightly as its schemas allow.

## Lowering

Every admitted schema lowers to a grammar. Each keyword is one of:

- **Enforced**: the grammar admits exactly the values the keyword allows, or a
  subset every valid output can still take (a fixed property order, closed
  objects, bounded numbers in plain decimal notation of at most 15 digits, 18
  for integers, so a value within its bounds stays within them when read as
  a double).
- **Loosened**: the grammar admits a superset, and the keyword is recorded as a
  relaxation of its tool or output.
- **Annotation**: no validation meaning; ignored exactly. Unknown keywords are
  annotations.

A keyword with validation meaning is never ignored without a relaxation.
Type-specific keywords constrain only values of their type. `allOf`, `$ref`
siblings and `anyOf` siblings compose by intersection; alternatives that admit
no value are dropped. A value position that admits no value at all is
loosened to any value and recorded as unsatisfiable.

A model's output syntax may be unable to carry a keyword (raw-text arguments
carry no string constraints; a dictionary syntax carries no numeric bounds)
or to write a value at all (a string containing the syntax's own delimiter).
The first is a loosening; the second leaves the value out of the grammar and
is recorded as unrepresentable. Neither fails the request.

An object that lists properties is closed unless its schema explicitly allows
more.

## String text

Every string grammar admits only valid JSON string text: no raw control
characters, and `\u` escapes of surrogates only as complete pairs, each pair
one character. A pattern constrains a string's value; the grammar constrains
its JSON text, so pattern grammars match characters JSON must escape only as
escape sequences. Unanchored patterns match anywhere in the string. Patterns
with no grammar equivalent (lookaround, backreferences) are loosened.

## Conformance

Every completed tool call and every naturally completed JSON output is checked
against its admitted schema. Output is always published as generated; the
protocols have no place for a conformance verdict.

- A violation under a loosened schema, or under no grammar at all, is expected,
  and is reported with the relaxations that allowed it.
- A violation under a schema the grammar enforces exactly is a grammar defect,
  and is reported as one.
- Output that stopped before completing is not checked.

## Acceptance

- Every valid schema prepares in every template family; only invalid schemas
  are rejected.
- Bounded numbers admit exactly the plain decimals within their bounds.
- Pattern grammars admit only valid JSON strings, and the pattern's values.
- Every loosened keyword is recorded, and the grammar still admits the values
  the keyword would have rejected.
