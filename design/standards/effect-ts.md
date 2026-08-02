---
applies_to:
  - "**/*.ts"
  - "**/*.tsx"
---

# Effect TS

Effect TS should be used for all new code and all code adjacent to other Effect TS code.
It should be adopted idiomatically and in full.

## Principles/definitions

Defininition: **Effectful** code is code which idiomatically and fully adopts Effect and comprehensively uses its patterns and primitives.

Definition: An **Effect boundary** is any boundary between effectful code and non-effectful code. This may include calling Effectful code from syncronous or asyncronous (Promise-based) code, or calling Promise-based code from Effectful code.

Principle 1: **Minimize Effect boundaries**

## Common anti-patterns

- Excessive use of promises in effect code. Use Effects, not promises.
- `tryPromise`, `promise`, and other Effect/async boundary code primitives should EXCLUSIVELY be used at explicitly intentional boundaries. If a given volume of code is meant to be Effectful, it must be fully Effectful, and never include unnecessary boundaries or dipping into promise-land.




## Errors

### Error anti-patterns

#### Cause-wrapping Errors

The following is a HACK that disguises an untyped error as a typed error.
Putting unknown cause into a container is effectively as bad as passing unknown.
The ONLY time such a pattern should ever be used is if there is a boundary that is completely out of our control that is completley unidentifiable.
The ONLY valid use of this pattern is if using catchAllCause, catching genuine defects that have no identifiable structure that we want to pass up as a not-defect.
THERE IS NO OTHER JUSTIFIABLE USE OF THIS PATTERN.

```ts
export class MyError extends Data.TaggedError("MyError")<{
  // ...
  readonly cause?: unknown
}> {}
```

INSTEAD: Make tagged errrors that carry meaningful data.
