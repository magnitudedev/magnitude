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

## State machines

Any nontrivial system or mechanism representable as a finite set of states and transitions should be represented formally as a state machine, in order to enforce valid transitions and states.

Use `defineFSM` from the utils package to define state machines. NEVER create helpers that bypass the typed transition mechanisms, else the purpose of the FSM is defeated.

## Data structures

### Strict representiveness

Every data structure should represent exclusively valid and accepted states, and should never be able to represent states that should not exist.

Example: Instead of adding more fields that are optional, use discriminated unions.

Avoid "data bags" - bags that can represent various states, rather than coherenet composition of structures in a way that guarantees alignment with what the code should consider to be valid.

### Data nesting

Nest data structures as needed to achieve strict representiveness.
Nest discriminated unions in other structures when needed, for example.

### Branded types

Use Schema.brand when a field is of a meaningful semantic purpose that should be made explicit for the purpose of ensuring consumers use it correctly.

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

#### Pseudo-tag dimensions

Creating a string or other property that represents variations of an error and then including bags of information

```ts
export class MyError extends Data.TaggedError("MyError")<{
  // ...
  readonly code: string // some "error code"
  // potentially, other "bag" fields that might exist only when code is a specific value
}> {}
```

If the error variants are finite and known, they should be represented as distinct tagged errors.

If a union of these is needed as a reusable concept, define a type with is an actual union of the tagged errors.

