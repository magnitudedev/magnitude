---
name: ts-generics
description: Use when working with advanced TypeScript generics — variance issues, conditional types, type parameter switching, or deciding between any/unknown/casts.
---

# TypeScript Generics

## Contents

**Principles**
1. Variance
2. `any` and `unknown`
3. Casts
4. Conditional Type Mechanics

**Patterns**
5. Never-Switching
6. `any` for Acceptance, Conditionals for Extraction
7. Distributive vs Non-distributive Conditionals
8. `infer` and Type Extraction
9. Mapped Types with Generics
10. Template Literal Types
11. Bounded Polymorphism
12. Recursive Conditional Types

---

# Principles

## 1. Variance

Type compatibility in TypeScript depends on *variance* — how subtype relationships flow through generic containers.

- **Covariant** positions preserve subtype direction: `Dog → Animal` implies `Array<Dog> → Array<Animal>`
- **Contravariant** positions reverse it: `Dog → Animal` implies `((a: Animal) => void) → ((d: Dog) => void)`
- **Invariant** positions require exact matches: neither direction works

Function parameters are contravariant (under `strictFunctionTypes`). Many Effect types have contravariant or invariant parameters:

```typescript
type SignalsConfig = Record<string, Signal<unknown>>

// This fails — Signal wraps PubSub with contravariant params
const config: SignalsConfig = {
  chunk: displayChunk  // Signal<{ chunk: string }> not assignable to Signal<unknown>
}
```

`unknown` is the top type of the normal type system, but in contravariant/invariant positions it is *not* a supertype of more specific types. This is the fundamental reason `any` exists as an escape hatch.

## 2. `any` and `unknown`

`unknown` is the type-safe top type. It accepts any value but requires narrowing before use — the compiler enforces this.

`any` is the type-unsafe escape hatch. It both accepts and provides any value, bypassing all type checking.

### When `any` is necessary

`any` is necessary in exactly one situation: **accepting values where variance prevents `unknown` from working as a supertype**. When a type has contravariant or invariant parameters and you need to store heterogeneous instances, `unknown` won't work and `any` is the only option:

```typescript
type Config = Record<string, Signal<any>>  // Signal<unknown> would reject Signal<Specific>
```

`any` is also used (but not strictly necessary) as a structural wildcard in conditional type pattern-matching, where `infer` extracts the real type:

```typescript
type ExtractValue<T> = T extends Container<any, infer V> ? V : never
// any is just the match shape — V carries the real type
```

### When `any` is safe

| Condition | Why |
|-----------|-----|
| Used only for matching in conditional types | `infer` extracts the real type; `any` is just the match pattern |
| Backed by compile-time constraints | e.g., `ValidSignalsConfig<T>` only allows signals from specific projections |
| Never exposed in public API | Public types use extracted types, not `any` |

### `any` is never appropriate for convenience

If the reason for `any` is "TypeScript can't figure this out" or "it's easier," the right answer is to fix the types. Every `any` should have a principled justification (variance bypass, or constrained match position).

### `as any` is never necessary

`x as any` is always a code smell. It throws away all type information on both sides of the expression. There is always a more specific alternative:

- If you need to widen: use `as unknown` (or a more specific supertype)
- If you need to narrow: use a proper type guard or a single `as` to the target type
- If TypeScript can't verify the narrowing: use `as TargetType` — at least the intent is clear
- If the type hierarchy doesn't allow a direct cast: restructure the types so it does

`as any` hides the intent. It makes the cast invisible in code review and impossible to find later when types change.

### `as unknown as X` is never necessary

A double cast (`as unknown as X`) indicates a broken type relationship. Every instance means either:

1. **The source type should be a subtype of X** — fix the type hierarchy so `as X` works directly
2. **The source type genuinely isn't related to X** — then you're lying to the compiler and should reconsider the design
3. **Variance is blocking the direct cast** — use `any` in the acceptance position instead (see Pattern 6)

Double casts bypass two layers of type safety. They are always fixable with better type design.

## 3. Casts

### When casts are safe

**Widening to `unknown`** — always safe, since everything is a subtype of `unknown`:
```typescript
const layers = projections.map(p => p.Layer) as Layer.Layer<unknown, never, unknown>[]
```

**Asserting a computed type guaranteed by constraints** — the constraint ensures the shape, but TypeScript can't see through indirection (dynamic property access, `Object.keys()` iteration):
```typescript
const signal = expose.signals[name] as Signal<unknown>
// ValidSignalsConfig<TProjections> guarantees this comes from projections
```

**Combining layer outputs** — the combined type is computed from known constituents:
```typescript
const AppLayer = ... as Layer.Layer<TAllServices, never, never>
// TAllServices is computed from ExtractProjectionOutputs<TProjections>
```

The general principle: **constraint + cast = safe**. Constrain at the definition site, cast at the usage site. The cast is safe because the constraint guarantees the shape — TypeScript just can't follow the indirection.

### When casts are dangerous

**Narrowing without constraint** — asserting a specific type from an unconstrained value:
```typescript
const data = response as UserData  // No guarantee response is UserData
```

**Using `any` in public API** — the caller gets no type safety:
```typescript
function getData(): any { ... }
```

### Casts should always be to a specific type

Every cast should express a clear intent: "I know this is X." The target type should be as specific as possible. If you can't name the specific type you're casting to, the cast isn't justified.

## 4. Conditional Type Mechanics

Conditional types (`T extends U ? A : B`) are the engine that powers generic type-level programming. Their behavior depends on a few key mechanics:

### Distributive behavior

When `T` is a bare type parameter, the conditional distributes over unions:

```typescript
type Wrap<T> = T extends string ? { s: T } : { n: T }
type W = Wrap<string | number>
// = { s: string } | { n: number }  — distributed over each member
```

This is useful for mapping over union members. But it's wrong when you want to check the type as a whole.

### Non-distributive behavior

Wrapping in tuples prevents distribution:

```typescript
type IsNever<T> = [T] extends [never] ? true : false
type R = IsNever<string | number>  // true — not distributed
```

**Rule of thumb:** Bare `T extends U` for union mapping. `[T] extends [U]` for whole-type checks.

### `never` distributes to `never`

For any distributive conditional type, `never` distributes to the empty union — which is `never`:

```typescript
type Wrap<T> = T extends string ? { s: T } : { n: T }
type W = Wrap<never>  // never (distributed over zero members)
```

This is why `[T] extends [never]` is needed to actually *check* for `never` — the bare form collapses to `never` before the check can evaluate.

---

# Patterns

## 5. Never-Switching

A type parameter defaults to `never`. A non-distributive conditional switches between an erased mode (loose types for runtime) and a concrete mode (full types for compile-time safety):

```typescript
type MyType<T = never> = [T] extends [never]
  ? ErasedVersion     // Default: loose types for runtime
  : ConcreteVersion<T> // Specified: full type safety
```

### How it works

`never` is the bottom type — assignable to anything, but nothing is assignable to it. When the parameter is `never` (default), the conditional selects the erased version with `unknown`/`any`/`string` types. When specified, it selects the concrete version parameterized by `T`.

The tuple wrapping (`[T] extends [never]`) prevents distributive behavior, ensuring the check treats `T` as a whole rather than splitting over unions.

### Erased/Concrete pairing

Always define both versions:

```typescript
interface MyTypeErased {
  // Loose types: unknown, any, string
}

type MyTypeConcrete<T> = {
  // Strict types using T
}

type MyType<T = never> = [T] extends [never] ? MyTypeErased : MyTypeConcrete<T>
```

### Examples from the codebase

**ToolDefinition** (`packages/tools/src/tool-definition.ts`):

```typescript
export type ToolDefinition<
  TInput = never,
  TOutput = never,
  ...6 more never defaults...
> = [TInput] extends [never]
  ? ToolDefinitionErased
  : ToolDefinitionConcrete<TInput, TOutput, ...>
```

Erased: `inputSchema: Schema.Schema.Any`, `execute(input: unknown, ...): Effect<unknown, ...>`
Concrete: `inputSchema: Schema.Schema<TInput, TInputEncoded, never>`, `execute(input: TInput, ...): Effect<TOutput, ...>`

**RoleDefinition** (`packages/roles/src/types.ts`):

```typescript
export type RoleDefinition<TTools = never, ...> = [TTools] extends [never]
  ? RoleDefinitionErased
  : RoleDefinitionConcrete<TTools & ToolCatalog, ...>
```

**Functional API switching** (`packages/event-core/src/agent/define.ts`):

The pattern can change function signatures, not just object shapes:

```typescript
readonly createClient: [TWorkerRequirements] extends [never]
  ? () => Promise<AgentClient<...>>           // No args needed
  : (req: Layer.Layer<TWorkerRequirements>) => Promise<AgentClient<...>>  // Requirements needed
```

**Field utility types** (`packages/tools/src/bindings.ts`):

```typescript
export type InputFields<T> = [T] extends [never]
  ? string                                    // Accept any field name
  : Extract<keyof T, string>                  // Only known field names

export type XmlArrayChildBinding<T> = [ArrayFields<T>] extends [never]
  ? XmlChildBinding                           // Plain strings
  : { /* typed against specific array field */ }[ArrayFields<T>]
```

**Event types** (`packages/xml-act/src/types.ts`):

```typescript
interface ToolInputChildStarted<TInput = unknown, B = unknown> {
  readonly field: [BindingChildren<B>] extends [never] ? string : ChildBindingField<...>
  readonly attributes: [BindingChildren<B>] extends [never]
    ? Readonly<Record<string, string | number | boolean>>
    : ChildAttrsPick<...>
}
```

### Default parameter cascade

When a type has multiple parameters, default all to `never`. Only the primary parameter needs the `[T] extends [never]` check — if it's `never`, the others are irrelevant:

```typescript
type ComplexType<A = never, B = never, C = never> =
  [A] extends [never] ? Erased : Concrete<A, B, C>
```

### Optional schema fields

Sub-pattern for fields that are optional when the type is erased:

```typescript
readonly emissionSchema?: [TEmission] extends [never]
  ? Schema.Schema.Any | undefined
  : Schema.Schema<TEmission, TEmissionEncoded, never>
```

## 6. `any` for Acceptance, Conditionals for Extraction

When variance blocks `unknown` as a supertype, use `any` to accept heterogeneous values and conditional types with `infer` to extract the real types for the public API.

### The three-layer structure

**Layer 1 — Accept with `any`:**

```typescript
type SignalsConfig = Record<string, Signal<any>>
```

`any` bypasses variance checking. Safe because it doesn't escape to the public API.

**Layer 2 — Extract with conditionals:**

```typescript
type SignalValue<T> = T extends Signal<infer V> ? V : never

type ProjectionStateFromTag<T> =
  T extends { readonly Tag: Context.Tag<infer Service, any> }
    ? Service extends { readonly get: Effect.Effect<infer S, any, any> }
      ? S
      : never
    : never
```

The `any` in the pattern match is just for structural matching — `infer` extracts the real type.

**Layer 3 — Re-apply extracted types to public API:**

```typescript
type ClientSignals<T extends Record<string, Signal<any>>> = {
  [K in keyof T]: (callback: (value: SignalValue<T[K]>) => void) => () => void
}
```

Now `client.on.chunk` receives `{ chunk: string }` — the real type, not `any`.

### The constraint + cast principle at work

The `any` acceptance is safe because it's constrained: `ValidSignalsConfig<TProjections>` ensures only valid signals are accepted. At runtime, when iterating over `Object.keys()`, TypeScript can't follow the constraint — so a cast is used, backed by the compile-time guarantee.

### Full type flow

```typescript
// 1. Definition — constrained
Agent.define<AppEvent>()({
  projections: [DisplayProjection],
  expose: {
    signals: {
      chunk: displayChunk  // ✓ Allowed — from DisplayProjection
      // orphan: orphanSignal  // ✗ Error — not from any projection
    }
  }
})

// 2. Internal — any/casts backed by constraints
const signal = expose.signals[name] as Signal<unknown>

// 3. Public API — extracted types
client.on.chunk((val) => {
  val.chunk  // ✓ string
  val.bad    // ✗ Error
})
```

| Layer | Strategy | Why |
|-------|----------|-----|
| Config acceptance | `any` | Bypasses variance |
| Type extraction | Conditional types with `infer` | Gets real types |
| Internal implementation | Casts backed by constraints | TS can't infer through iteration |
| Public API | Extracted types | Full type safety |

## 7. Distributive vs Non-distributive Conditionals

### Distributive (bare `T`)

When `T` is a bare type parameter, `T extends U ? A : B` distributes over unions:

```typescript
type Result = string | number extends string ? "yes" : "no"
// = (string extends string ? "yes" : "no") | (number extends string ? "yes" : "no")
// = "yes" | "no"

type Wrap<T> = T extends string ? { s: T } : { n: T }
type W = Wrap<string | number>  // { s: string } | { n: number }
```

### Non-distributive (tuple-wrapped `[T]`)

```typescript
type NotNever<T> = [T] extends [never] ? false : true
type R1 = NotNever<string>          // true
type R2 = NotNever<never>           // false
type R3 = NotNever<string | number> // true — not distributed
```

### When to use which

- Union mapping (apply per-member logic) → bare `T extends U`
- Whole-type checking (is T never? is T exactly X?) → `[T] extends [U]`

### `never` distributes to `never`

In distributive mode, `never` distributes over zero members — producing `never`:

```typescript
type Wrap<T> = T extends string ? { s: T } : { n: T }
type W = Wrap<never>  // never
```

This is why `[T] extends [never]` is required to actually detect `never`.

## 8. `infer` and Type Extraction

`infer` introduces a type variable that TypeScript fills in by matching the `extends` pattern:

```typescript
type Unwrap<T> = T extends Promise<infer U> ? U : T
type R = Unwrap<Promise<string>>  // string
```

### Covariant vs contravariant `infer`

`infer` in covariant positions (output side) narrows to the specific type. In contravariant positions (input side), it widens to the union of all possible types:

```typescript
// Covariant: narrows
type ReturnOf<T> = T extends (...args: any[]) => infer R ? R : never
type R1 = ReturnOf<(x: string) => number>  // number

// Contravariant: widens
type ParamOf<T> = T extends (x: infer P) => any ? P : never
type R2 = ParamOf<(x: string | number) => void>  // string | number
```

### Nested `infer`

Multiple `infer` positions and nesting are allowed:

```typescript
type DeepAwaited<T> = T extends Promise<infer U> ? DeepAwaited<U> : T
type R = DeepAwaited<Promise<Promise<string>>>  // string
```

### `infer` with constraints

TypeScript 4.7+ supports constraining inferred types:

```typescript
type FirstIfString<T> = T extends [infer S extends string, ...any[]]
  ? S
  : never

type R1 = FirstIfString<["hello", number]>  // "hello"
type R2 = FirstIfString<[number, string]>   // never
```

## 9. Mapped Types with Generics

### Key remapping with `as`

The `as` clause transforms or filters keys:

```typescript
// Filter: only string-valued keys
type StringKeys<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K]
}

// Transform: prepend "get"
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K]
}
```

### Combining with conditional types

```typescript
type PickByType<T, ValueType> = {
  [K in keyof T as T[K] extends ValueType ? K : never]: T[K]
}
```

### Homomorphic mapped types

Iterating over `keyof T` preserves modifiers (readonly, optional) and references the original property type. Non-homomorphic mappings lose these.

```typescript
type Readonly<T> = { readonly [K in keyof T]: T[K] }    // Preserves optional
type Optional<T> = { [K in keyof T]?: T[K] }            // Preserves readonly
type Fake<K extends string> = { [P in K]: string }       // Not homomorphic — no preservation
```

## 10. Template Literal Types

Generics combine with template literals for type-safe string patterns:

```typescript
type EventName<T extends string> = `on${Capitalize<T>}`
type R = EventName<"click">  // "onClick"
```

With mapped types:

```typescript
type EventHandlers<T extends string> = {
  [K in T as `on${Capitalize<K>}`]: (event: Event<K>) => void
}
// { onClick: ...; onFocus: ... }
```

Built-in intrinsic string types: `Uppercase<T>`, `Lowercase<T>`, `Capitalize<T>`, `Uncapitalize<T>`.

## 11. Bounded Polymorphism

### When bounds are needed

Unbounded generics (`T` alone) work for passing types through. When you need to *use* the type — access properties, call methods — add bounds:

```typescript
function length<T extends { length: number }>(x: T) {
  return x.length  // ✓
}
```

### Bounds with `keyof`

```typescript
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key]
}
```

### F-bounded polymorphism

A type referencing itself in its own constraint. Powerful but can create hard-to-debug type relationships — use sparingly:

```typescript
interface Comparable<T extends Comparable<T>> {
  compareTo(other: T): number
}
```

## 12. Recursive Conditional Types

TypeScript supports recursive conditional types (since 4.1):

```typescript
type DeepReadonly<T> = T extends primitive
  ? T
  : { readonly [K in keyof T]: DeepReadonly<T[K]> }

type PathKeys<T> = T extends object
  ? { [K in keyof T & string]: K | `${K}.${PathKeys<T[K]>}` }[keyof T & string]
  : never
```

### Depth limits

TypeScript has an internal recursion depth limit (~25 levels for type instantiation). Hitting it causes "Type instantiation is excessively deep" errors. Workarounds:

- Flatten recursion with accumulator types
- Use concrete base cases to terminate early
- Split deeply recursive types into composable layers

---

## Quick Reference

| Pattern | Syntax | Use When |
|---------|--------|----------|
| Never-switching | `[T] extends [never] ? Erased : Concrete<T>` | Type needs both generic and erased modes |
| `any` acceptance + extraction | `Signal<any>` + `T extends Signal<infer V> ? V` | Variance blocks `unknown` as supertype |
| Constraint + cast | Constrain at definition, cast at usage | TS can't infer through indirection |
| Non-distributive check | `[T] extends [U]` | Whole-type check, not union distribution |
| Functional API switch | `[T] extends [never] ? () => R : (arg: T) => R` | Function arity varies by type param |
| Optional erased field | `[T] extends [never] ? Schema.Any \| undefined : Schema<T>` | Field optional when type is erased |
| Key filtering | `[K in keyof T as T[K] extends V ? K : never]` | Pick keys by value type |
| Bounded polymorphism | `T extends { length: number }` | Need to use T's properties |
| Recursive conditional | `T extends object ? { [K in keyof T]: Rec<T[K]> } : T` | Deep type transformations |
| `infer` extraction | `T extends Wrapper<infer V> ? V : never` | Get real type from wrapper |
| Template literal | `` `on${Capitalize<T>}` `` | Type-safe string patterns |

### Anti-patterns

| Anti-pattern | Why | Fix |
|-------------|-----|-----|
| `as any` | Throws away all type info on both sides | Use a specific type or `as unknown` |
| `as unknown as X` | Bypasses two layers of type safety | Fix the type hierarchy so `as X` works |
| `any` in public API | Consumer gets no type safety | Use conditional extraction |
| Cast without constraint | Lying to the compiler | Add a constraint that guarantees the shape |
| Over-constraining `T` | Unnecessary rigidity | Only bound when you need to use T's properties |
| Bare `T extends never` | Distributes over `never`, always returns `never` | Use `[T] extends [never]` |
