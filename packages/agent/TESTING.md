# Testing

Tests use [vitest](https://vitest.dev/) with [@effect/vitest](https://effect.website/docs/testing) for Effect-native lifecycle management.

```bash
bunx vitest run          # run all tests
bunx vitest              # watch mode
```

## Structure

- **`src/test-harness/__tests__/`** — Tests of the test harness itself (verifies harness mechanics work correctly)
- **`tests/`** — Agent integration tests that use the harness to verify agent behavior

## Writing tests

Tests use `it.live()` from `@effect/vitest` with the `TestHarness` Effect service:

```typescript
import { describe, it, expect } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'

it.live('my test', () =>
  Effect.gen(function* () {
    const h = yield* TestHarness
    yield* h.user('hello')
    const completed = yield* h.wait.turnCompleted(null)
    expect(completed.result.success).toBe(true)
  }).pipe(Effect.provide(TestHarnessLive()))
)
```

`TestHarnessLive` accepts options for custom tool overrides, seeded files, session context, etc.
