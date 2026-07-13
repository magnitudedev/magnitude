/**
 * Effect-aware test assertion helpers for VCS tests.
 */
import { Effect } from "effect"
import {
  VcsError,
  OperationNotFound,
  InvalidPointInTime,
  CorruptSnapshot,
  type VcsFailure,
} from "../../src/errors"

/** Run an Effect and return its success value. Throws on failure. */
export async function runEffect<A, E>(
  eff: Effect.Effect<A, E>,
): Promise<A> {
  return Effect.runPromise(eff as Effect.Effect<A, never>)
}

/** Run an Effect and expect it to fail. Returns the error. */
export async function expectFailure<E>(
  eff: Effect.Effect<unknown, E>,
): Promise<E> {
  const result = await Effect.runPromise(
    Effect.either(eff as Effect.Effect<unknown, E>),
  )
  if (result._tag === "Right") {
    throw new Error(`Expected Effect to fail, but it succeeded with: ${JSON.stringify(result.right)}`)
  }
  return result.left
}

/** Run an Effect and expect it to fail with a specific error type. */
export async function expectFailureWith<E extends VcsFailure>(
  eff: Effect.Effect<unknown, VcsFailure>,
  check: (error: VcsFailure) => error is E,
): Promise<E> {
  const error = await expectFailure(eff)
  if (!check(error)) {
    throw new Error(`Expected specific error type, got: ${String(error)}`)
  }
  return error
}

// ── Error type guards ────────────────────────────────────────────────

export function isVcsError(error: VcsFailure): error is VcsError {
  return error instanceof VcsError
}

export function isOperationNotFound(error: VcsFailure): error is OperationNotFound {
  return error instanceof OperationNotFound
}

export function isInvalidPointInTime(error: VcsFailure): error is InvalidPointInTime {
  return error instanceof InvalidPointInTime
}

export function isCorruptSnapshot(error: VcsFailure): error is CorruptSnapshot {
  return error instanceof CorruptSnapshot
}
