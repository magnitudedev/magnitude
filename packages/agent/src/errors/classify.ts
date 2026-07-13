/**
 * Error classification.
 *
 * Single source of truth for turning raw errors (HTTP responses, exceptions,
 * Effect causes) into the canonical TurnOutcome type.
 */

import { Cause } from 'effect'
import type { TurnOutcome, UnexpectedErrorDetail } from '../events'
import { describeThrown, stackTraceLines } from './formatters'

function unknownErrorDetail(err: unknown): UnexpectedErrorDetail {
  if (Cause.isCause(err)) {
    const failure = Cause.failureOption(err)
    if (failure._tag === 'Some') return unknownErrorDetail(failure.value)
    return {
      _tag: 'Unknown',
      message: [
        'Unexpected runtime error',
        'cause:',
        Cause.pretty(err),
      ].join('\n'),
    }
  }

  return {
    _tag: 'Unknown',
    message: [
      'Unexpected runtime error',
      `error: ${describeThrown(err)}`,
      ...stackTraceLines(err),
    ].join('\n'),
  }
}

/**
 * Classify an arbitrary unknown error (thrown exception, Effect cause, etc.)
 * into a TurnOutcome. Used for non-agent error sites — framework errors,
 * render boundaries, overlay fetch failures.
 */
export function classifyUnknownError(err: unknown): TurnOutcome {
  return {
    _tag: 'UnexpectedError',
    detail: unknownErrorDetail(err),
    requestId: null,
  }
}
