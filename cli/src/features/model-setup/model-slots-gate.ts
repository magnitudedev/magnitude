import type { ModelSlotsUnavailable } from '@magnitudedev/sdk'

/**
 * Provider discovery failures are expected when an optional provider has not
 * been configured. Only a failure to read the saved slot configuration blocks
 * the CLI from opening a session.
 */
export function blockingModelSlotsFailure(
  state: Pick<ModelSlotsUnavailable, 'failures'>,
): string | null {
  const failures = state.failures.filter(
    (failure) => failure._tag === 'configuration_unavailable',
  )

  return failures.length > 0
    ? failures.map((failure) => failure.message).join('; ')
    : null
}
