/**
 * Autopilot Worker
 *
 * When the leader and all workers are idle and autopilot is enabled,
 * generates a continuation message by invoking the advisor as the user.
 * The advisor uses the same prompt and model as the manual message_advisor
 * tool; the only difference is the terminal message context (no
 * <message_advisor> tag, and autopilot state is active).
 *
 * The generated text is parsed for [AUTOPILOT_OFF] to determine whether
 * the advisor is continuing the conversation or signalling completion.
 *
 * Mechanism: `onProjectionsSettled` fires on every bus event. The worker
 * checks the idle condition on each invocation and only proceeds when
 * all agents are idle AND the root fork is stable AND autopilot is enabled.
 */

import { Effect, Cause } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'

import type { AppEvent } from '../events'
import { AgentLifecycleProjection } from '../projections/agent-lifecycle'
import { TurnProjection } from '../projections/turn'
import { AutopilotStateProjection } from '../projections/autopilot-state'
import { WindowProjection } from '../window'
import {
  buildAdvisorSystemPrompt,
  parseAutopilotResponse,
  collectAdvisorText,
} from '../advisor/orchestrate'
import { advisorWindowToPrompt } from '../window/render'
import { AgentModelResolver } from '../model/model-resolver'
import {
  finalizeModelAttemptFailure,
  modelAttemptRetryability,
  type AgentStreamStartFailure,
} from '../errors'
import { connectionRetrySchedule, TERMINAL_RETRY_EXHAUSTED_MESSAGE } from '../util/retry-backoff'

// =============================================================================
// Concurrent call guard
// =============================================================================

// Safety net: prevents overlapping LLM calls if onProjectionsSettled
// re-enters while a stream is still running.
let isGenerating = false

// =============================================================================
// Worker
// =============================================================================

export const Autopilot = Worker.define<AppEvent>()({
  name: 'Autopilot',

  signalHandlers: () => [],

  onProjectionsSettled: ({ publish, read }) =>
    Effect.gen(function* () {
      // Guard: concurrent call
      if (isGenerating) return

      // Guard: autopilot enabled
      const autopilotState = yield* read(AutopilotStateProjection)
      if (!autopilotState.enabled) return

      // Guard: pending preview exists
      if (autopilotState.pendingContent !== null) return

      // Guard: all agents idle
      const agentStatus = yield* read(AgentLifecycleProjection)
      const allAgentsIdle = Array.from(agentStatus.agents.values()).every(
        (agent) => agent.status === 'idle',
      )
      if (!allAgentsIdle) return

      // Guard: root fork stable
      const rootTurn = yield* read(TurnProjection, null)
      if (!rootTurn) return
      if (
        !(
          rootTurn._tag === 'idle' &&
          rootTurn.triggers.length === 0
        )
      ) {
        return
      }

      // All conditions met: generate autopilot message via advisor
      const modelResolver = yield* AgentModelResolver
      const advisorModel = yield* modelResolver.resolvePrimary('advisor', 'autopilot')
      const httpClient = yield* HttpClient.HttpClient

      const systemPrompt = buildAdvisorSystemPrompt()

      // Build compaction-aware context from WindowProjection
      const windowState = yield* read(WindowProjection, null)
      const prompt = advisorWindowToPrompt({
        windowState,
        systemPrompt,
        autopilotEnabled: true,
        advisorLastAutopilotKnowledge: null,
      })

      const maxTokens = Math.min(advisorModel.profile.maxOutputTokens, 1200)

      isGenerating = true

      // Notify TUI that generation is starting (for spinner state)
      yield* publish({ type: 'autopilot_generation_started', forkId: null })

      try {
        const streamResult = yield* advisorModel.model.stream(prompt, [], { maxTokens }).pipe(
          Effect.provideService(HttpClient.HttpClient, httpClient),
          Effect.retry({
            schedule: connectionRetrySchedule,
            while: (err: AgentStreamStartFailure) => modelAttemptRetryability(err)._tag === 'UpstreamRetryable',
          }),
          Effect.catchAll((err: AgentStreamStartFailure) =>
            Effect.gen(function* () {
              const decision = finalizeModelAttemptFailure({
                failure: err,
                retryCount: Number.MAX_SAFE_INTEGER,
                maxRetries: 0,
              })
              const outcome = decision.outcome
              const message =
                outcome._tag === 'ConnectionFailure'
                  ? TERMINAL_RETRY_EXHAUSTED_MESSAGE
                  : err instanceof Error
                    ? err.message
                    : String(err)

              logger.error({ err }, '[Autopilot] Connection error after retries')
              yield* publish({
                type: 'autopilot_outcome',
                forkId: null,
                result: { _tag: 'error', message },
              })
              return null
            }),
          ),
        )

        if (streamResult === null) return

        const result = yield* collectAdvisorText(streamResult)
        const parsed = parseAutopilotResponse(result.text)

        switch (parsed._tag) {
          case 'message': {
            yield* publish({
              type: 'autopilot_outcome',
              forkId: null,
              result: { _tag: 'success', content: parsed.content },
            })
            break
          }
          case 'finish': {
            if (parsed.content !== null) {
              yield* publish({
                type: 'autopilot_outcome',
                forkId: null,
                result: { _tag: 'success', content: parsed.content },
              })
            }
            yield* publish({
              type: 'autopilot_toggled',
              forkId: null,
              enabled: false,
            })
            break
          }
        }
      } finally {
        isGenerating = false
      }
    }).pipe(
      Effect.catchAllCause((cause) =>
        Effect.gen(function* () {
          isGenerating = false
          logger.error({ cause: Cause.pretty(cause) }, '[Autopilot] Error in generation')
          yield* publish({
            type: 'autopilot_outcome',
            forkId: null,
            result: { _tag: 'error', message: Cause.pretty(cause) },
          })
        }),
      ),
    )
})
