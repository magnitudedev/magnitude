/**
 * Observer Worker
 *
 * Evaluates every agent turn for signs of struggle, tunneling,
 * or strategy decay. Fires after every turn_outcome event, regardless
 * of whether the turn will chain-continue or go idle.
 *
 * Concurrency: at most one observer call per fork at a time.
 * Each fork has an observer loop fiber that runs observeOnce, then
 * checks if a new turn_outcome arrived while it was running. If so,
 * it re-evaluates with the latest pending event. This ensures no turn
 * goes unobserved without stacking concurrent calls.
 *
 * When escalation is flagged (escalate: true):
 * - Leader escalation queues an advisor-required communication for the next
 *   leader turn. The turn projection claims that communication at turn start.
 */

import { Effect, Stream, Cause, Ref, Fiber } from 'effect'
import { Worker, type WorkerReadFn, type PublishFn } from '@magnitudedev/event-core'
import { createHarness, type HarnessEvent, type ToolInputRejected, type TurnOutcome } from '@magnitudedev/harness'
import { logger } from '@magnitudedev/logger'
import type { StreamFailure } from '@magnitudedev/ai'

import type { AppEvent } from '../events'
import { AgentLifecycleProjection } from '../projections/agent-lifecycle'
import { SessionContextProjection } from '../projections/session-context'
import { TurnProjection, hasPendingAdvisorRequirement } from '../projections/turn'
import { WindowProjection, type ForkWindowState } from '../window'
import { observerWindowToPrompt, getObserverSystemPrompt } from './prompt'
import { AgentModelResolver } from '../model/model-resolver'
import { AgentModelOperationContextTag } from '../model/agent-model'
import { modelAttemptRetryability, type AgentStreamStartFailure } from '../errors'
import { connectionRetrySchedule } from '../util/retry-backoff'
import { createId } from '../util/id'
import { observerToolkit, type EscalateInput } from './schema'
import type { ObserverJustification } from './justifications'
import {
  ObserverIdle,
  ObserverRunLifecycle,
  ObserverStateTag,
  initialObserverForkState,
  type ObserverForkState,
  type ObserverTurnOutcomeEvent,
} from './state'

// =============================================================================
// Report state tracking
// =============================================================================

const PASS_TOOL_NAME = 'pass'
const ESCALATE_TOOL_NAME = 'escalate'

/** Tool names the observer may call. */
const VALID_OBSERVER_TOOLS = [PASS_TOOL_NAME, ESCALATE_TOOL_NAME] as const

type ObserverFailureOutcome = Exclude<TurnOutcome, { readonly _tag: 'Completed' }>
type CompletedTurnOutcome = Extract<TurnOutcome, { readonly _tag: 'Completed' }>

type ObserverReportFailure =
  | { readonly _tag: 'ToolInputRejected'; readonly event: ToolInputRejected }
  | { readonly _tag: 'TurnOutcome'; readonly outcome: ObserverFailureOutcome }

type ObserverProtocolViolation =
  | { readonly _tag: 'CompletedWithoutReport'; readonly outcome: CompletedTurnOutcome }
  | { readonly _tag: 'UnexpectedTerminalOutcome'; readonly outcome: TurnOutcome }
  | { readonly _tag: 'MissingTerminalOutcome' }

type ObserverReportStatus =
  | { readonly _tag: 'Success'; readonly toolName: 'pass' | 'escalate'; readonly justification: ObserverJustification | null; readonly reasoning: string }
  | { readonly _tag: 'Failure'; readonly failure: ObserverReportFailure }
  | { readonly _tag: 'ProtocolViolation'; readonly violation: ObserverProtocolViolation }

type ObserverReportState =
  | { readonly _tag: 'AwaitingReport'; readonly reasoning: string }
  | ObserverReportStatus

function observerFailureOutcome(outcome: TurnOutcome): ObserverFailureOutcome | null {
  switch (outcome._tag) {
    case 'Completed':
      return null
    case 'ToolInputValidationFailure':
    case 'ToolExecutionError':
    case 'GateRejected':
      return outcome.toolName === PASS_TOOL_NAME || outcome.toolName === ESCALATE_TOOL_NAME ? outcome : null
    default:
      return outcome
  }
}

function reduceObserverReportState(state: ObserverReportState, event: HarnessEvent): ObserverReportState {
  switch (event._tag) {
    case 'ThoughtDelta': {
      if (state._tag === 'AwaitingReport') {
        return { ...state, reasoning: state.reasoning + event.text }
      }
      if (state._tag === 'Success') {
        return { ...state, reasoning: state.reasoning + event.text }
      }
      return state
    }

    case 'ToolExecutionStarted':
      if (!VALID_OBSERVER_TOOLS.includes(event.toolName as any)) return state
      if (state._tag === 'Failure' || state._tag === 'ProtocolViolation') return state
      if (event.toolName === PASS_TOOL_NAME) {
        return { _tag: 'Success', toolName: 'pass', justification: null, reasoning: state.reasoning }
      }
      if (event.toolName === ESCALATE_TOOL_NAME) {
        return {
          _tag: 'Success',
          toolName: 'escalate',
          justification: (event.input as EscalateInput)?.justification ?? null,
          reasoning: state.reasoning,
        }
      }
      return state

    case 'ToolInputRejected':
      if (!VALID_OBSERVER_TOOLS.includes(event.toolName as any)) return state
      return { _tag: 'Failure', failure: { _tag: 'ToolInputRejected', event } }

    case 'TurnEnd': {
      const failureOutcome = observerFailureOutcome(event.outcome)
      if (failureOutcome) {
        return { _tag: 'Failure', failure: { _tag: 'TurnOutcome', outcome: failureOutcome } }
      }
      if (state._tag === 'Success' || state._tag === 'Failure') return state
      if (event.outcome._tag === 'Completed') {
        return {
          _tag: 'ProtocolViolation',
          violation: { _tag: 'CompletedWithoutReport', outcome: event.outcome },
        }
      }
      return {
        _tag: 'ProtocolViolation',
        violation: { _tag: 'UnexpectedTerminalOutcome', outcome: event.outcome },
      }
    }

    default:
      return state
  }
}

function finalizeObserverReportState(state: ObserverReportState): ObserverReportStatus {
  if (state._tag !== 'AwaitingReport') return state
  return {
    _tag: 'ProtocolViolation',
    violation: { _tag: 'MissingTerminalOutcome' },
  }
}

function observerProtocolViolationLogData(violation: ObserverProtocolViolation) {
  switch (violation._tag) {
    case 'CompletedWithoutReport':
      return {
        violation: violation._tag,
        message: `A tool call was required but no execution event was observed`,
        terminalOutcome: violation.outcome,
      }
    case 'UnexpectedTerminalOutcome':
      return {
        violation: violation._tag,
        message: `A tool call was required but the terminal outcome did not match an expected tool`,
        terminalOutcome: violation.outcome,
      }
    case 'MissingTerminalOutcome':
      return {
        violation: violation._tag,
        message: `A tool call was required but the harness turn ended before a terminal event was observed`,
      }
  }
}

function observerFailureLogData(failure: ObserverReportFailure) {
  if (failure._tag === 'ToolInputRejected') {
    return {
      source: 'ToolInputRejected',
      toolCallId: failure.event.toolCallId,
      providerToolCallId: failure.event.providerToolCallId,
      toolName: failure.event.toolName,
      toolKey: failure.event.toolKey,
      issue: failure.event.issue,
    }
  }

  const outcome = failure.outcome
  switch (outcome._tag) {
    case 'ToolInputValidationFailure':
      return {
        source: 'TurnEnd',
        outcome: outcome._tag,
        toolCallId: outcome.toolCallId,
        providerToolCallId: outcome.providerToolCallId,
        toolName: outcome.toolName,
        toolKey: outcome.toolKey,
        issue: outcome.issue,
      }
    case 'ToolExecutionError':
      return {
        source: 'TurnEnd',
        outcome: outcome._tag,
        toolCallId: outcome.toolCallId,
        providerToolCallId: outcome.providerToolCallId,
        toolName: outcome.toolName,
        toolKey: outcome.toolKey,
        error: outcome.error,
      }
    case 'GateRejected':
      return {
        source: 'TurnEnd',
        outcome: outcome._tag,
        toolCallId: outcome.toolCallId,
        providerToolCallId: outcome.providerToolCallId,
        toolName: outcome.toolName,
      }
    case 'SafetyStop':
      return { source: 'TurnEnd', outcome: outcome._tag, reason: outcome.reason }
    case 'ThoughtLimitExceeded':
      return { source: 'TurnEnd', outcome: outcome._tag, limit: outcome.limit }
    case 'EngineDefect':
      return { source: 'TurnEnd', outcome: outcome._tag, message: outcome.message }
    case 'StreamFailed':
      return {
        source: 'TurnEnd',
        outcome: outcome._tag,
        streamFailure: outcome.terminal.cause._tag,
        streamFailureDetail: streamFailureLogDetail(outcome.terminal.cause),
      }
    default:
      return { source: 'TurnEnd', outcome: outcome._tag }
  }
}

function streamFailureLogDetail(failure: StreamFailure): string {
  switch (failure._tag) {
    case 'StreamOperationalFailure':
      return failure.reason._tag
    case 'StreamProviderError':
      return failure.providerError.code ?? failure.providerError.type ?? 'ProviderError'
    case 'StreamProviderCorrectnessViolation':
      return failure.violation._tag
    case 'StreamClientCorrectnessViolation':
      return failure.evidence._tag
  }
}

// =============================================================================
// Scheduler state helpers
// =============================================================================

type StartObserverCommand = {
  readonly _tag: 'start_observer'
  readonly forkId: string | null
  readonly event: ObserverTurnOutcomeEvent
  readonly pendingEvent: ObserverTurnOutcomeEvent | null
}

type SchedulerCommand = StartObserverCommand | { readonly _tag: 'none' }

const noCommand: SchedulerCommand = { _tag: 'none' }

function forkLabel(forkId: string | null): string {
  return forkId ?? 'root'
}

function getForkEntry(
  state: ReadonlyMap<string | null, ObserverForkState>,
  forkId: string | null,
): ObserverForkState {
  return state.get(forkId) ?? initialObserverForkState()
}

function isPrunable(entry: ObserverForkState): boolean {
  return entry.observer._tag === 'idle' && entry.observer.pendingEvent === null && entry.fiber === null
}

function setForkEntry(
  state: ReadonlyMap<string | null, ObserverForkState>,
  forkId: string | null,
  entry: ObserverForkState,
): Map<string | null, ObserverForkState> {
  const next = new Map(state)
  if (isPrunable(entry)) {
    next.delete(forkId)
  } else {
    next.set(forkId, entry)
  }
  return next
}

function setRunningPendingEvent(
  entry: ObserverForkState,
  event: ObserverTurnOutcomeEvent,
): ObserverForkState {
  if (entry.observer._tag !== 'running') return entry
  return {
    ...entry,
    observer: ObserverRunLifecycle.hold(entry.observer, { pendingEvent: event }),
  }
}

function nextTurnOutcomeCommand(
  forkId: string | null,
  event: ObserverTurnOutcomeEvent,
  entry: ObserverForkState,
): { readonly entry: ObserverForkState; readonly command: SchedulerCommand } {
  switch (entry.observer._tag) {
    case 'idle':
      return {
        entry: { ...entry, observer: new ObserverIdle({ pendingEvent: null }) },
        command: { _tag: 'start_observer', forkId, event, pendingEvent: null },
      }
    case 'running':
      return { entry: setRunningPendingEvent(entry, event), command: noCommand }
  }
}

function cancelObserverForFork(forkId: string) {
  return Effect.gen(function* () {
    const state = yield* ObserverStateTag
    const active = yield* Ref.get(state)
    const run = active.get(forkId)
    if (run?.fiber) {
      yield* Fiber.interrupt(run.fiber)
    }
    yield* Ref.update(state, (m) => {
      const next = new Map(m)
      next.delete(forkId)
      return next
    })
  })
}

// =============================================================================
// Single observer evaluation (no scheduling logic)
// =============================================================================

type ObserverOutcomeEvent = Extract<AppEvent, { type: 'observer_outcome' }>

interface ObserverEvaluationResult {
  readonly outcome: ObserverOutcomeEvent
  readonly windowState: ForkWindowState
}

function observeOnce(
  event: ObserverTurnOutcomeEvent,
  read: WorkerReadFn<AppEvent>,
  observerTurnId: string,
) {
  return Effect.gen(function* () {
    // Read window state for the observed fork — includes prior observer_turn entries
    const windowState = yield* read(WindowProjection, event.forkId)
    if (!windowState) {
      logger.warn({ forkId: event.forkId }, '[Observer] No window state for fork')
      return null
    }

    // Resolve observer model
    const modelResolver = yield* AgentModelResolver
    const observerModel = yield* modelResolver.resolveSecondary()

    // Build prompt — single system prompt for all observers
    const systemPrompt = getObserverSystemPrompt()
    const sessionContextState = yield* read(SessionContextProjection)
    const prompt = observerWindowToPrompt({
      windowState,
      systemPrompt,
      observedForkId: event.forkId,
      timezone: sessionContextState.context?.timezone ?? null,
    })

    // Create harness with single shared toolkit
    const harness = createHarness({
      model: observerModel.model,
      toolkit: observerToolkit,
    })

    // Run turn — force tool-only response
    const liveTurn = yield* harness.runTurn(prompt, {
      toolChoice: 'required',
    }).pipe(
      Effect.provideService(AgentModelOperationContextTag, {
        operationKind: 'observer',
        operationId: observerTurnId,
        relatedTurnId: event.turnId,
        chainId: event.chainId,
        forkId: event.forkId,
      }),
      Effect.retry({
        schedule: connectionRetrySchedule,
        while: (err: AgentStreamStartFailure) => modelAttemptRetryability(err)._tag === 'UpstreamRetryable',
      }),
      Effect.catchAll((err: AgentStreamStartFailure) =>
        Effect.gen(function* () {
          logger.error({ err, forkId: event.forkId }, '[Observer] Connection error after retries')
          return null
        }),
      ),
    )

    if (liveTurn === null) return null

    const reportState = yield* Stream.runFold(
      liveTurn.events,
      { _tag: 'AwaitingReport', reasoning: '' } as ObserverReportState,
      reduceObserverReportState,
    )

    const reportStatus = finalizeObserverReportState(reportState)

    switch (reportStatus._tag) {
      case 'Failure':
        logger.error(
          {
            forkId: event.forkId,
            observedTurnId: event.turnId,
            observerTurnId,
            failure: observerFailureLogData(reportStatus.failure),
          },
          '[Observer] tool call failed',
        )
        return null
      case 'ProtocolViolation':
        logger.error(
          {
            forkId: event.forkId,
            observedTurnId: event.turnId,
            observerTurnId,
            ...observerProtocolViolationLogData(reportStatus.violation),
          },
          '[Observer] invalid observer turn',
        )
        return null
      case 'Success':
        break
    }

    const escalate = reportStatus.toolName === 'escalate'
    const justification = reportStatus.justification
    const reasoning = reportStatus.reasoning.trim()

    const observerOutcome: ObserverOutcomeEvent = {
      type: 'observer_outcome',
      forkId: event.forkId,
      observedTurnId: event.turnId,
      observerTurnId,
      chainId: event.chainId,
      escalate,
      justification,
      reasoning,
    }

    return { outcome: observerOutcome, windowState }
  }).pipe(
    Effect.catchAllCause((cause) =>
      Effect.sync(() => {
        logger.error(
          { cause: Cause.pretty(cause), forkId: event.forkId, observedTurnId: event.turnId, observerTurnId },
          '[Observer] Observer evaluation failed',
        )
        return null
      }),
    ),
  )
}

function clearCurrentPending(
  entry: ObserverForkState,
  runId: string,
  event: ObserverTurnOutcomeEvent,
): ObserverForkState {
  if (entry.observer._tag !== 'running' || entry.observer.runId !== runId) return entry
  return {
    ...entry,
    observer: ObserverRunLifecycle.hold(entry.observer, {
      pendingEvent: entry.observer.pendingEvent === event ? null : entry.observer.pendingEvent,
    }),
  }
}

function markObserverRunning(
  entry: ObserverForkState,
  runId: string,
  pendingEvent: ObserverTurnOutcomeEvent | null,
): ObserverForkState {
  switch (entry.observer._tag) {
    case 'idle':
      return {
        ...entry,
        observer: ObserverRunLifecycle.transition(entry.observer, 'running', { runId, pendingEvent }),
        fiber: null,
      }
    case 'running':
      return {
        ...entry,
        observer: ObserverRunLifecycle.hold(entry.observer, { runId, pendingEvent }),
        fiber: null,
      }
  }
}

function markObserverIdleIfCurrent(entry: ObserverForkState, runId: string): ObserverForkState {
  if (entry.observer._tag !== 'running' || entry.observer.runId !== runId) return entry
  return {
    ...entry,
    observer: ObserverRunLifecycle.transition(entry.observer, 'idle', { pendingEvent: entry.observer.pendingEvent }),
    fiber: null,
  }
}

function publishObserverResult(
  result: ObserverEvaluationResult,
  publish: PublishFn<AppEvent>,
  _read: WorkerReadFn<AppEvent>,
) {
  return Effect.gen(function* () {
    const { outcome } = result
    yield* publish(outcome)
  })
}

// =============================================================================
// Per-fork observer loop
// =============================================================================

/**
 * Runs a single observer evaluation, then checks if a new turn_outcome
 * arrived while it was running. If so, re-evaluates with the latest
 * pending event. Continues until no pending event remains.
 */
function observerLoop(
  runId: string,
  initialEvent: ObserverTurnOutcomeEvent,
  publish: PublishFn<AppEvent>,
  read: WorkerReadFn<AppEvent>,
) {
  const forkId = initialEvent.forkId

  return Effect.gen(function* () {
    const stateRef = yield* ObserverStateTag
    let currentEvent: ObserverTurnOutcomeEvent | null = initialEvent

    while (currentEvent !== null) {
      const advisorRequirementPending =
        forkId === null && hasPendingAdvisorRequirement(yield* read(TurnProjection, null))

      if (advisorRequirementPending) {
        logger.info({ forkId: forkLabel(forkId) }, '[Observer] Deferring evaluation: advisor-required escalation pending')
        yield* Ref.update(stateRef, (m) => {
          const entry = getForkEntry(m, forkId)
          return setForkEntry(m, forkId, markObserverIdleIfCurrent(entry, runId))
        })
        return
      }

      const observerTurnId = createId()

      yield* Ref.update(stateRef, (m) => {
        const entry = getForkEntry(m, forkId)
        return setForkEntry(m, forkId, clearCurrentPending(entry, runId, currentEvent!))
      })

      const result = yield* observeOnce(currentEvent, read, observerTurnId)
      if (result) {
        yield* publishObserverResult(result, publish, read)
      }

      const active = yield* Ref.get(stateRef)
      const entry = active.get(forkId)
      currentEvent =
        entry?.observer._tag === 'running' && entry.observer.runId === runId
          ? entry.observer.pendingEvent
          : null
    }

    yield* Ref.update(stateRef, (m) => {
      const entry = getForkEntry(m, forkId)
      return setForkEntry(m, forkId, markObserverIdleIfCurrent(entry, runId))
    })
  }).pipe(
    Effect.catchAllCause((cause) =>
      Effect.gen(function* () {
        logger.error({ cause: Cause.pretty(cause), forkId: forkLabel(forkId) }, '[Observer] Observer loop failed')
        const stateRef = yield* ObserverStateTag
        yield* Ref.update(stateRef, (m) => {
          const entry = getForkEntry(m, forkId)
          return setForkEntry(m, forkId, markObserverIdleIfCurrent(entry, runId))
        })
      }),
    ),
  )
}

function startObserverRun(
  command: StartObserverCommand,
  publish: PublishFn<AppEvent>,
  read: WorkerReadFn<AppEvent>,
) {
  return Effect.gen(function* () {
    const stateRef = yield* ObserverStateTag
    const runId = createId()

    yield* Ref.update(stateRef, (state) => {
      const entry = getForkEntry(state, command.forkId)
      return setForkEntry(
        state,
        command.forkId,
        markObserverRunning(entry, runId, command.pendingEvent),
      )
    })

    const fiber = yield* Effect.forkDaemon(
      observerLoop(runId, command.event, publish, read),
    )

    yield* Ref.update(stateRef, (state) => {
      const entry = state.get(command.forkId)
      if (entry?.observer._tag !== 'running' || entry.observer.runId !== runId) return state
      return setForkEntry(state, command.forkId, { ...entry, fiber })
    })
  })
}

// =============================================================================
// Worker
// =============================================================================

export const ObserverWorker = Worker.define<AppEvent>()({
  name: 'Observer',

  signalHandlers: (on) => [
    on(AgentLifecycleProjection.signals.agentKilled, (value) => cancelObserverForFork(value.forkId)),
    on(AgentLifecycleProjection.signals.subagentUserKilled, (value) => cancelObserverForFork(value.forkId)),
    on(AgentLifecycleProjection.signals.workerIdleClosed, (value) => cancelObserverForFork(value.forkId)),
  ],

  eventHandlers: {
    turn_outcome: (event, publish, read) =>
      Effect.gen(function* () {
        // TEMPORARILY DISABLED: worker observer (forkId !== null).
        // Only observe leader turns (forkId === null) for now.
        // Remove this early return to re-enable worker observation.
        if (event.forkId !== null) return

        const advisorRequirementPending = hasPendingAdvisorRequirement(yield* read(TurnProjection, null))
        const state = yield* ObserverStateTag
        const command = yield* Ref.modify(state, (m) => {
          const entry = getForkEntry(m, event.forkId)
          if (advisorRequirementPending) {
            return [noCommand, setForkEntry(m, event.forkId, entry)] as const
          }
          const result = nextTurnOutcomeCommand(event.forkId, event, entry)
          return [result.command, setForkEntry(m, event.forkId, result.entry)] as const
        })

        if (command._tag === 'start_observer') {
          yield* startObserverRun(command, publish, read)
        }
      }),
  },
})
