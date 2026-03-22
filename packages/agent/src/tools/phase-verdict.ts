import { Context, Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

const { ForkContext } = Fork

const PhaseVerdictErrorSchema = ToolErrorSchema('PhaseVerdictError', {})

export interface PhaseVerdictContext {
  readonly parentForkId: string | null
  readonly criteriaIndex: number
  readonly criteriaName: string
}

export class PhaseVerdictContextTag extends Context.Tag('PhaseVerdictContext')<PhaseVerdictContextTag, PhaseVerdictContext>() {}

export const phaseVerdictTool = defineTool({
  name: 'phase-verdict',
  group: 'default',
  description: 'Report your review verdict with reasoning.',
  inputSchema: Schema.Struct({
    reason: Schema.String,
    passed: Schema.Boolean,
  }),
  outputSchema: Schema.String,
  errorSchema: PhaseVerdictErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      const workerBus = yield* WorkerBusTag<AppEvent>()
      const { forkId } = yield* ForkContext
      const verdictContext = yield* PhaseVerdictContextTag

      const event: AppEvent = {
        type: 'phase_criteria_verdict',
        forkId,
        parentForkId: verdictContext.parentForkId,
        criteriaIndex: verdictContext.criteriaIndex,
        criteriaName: verdictContext.criteriaName,
        criteriaType: 'agent',
        status: input.passed ? 'passed' : 'failed',
        agentId: forkId ?? 'reviewer',
        reason: input.reason,
      }

      yield* workerBus.publish(event)
      return 'verdict submitted'
    }),
  label: () => 'Submitting verdict',
})

export const phaseVerdictXmlBinding = defineXmlBinding(phaseVerdictTool, {
  input: {
    attributes: [
      { field: 'reason', attr: 'reason' },
      { field: 'passed', attr: 'passed' },
    ],
  },
  output: {},
} as const)
