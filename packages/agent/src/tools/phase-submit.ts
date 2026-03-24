import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { getCurrentPhase, validateFields } from '@magnitudedev/skills'
import { WorkflowStateReaderTag } from './workflow-reader'
import { expandWorkspacePath } from '../workspace/workspace-path'
import { WorkingDirectoryTag } from '../execution/working-directory'

const { ForkContext } = Fork

const PhaseSubmitErrorSchema = ToolErrorSchema('PhaseSubmitError', {})

const FieldSchema = Schema.Struct({
  name: Schema.String,
  value: Schema.String,
})

export const phaseSubmitTool = defineTool({
  name: 'phase-submit',
  group: 'default',
  description: 'Submit deliverables for the current workflow phase.',
  inputSchema: Schema.Struct({
    fields: Schema.optional(Schema.Array(FieldSchema)),
  }),
  outputSchema: Schema.String,
  errorSchema: PhaseSubmitErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      const workerBus = yield* WorkerBusTag<AppEvent>()
      const { forkId } = yield* ForkContext
      const workflowStateReader = yield* WorkflowStateReaderTag
      const workflowState = yield* Effect.map(workflowStateReader.getState(forkId), (state) => state.workflowState)

      if (!workflowState || workflowState.status === 'completed') {
        return yield* Effect.fail({
          _tag: 'PhaseSubmitError' as const,
          message: 'No active workflow. Activate a skill with phases first.',
        })
      }

      if (!input.fields || input.fields.length === 0) {
        return yield* Effect.fail({
          _tag: 'PhaseSubmitError' as const,
          message: 'Provide fields.',
        })
      }

      const { workspacePath } = yield* WorkingDirectoryTag
      const phase = getCurrentPhase(workflowState)
      const fileFieldNames = new Set(
        (phase?.submit?.fields ?? []).filter((f) => f.type === 'file').map((f) => f.name)
      )
      const fields = new Map(input.fields.map((field) => [
        field.name,
        fileFieldNames.has(field.name) ? expandWorkspacePath(field.value, workspacePath) : field.value,
      ] as const))

      const validation = validateFields(workflowState, fields)
      if (!validation.valid) {
        const details = validation.errors.map((error) => {
          if (error.type === 'missing') return `- Missing required field: ${error.name}`
          return `- File not found for ${error.name}: ${error.path}`
        }).join('\n')

        return yield* Effect.fail({
          _tag: 'PhaseSubmitError' as const,
          message: `Submission failed:\n${details}`,
        })
      }

      yield* workerBus.publish({
        type: 'phase_submitted',
        forkId,
        fields,
      })

      return 'Phase submitted.'
    }),
  label: () => 'Submitting workflow phase',
})

export const phaseSubmitXmlBinding = defineXmlBinding(phaseSubmitTool, {
  input: {
    children: [{
      field: 'fields',
      tag: 'field',
      attributes: [
        { field: 'name', attr: 'name' },
        { field: 'value', attr: 'value' },
      ],
    }],
  },
  output: {},
} as const)
