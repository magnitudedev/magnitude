/**
 * Observer toolkit schemas.
 *
 * Two tools: pass (no args) and escalate (justification string).
 * The observer picks one after reasoning. toolChoice: 'required' forces a call.
 */

import { Effect, Schema } from 'effect'
import { defineHarnessTool, defineToolkit } from '@magnitudedev/harness'
import { NoInputSchema } from '@magnitudedev/utils/schema'
import type { ObserverJustification } from './justifications'

// =============================================================================
// Justification schema
// =============================================================================

const JUSTIFICATION_VALUES = [
  'difficulty',
  'churn',
  'frustration',
] as const

export const JustificationSchema = Schema.Literal(...JUSTIFICATION_VALUES)

// =============================================================================
// Tools
// =============================================================================

export const passTool = defineHarnessTool({
  definition: {
    name: 'pass',
    description: 'Call this when the observed agent is operating acceptably. No escalation needed.',
    inputSchema: NoInputSchema,
    outputSchema: Schema.Struct({ status: Schema.Literal('ok') }),
  },
  execute: () => Effect.succeed({ status: 'ok' as const }),
})

export const escalateTool = defineHarnessTool({
  definition: {
    name: 'escalate',
    description: 'Call this when the observed agent has a problem that warrants intervention. Provide a justification describing what you see.',
    inputSchema: Schema.Struct({
      justification: JustificationSchema.annotations({
        description: 'Why escalation is warranted. Pick the single best match.',
      }),
    }),
    outputSchema: Schema.Struct({ status: Schema.Literal('ok') }),
  },
  execute: () => Effect.succeed({ status: 'ok' as const }),
})

// =============================================================================
// Toolkit (single shared toolkit for both worker and leader)
// =============================================================================

export const observerToolkit = defineToolkit({
  pass: { tool: passTool },
  escalate: { tool: escalateTool },
})

export type PassInput = Schema.Schema.Type<typeof passTool.definition.inputSchema>
export type EscalateInput = Schema.Schema.Type<typeof escalateTool.definition.inputSchema>
