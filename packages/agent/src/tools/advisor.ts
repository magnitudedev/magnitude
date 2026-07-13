/**
 * message_advisor tool
 *
 * Lets the leader ask the advisor role for strategic feedback over a filtered
 * WindowProjection view. The advisor receives no tools.
 */

import { Effect, Schema } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import {
  AdvisorErrorSchema,
  executeMessageAdvisor,
} from '../advisor/orchestrate'

export {
  collectAdvisorText,
  AdvisorErrorSchema,
  advisorError,
  streamStartFailureMessage,
  streamErrorMessage,
  callerLabel,
  executeMessageAdvisor,
} from '../advisor/orchestrate'

export const messageAdvisorTool = defineHarnessTool({
  definition: {
    name: 'message_advisor',
    description: 'Ask the advisor role for strategic feedback. The advisor sees the filtered conversation context plus your message. It has no tools and no project access. The advisor remembers previous messages you have sent it — you do not need to restate context when calling this tool again. You may converse with the advisor similarly to how you would converse with the user.',
    inputSchema: Schema.Struct({
      message: Schema.String.annotations({
        description: 'The question or context to send to the advisor. Include what you want advice on and any specific risks or decision points.',
      }),
    }),
    outputSchema: Schema.String,
  },
  errorSchema: AdvisorErrorSchema,
  execute: executeMessageAdvisor,
})
