import { TURN_CONTROL_FINISH } from '@magnitudedev/xml-act'

export const UNCLOSED_THINK_REMINDER = 'Your response had an unclosed thinking block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const UNCLOSED_ACTIONS_REMINDER = 'Your response had an unclosed actions block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const ONESHOT_LIVENESS_REMINDER = `You yielded but no subagents are active and there is no user to respond. Continue working toward the task or call <${TURN_CONTROL_FINISH}/> when complete.`

export function formatNonexistentAgentError(destList: string): string {
  return `Message sent to nonexistent agent ID(s): ${destList}. The message was not delivered. Check the agent ID and ensure the agent has been created and is still active.`
}
