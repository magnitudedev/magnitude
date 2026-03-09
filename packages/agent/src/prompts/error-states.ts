export const UNCLOSED_THINK_REMINDER = 'Your response had an unclosed thinking block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const UNCLOSED_ACTIONS_REMINDER = 'Your response had an unclosed actions block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const UNCLOSED_INSPECT_REMINDER = 'Your response had an unclosed inspect block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'


export function formatNonexistentAgentError(destList: string): string {
  return `Message sent to nonexistent agent ID(s): ${destList}. The message was not delivered. Check the agent ID and ensure the agent has been created and is still active.`
}
