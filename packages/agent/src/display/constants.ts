export const HIDE_THINKING_LABELS = true

export const TRAIT_LABELS = [
  '[ATTENTIVE]', '[STRATEGIC]', '[PROACTIVE]', '[RESPECTFUL]',
  '[GROUNDED]', '[INTROSPECTIVE]', '[TASK]', '[SKIP]',
] as const

export const EMPTY_TOOL_COUNTS = {
  commands: 0,
  reads: 0,
  writes: 0,
  edits: 0,
  searches: 0,
  webSearches: 0,
  webFetches: 0,
  artifactWrites: 0,
  artifactUpdates: 0,
  other: 0,
} as const
