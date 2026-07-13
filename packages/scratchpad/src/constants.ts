/**
 * Scratchpad subdirectories that are pre-created for each session.
 * These correspond to directories referenced in agent/worker prompts.
 */
export const SCRATCHPAD_SUBDIRS = [
  'reports',
  'designs',
  'plans',
  'thoughts',
  'results',
] as const

export type ScratchpadSubdir = (typeof SCRATCHPAD_SUBDIRS)[number]
