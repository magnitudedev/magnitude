/**
 * Explorer Agent Definition
 *
 * Read-only agent that answers informational questions by exploring the codebase.
 * Uses secondary model. Communicates back via parent.message.
 */

import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
import { artifactReadTool, artifactWriteTool } from '../tools/artifact-tools'
import { classifyShellCommand } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding what to investigate next',
  description: "How can you gather the needed information quickly and efficiently? What tools and techniques will get you there fastest — tree, search, read, shell, web? Prioritize high-signal sources. Don't read aimlessly.",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what to read, search, or explore this turn. Maximize coverage per turn by reading multiple files in parallel.',
})

const tools = toolSet({
  fileRead:      readTool,
  fileTree:      treeTool,
  fileSearch:    searchTool,
  fileView:      viewTool,
  shell:         shellTool,
  webSearch:     webSearchTool,
  webFetch:      webFetchTool,
  artifactRead:  artifactReadTool,
  artifactWrite: artifactWriteTool,

  think:         thinkTool,
})

export const createExplorer = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'explorer',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [strategyLens, turnLens],
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'readonly') return p.allow()
      // Explorers are read-only — reject anything above readonly
      return p.reject('Explorers can only run read-only shell commands (ls, cat, git log, etc).')
    },
    _default() { return p.allow() },
  }),

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      return continue_()
    },
  },


  display: (d) => ({
    think() { return d.hidden() },
    _default() { return d.visible() },
  }),
})