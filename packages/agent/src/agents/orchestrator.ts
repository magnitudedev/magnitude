/**
 * Orchestrator Agent Definition
 *
 * The user-facing brain. Manages proposals, dispatches agents,
 * and communicates with user. Does not directly edit code — delegates
 * to sub-agents via the agent tools.
 */

import { toolSet, defineAgent, continue_, yield_, approvalThinkingLens, assumptionsThinkingLens, intentThinkingLens, taskThinkingLens, turnThinkingLens } from '@magnitudedev/agent-definition'
import type { PolicyContext } from './types'
import { agentsStatusObservable } from '../observables/agents-status-observable'
import { thinkTool } from '../tools/globals'
import {
  agentCreateTool,
  agentPauseTool,
  agentDismissTool,
} from '../tools/agent-tools'
import {
  artifactSyncTool,
  artifactReadTool,
  artifactWriteTool,
  artifactUpdateTool,
} from '../tools/artifact-tools'
// import { gatherTool } from '../tools/gather'
import { readTool, writeTool, editTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { webFetchTool } from '../tools/web-fetch-tool'
import { webSearchTool } from '../tools/web-search-tool'

import { classifyShellCommand, detectsOutsideCwd } from '@magnitude/shell-classifier'

export const createOrchestrator = (systemPrompt: string) => {
  const tools = toolSet({
    think:                 thinkTool,

    // Codebase context & edits
    // gather:                gatherTool,
    fileRead:              readTool,
    fileWrite:             writeTool,
    fileEdit:              editTool,
    fileTree:              treeTool,
    fileSearch:            searchTool,
    shell:                 shellTool,
    webSearch:             webSearchTool,
    webFetch:              webFetchTool,

    // Artifact management
    artifactSync:          artifactSyncTool,
    artifactRead:          artifactReadTool,
    artifactWrite:         artifactWriteTool,
    artifactUpdate:        artifactUpdateTool,

    // Agent management
    agentCreate:           agentCreateTool,
    agentPause:            agentPauseTool,
    agentDismiss:          agentDismissTool,
  })

  return defineAgent<typeof tools, PolicyContext>(tools, {
    id: 'orchestrator',
    model: 'primary',
    systemPrompt,
    thinkingLenses: [approvalThinkingLens, assumptionsThinkingLens, intentThinkingLens, taskThinkingLens, turnThinkingLens],
    observables: [agentsStatusObservable],
    permission: (p) => ({
      shell(input, pctx) {
        const result = classifyShellCommand(input.command)
        if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden and cannot be executed.')
        if (detectsOutsideCwd(input.command, pctx.cwd)) return p.reject('This command targets paths outside the working directory.')
        return p.allow()
      },
      _default() { return p.allow() },
    }),

    turn: {
      decide(turnCtx) {
        if (turnCtx.cancelled) return yield_()
        if (turnCtx.error) return continue_()

        // No tools called — yield (messages alone don't justify another turn)
        if (turnCtx.toolsCalled.length === 0) return yield_()

        // Yield only if the last tool in the turn was a yielder
        const yielders = ['agentCreate']
        if (turnCtx.lastTool && yielders.includes(turnCtx.lastTool)) return yield_()
        if (turnCtx.messagesSent.some(m => m.dest !== 'user')) return yield_()
        return continue_()
      },
    },



    display: (d) => ({
      think()              { return d.hidden() },
      inspect()            { return d.hidden() },
      artifactSync()       { return d.hidden() },
      artifactRead()       { return d.visible() },
      artifactWrite()      { return d.visible() },
      artifactUpdate()     { return d.visible() },
      agentCreate()        { return d.hidden() },
      agentPause()         { return d.hidden() },
      agentDismiss()       { return d.hidden() },

      // gather()             { return d.visible() },
      shell()              { return d.visible() },
      _default()           { return d.visible() },
    }),
  })
}
