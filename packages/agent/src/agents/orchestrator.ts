/**
 * Orchestrator Agent Definition
 *
 * The user-facing brain. Manages proposals, dispatches agents,
 * and communicates with user. Does not directly edit code — delegates
 * to sub-agents via the agent tools.
 */

import { toolSet, defineAgent, continue_, yield_, defineThinkingLens } from '@magnitudedev/agent-definition'
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

import { classifyShellCommand, detectsOutsideCwd } from '@magnitudedev/shell-classifier'

const intentLens = defineThinkingLens({
  name: 'intent',
  trigger: 'When you receive a message from the user',
  description: 'Carefully consider what the user means and what they actually want. Look past the literal request to understand the underlying goal.',
})

const ideateLens = defineThinkingLens({
  name: 'ideate',
  trigger: 'When the problem requires creative thinking or there are multiple possible approaches',
  description: 'Think freely about the problem space. Generate and consider different approaches, ideas, or solutions before committing to one. Explore tradeoffs and implications.',
})

const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding how to execute work',
  description: 'Plan your execution approach. Consider parallelism, subagent delegation, and long-horizon sequencing. Which agents to create, reuse, or dismiss? What can run in parallel? What depends on what?',
})

const protocolLens = defineThinkingLens({
  name: 'protocol',
  trigger: "Before initiating any observable changes or making decisions the user hasn't explicitly specified",
  description: 'Check your interaction protocol. Do you have approval to act? Are you making assumptions that should be communicated to the user first?',
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When your turn involves communications and actions that could benefit from planning',
  description: 'Plan what to communicate, what actions to take, and which turn control to use. If acting this turn, remember that you cannot communicate the results of those actions until next turn.',
})

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
    thinkingLenses: [intentLens, ideateLens, strategyLens, protocolLens, turnLens],
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
