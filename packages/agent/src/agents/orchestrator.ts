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
import { backgroundProcessesObservable } from '../observables/background-processes-observable'
import { thinkTool, skillTool, phaseSubmitTool } from '../tools/globals'
import { agentCreateTool, agentKillTool } from '../tools/agent-tools'
// import { gatherTool } from '../tools/gather'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webFetchTool } from '../tools/web-fetch-tool'
import { webSearchTool } from '../tools/web-search-tool'

import { classifyShellCommand, writesStayWithin } from '@magnitudedev/shell-classifier'

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
  description: 'Consider how to best tackle work - subagents, parallelism, sequencing, workspace usage.',
})

const protocolLens = defineThinkingLens({
  name: 'protocol',
  trigger: "When any relevant protocol applies",
  description: 'Adhere to all protocols',
})

const practicesLens = defineThinkingLens({
  name: 'practices',
  trigger: "When any default practices apply",
  description: 'Consider which default practices apply in this situation',
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When your turn involves communications and actions that could benefit from planning',
  description: 'Plan what to communicate, what actions to take, and which turn control to use. If acting this turn, remember that you cannot communicate the results of those actions until next turn.',
})

// One-shot specific lenses
const constraintsLens = defineThinkingLens({
  name: 'constraints',
  trigger: 'When planning work, delegating to subagents, or evaluating progress',
  description:
    'What are the exact requirements? Have I extracted all testable constraints? Which have I verified? Which remain? Am I missing any implicit requirements?',
})

const pivotLens = defineThinkingLens({
  name: 'pivot',
  trigger: 'When an approach is not making progress or results are unexpected',
  description:
    'Is my current approach working? Are my subagents stuck or spinning? Should I try a different strategy, parallelize an alternative, or cut losses on this path? What signals indicate I should change direction?',
})

const validationLens = defineThinkingLens({
  name: 'validation',
  trigger: 'When evaluating whether work is complete or results are acceptable',
  description:
    'Have I empirically tested my complete solution, not just individual pieces? Are there edge cases or details I haven\'t checked? Am I accepting results that look wrong or suspicious?',
})

export const oneshotThinkingLenses = [constraintsLens, pivotLens, strategyLens, validationLens, turnLens]

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
    fileView:              viewTool,
    shell:                 shellTool,
    shellBg:               shellBgTool,
    webSearch:             webSearchTool,
    webFetch:              webFetchTool,

    // Agent management
    agentCreate:           agentCreateTool,
    agentKill:             agentKillTool,

    // Skills & workflows
    skill:                 skillTool,
    phaseSubmit:           phaseSubmitTool,
  })

  return defineAgent<typeof tools, PolicyContext>(tools, {
    id: 'orchestrator',
    model: 'primary',
    systemPrompt,
    thinkingLenses: [intentLens, ideateLens, strategyLens, protocolLens, practicesLens, turnLens],
    observables: [agentsStatusObservable, backgroundProcessesObservable],
    permission: (p) => ({
      shell(input, pctx) {
        const result = classifyShellCommand(input.command)
        const allowedPrefixes = pctx.workspacePath ? [pctx.workspacePath] : undefined
        if (!pctx.disableShellSafeguards && result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden and cannot be executed.')
        if (!pctx.disableCwdSafeguards && !writesStayWithin(input.command, pctx.cwd, ...(allowedPrefixes ?? []))) return p.reject('This command targets paths outside the working directory.')
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
        const yielders = ['agentCreate', 'agentKill']
        if (turnCtx.lastTool && yielders.includes(turnCtx.lastTool)) return yield_()
        if (turnCtx.messagesSent.some(m => m.dest !== 'user')) return yield_()
        return continue_()
      },
    },



    display: (d) => ({
      think()              { return d.hidden() },
      inspect()            { return d.hidden() },
      agentCreate()        { return d.hidden() },
      agentKill()          { return d.hidden() },

      // gather()             { return d.visible() },
      shell()              { return d.visible() },
      _default()           { return d.visible() },
    }),
  })
}
