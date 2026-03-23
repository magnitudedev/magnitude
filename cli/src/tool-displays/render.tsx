import type { ToolState } from '@magnitudedev/agent'
import type { CommonToolProps } from './types'
import { shellDisplay } from './displays/shell'
import { diffDisplay } from './displays/diff'
import { contentDisplay } from './displays/content'
import { fileReadDisplay } from './displays/file-read'
import { fileSearchDisplay } from './displays/file-search'
import { fileTreeDisplay } from './displays/file-tree'
import { webSearchDisplay } from './displays/web-search'
import { webFetchDisplay } from './displays/web-fetch'
import { agentCreateDisplay } from './displays/agent-create'
import { agentKillDisplay } from './displays/agent-kill'
import { skillDisplay } from './displays/skill'
import { browserActionDisplay } from './displays/browser-action'
import { phaseSubmitDisplay } from './displays/phase-submit'
import { phaseVerdictDisplay } from './displays/phase-verdict'
import { defaultDisplay } from './displays/default'

type WorkflowSubmitState = {
  toolKey: 'workflow-submit'
  phase?: string
  output?: string
  errorMessage?: string
}

type RenderableToolState = ToolState | WorkflowSubmitState

export function renderToolStep(state: RenderableToolState, common: CommonToolProps) {
  switch (state.toolKey) {
    case 'shell': return shellDisplay.render({ state, ...common })
    case 'fileRead': return fileReadDisplay.render({ state, ...common })
    case 'fileWrite': return contentDisplay.render({ state, ...common })
    case 'fileEdit': return diffDisplay.render({ state, ...common })
    case 'fileTree': return fileTreeDisplay.render({ state, ...common })
    case 'fileSearch': return fileSearchDisplay.render({ state, ...common })
    case 'webSearch': return webSearchDisplay.render({ state, ...common })
    case 'webFetch': return webFetchDisplay.render({ state, ...common })
    case 'agentCreate': return agentCreateDisplay.render({ state, ...common })
    case 'agentKill': return agentKillDisplay.render({ state, ...common })
    case 'skill': return skillDisplay.render({ state, ...common })
    case 'click':
    case 'doubleClick':
    case 'rightClick':
    case 'type':
    case 'scroll':
    case 'drag':
    case 'navigate':
    case 'goBack':
    case 'switchTab':
    case 'newTab':
    case 'screenshot':
    case 'evaluate':
      return browserActionDisplay.render({ state, ...common })
    case 'phase-submit':
    case 'workflow-submit':
      return phaseSubmitDisplay.render({ state, ...common })
    case 'phase-verdict':
      return phaseVerdictDisplay.render({ state, ...common })
    default:
      return defaultDisplay.render({ state, ...common })
  }
}

export function summarizeToolStep(state: RenderableToolState): string {
  switch (state.toolKey) {
    case 'shell': return shellDisplay.summary(state)
    case 'fileRead': return fileReadDisplay.summary(state)
    case 'fileWrite': return contentDisplay.summary(state)
    case 'fileEdit': return diffDisplay.summary(state)
    case 'fileTree': return fileTreeDisplay.summary(state)
    case 'fileSearch': return fileSearchDisplay.summary(state)
    case 'webSearch': return webSearchDisplay.summary(state)
    case 'webFetch': return webFetchDisplay.summary(state)
    case 'agentCreate': return agentCreateDisplay.summary(state)
    case 'agentKill': return agentKillDisplay.summary(state)
    case 'skill': return skillDisplay.summary(state)
    case 'click':
    case 'doubleClick':
    case 'rightClick':
    case 'type':
    case 'scroll':
    case 'drag':
    case 'navigate':
    case 'goBack':
    case 'switchTab':
    case 'newTab':
    case 'screenshot':
    case 'evaluate':
      return browserActionDisplay.summary(state)
    case 'phase-submit':
    case 'workflow-submit':
      return phaseSubmitDisplay.summary(state)
    case 'phase-verdict':
      return phaseVerdictDisplay.summary(state)
    default:
      return defaultDisplay.summary(state)
  }
}
