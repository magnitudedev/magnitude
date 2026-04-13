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
import { skillDisplay } from './displays/skill'
import { browserActionDisplay } from './displays/browser-action'
import { defaultDisplay } from './displays/default'

type RenderableToolState = ToolState

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
    default:
      return defaultDisplay.summary(state)
  }
}
