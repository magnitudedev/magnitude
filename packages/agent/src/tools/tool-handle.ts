import { SchemaAccumulator } from '@magnitudedev/xml-act'
import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { ToolStateEvent } from '@magnitudedev/tools'
import { normalizeToolEvent } from '../normalizer'
import { TOOL_DEFINITIONS, type ToolKey, type ToolStateFor } from './tool-definitions'

export type ToolState = { [K in ToolKey]: ToolStateFor<K> }[ToolKey]
type AnyToolEvent = ToolStateEvent<unknown, unknown, unknown, unknown>
type ToolReducer<S> = { bivarianceHack(state: S, event: AnyToolEvent): S }['bivarianceHack']

export interface ToolHandle {
  readonly toolKey: ToolKey
  readonly state: ToolState
  process(raw: ToolCallEvent): ToolHandle
  interrupt(): ToolHandle
}

export function createToolHandle(toolKey: ToolKey): ToolHandle {
  const acc = new SchemaAccumulator()
  switch (toolKey) {
    case 'shell':
      return buildHandle('shell', TOOL_DEFINITIONS.shell.model.initial, acc, TOOL_DEFINITIONS.shell.model.reduce)
    case 'fileRead':
      return buildHandle('fileRead', TOOL_DEFINITIONS.fileRead.model.initial, acc, TOOL_DEFINITIONS.fileRead.model.reduce)
    case 'fileWrite':
      return buildHandle('fileWrite', TOOL_DEFINITIONS.fileWrite.model.initial, acc, TOOL_DEFINITIONS.fileWrite.model.reduce)
    case 'fileEdit':
      return buildHandle('fileEdit', TOOL_DEFINITIONS.fileEdit.model.initial, acc, TOOL_DEFINITIONS.fileEdit.model.reduce)
    case 'fileTree':
      return buildHandle('fileTree', TOOL_DEFINITIONS.fileTree.model.initial, acc, TOOL_DEFINITIONS.fileTree.model.reduce)
    case 'fileSearch':
      return buildHandle('fileSearch', TOOL_DEFINITIONS.fileSearch.model.initial, acc, TOOL_DEFINITIONS.fileSearch.model.reduce)
    case 'webSearch':
      return buildHandle('webSearch', TOOL_DEFINITIONS.webSearch.model.initial, acc, TOOL_DEFINITIONS.webSearch.model.reduce)
    case 'webFetch':
      return buildHandle('webFetch', TOOL_DEFINITIONS.webFetch.model.initial, acc, TOOL_DEFINITIONS.webFetch.model.reduce)
    case 'agentCreate':
      return buildHandle('agentCreate', TOOL_DEFINITIONS.agentCreate.model.initial, acc, TOOL_DEFINITIONS.agentCreate.model.reduce)
    case 'agentKill':
      return buildHandle('agentKill', TOOL_DEFINITIONS.agentKill.model.initial, acc, TOOL_DEFINITIONS.agentKill.model.reduce)
    case 'skill':
      return buildHandle('skill', TOOL_DEFINITIONS.skill.model.initial, acc, TOOL_DEFINITIONS.skill.model.reduce)
    case 'click':
      return buildHandle('click', TOOL_DEFINITIONS.click.model.initial, acc, TOOL_DEFINITIONS.click.model.reduce)
    case 'doubleClick':
      return buildHandle('doubleClick', TOOL_DEFINITIONS.doubleClick.model.initial, acc, TOOL_DEFINITIONS.doubleClick.model.reduce)
    case 'rightClick':
      return buildHandle('rightClick', TOOL_DEFINITIONS.rightClick.model.initial, acc, TOOL_DEFINITIONS.rightClick.model.reduce)
    case 'type':
      return buildHandle('type', TOOL_DEFINITIONS.type.model.initial, acc, TOOL_DEFINITIONS.type.model.reduce)
    case 'scroll':
      return buildHandle('scroll', TOOL_DEFINITIONS.scroll.model.initial, acc, TOOL_DEFINITIONS.scroll.model.reduce)
    case 'drag':
      return buildHandle('drag', TOOL_DEFINITIONS.drag.model.initial, acc, TOOL_DEFINITIONS.drag.model.reduce)
    case 'navigate':
      return buildHandle('navigate', TOOL_DEFINITIONS.navigate.model.initial, acc, TOOL_DEFINITIONS.navigate.model.reduce)
    case 'goBack':
      return buildHandle('goBack', TOOL_DEFINITIONS.goBack.model.initial, acc, TOOL_DEFINITIONS.goBack.model.reduce)
    case 'switchTab':
      return buildHandle('switchTab', TOOL_DEFINITIONS.switchTab.model.initial, acc, TOOL_DEFINITIONS.switchTab.model.reduce)
    case 'newTab':
      return buildHandle('newTab', TOOL_DEFINITIONS.newTab.model.initial, acc, TOOL_DEFINITIONS.newTab.model.reduce)
    case 'screenshot':
      return buildHandle('screenshot', TOOL_DEFINITIONS.screenshot.model.initial, acc, TOOL_DEFINITIONS.screenshot.model.reduce)
    case 'evaluate':
      return buildHandle('evaluate', TOOL_DEFINITIONS.evaluate.model.initial, acc, TOOL_DEFINITIONS.evaluate.model.reduce)
    case 'phase-submit':
      return buildHandle('phase-submit', TOOL_DEFINITIONS['phase-submit'].model.initial, acc, TOOL_DEFINITIONS['phase-submit'].model.reduce)
    case 'workflow-submit':
      return buildHandle('workflow-submit', TOOL_DEFINITIONS['workflow-submit'].model.initial, acc, TOOL_DEFINITIONS['workflow-submit'].model.reduce)
    case 'phase-verdict':
      return buildHandle('phase-verdict', TOOL_DEFINITIONS['phase-verdict'].model.initial, acc, TOOL_DEFINITIONS['phase-verdict'].model.reduce)
  }
}

function buildHandle<K extends ToolKey>(
  toolKey: K,
  state: ToolStateFor<K>,
  acc: SchemaAccumulator,
  reduce: ToolReducer<ToolStateFor<K>>,
): ToolHandle {
  return {
    toolKey,
    get state() { return state },
    process(raw: ToolCallEvent): ToolHandle {
      acc.ingest(raw)
      const event = normalizeToolEvent(raw, acc)
      if (event) {
        return buildHandle(toolKey, reduce(state, event), acc, reduce)
      }
      return this
    },
    interrupt(): ToolHandle {
      return buildHandle(toolKey, reduce(state, { type: 'interrupted' }), acc, reduce)
    },
  }
}
