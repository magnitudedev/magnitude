import { describe, expect, it } from 'vitest'
import type { ConfigState, SlotConfig } from '../src/ambient/config-ambient'
import { ConfigAmbient } from '../src/ambient/config-ambient'
import { materializeAgentToolkit, selectAgentToolKeys, toolUniverseToolkit } from '../src/tools/toolkits'
import { ReasoningEffortSchema } from '@magnitudedev/ai'
import { AmbientServiceTag, EventEngine } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import type { AppEvent } from '../src/events'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { AgentToolkitProjection } from '../src/projections/agent-toolkit'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const ToolkitProjectionAgent = EventEngine.make<AppEvent>()({
  name: 'DynamicImageToolkitProjectionAgent',
  schemaVersion: 'test',
  projections: [AgentLifecycleProjection, AgentToolkitProjection],
  workers: [],
})

function slot(slotId: 'primary' | 'secondary', vision: boolean | undefined): SlotConfig {
  return {
    slotId, providerId: 'test', providerModelId: slotId,
    profile: { contextWindow: 100_000, maxOutputTokens: 4_000 },
    vision, hardCap: 96_000, softCap: 80_000, reasoningEffort: ReasoningEffortSchema.make('medium'),
    isUserOverride: false, isFallback: false,
  }
}

function config(primary: boolean | undefined, secondary: boolean | undefined): ConfigState {
  return {
    revision: 1,
    catalogLoaded: true,
    bySlot: {
      primary: { _tag: 'Ready', config: slot('primary', primary) },
      secondary: { _tag: 'Ready', config: slot('secondary', secondary) },
    },
  }
}

function imageTools(state: ConfigState, role: 'leader' | 'advisor' = 'leader'): string[] {
  const keys = selectAgentToolKeys({ roleId: role, configState: state, solo: false, vcsAvailable: false })
  return keys.filter(key => key === 'fileView' || key === 'queryImage')
}

describe('dynamic image tooling', () => {
  it('uses view when the active slot has vision', () => {
    expect(imageTools(config(true, false))).toEqual(['fileView'])
  })

  it('uses query_image when only the opposite slot has vision', () => {
    expect(imageTools(config(false, true))).toEqual(['queryImage'])
  })

  it('exposes no image tool when neither capability is available', () => {
    expect(imageTools(config(false, false))).toEqual([])
    expect(imageTools(config(undefined, undefined))).toEqual([])
  })

  it('treats an unavailable opposite slot as non-vision', () => {
    const state: ConfigState = {
      ...config(false, true),
      revision: 2,
      bySlot: {
        primary: { _tag: 'Ready', config: slot('primary', false) },
        secondary: { _tag: 'Unavailable', slotId: 'secondary', reason: 'provider_unavailable' },
      },
    }
    expect(imageTools(state)).toEqual([])
  })

  it('evaluates secondary roles symmetrically', () => {
    const keys = selectAgentToolKeys({
      roleId: 'engineer',
      configState: config(true, false),
      solo: false,
      vcsAvailable: false,
    })
    expect(keys.filter(key => key === 'fileView' || key === 'queryImage')).toEqual(['queryImage'])
  })

  it('does not churn materialized tools when only the config revision changes', () => {
    const firstConfig = config(true, false)
    const secondConfig = { ...config(true, true), revision: 2 }
    const firstKeys = selectAgentToolKeys({ roleId: 'leader', configState: firstConfig, solo: false, vcsAvailable: false })
    const secondKeys = selectAgentToolKeys({ roleId: 'leader', configState: secondConfig, solo: false, vcsAvailable: false })

    expect(secondKeys).toEqual(firstKeys)
    expect(materializeAgentToolkit(toolUniverseToolkit, secondKeys))
      .toBe(materializeAgentToolkit(toolUniverseToolkit, firstKeys))
  })

  it('reacts to config ambient changes at the fork projection boundary', async () => {
    const client = await ToolkitProjectionAgent.createClient(ToolUniverseSourceLive)
    try {
      await client.runEffect(Effect.gen(function* () {
        const ambient = yield* AmbientServiceTag
        yield* ambient.update(ConfigAmbient, config(false, true))
      }))
      await client.send({
        type: 'session_initialized',
        forkId: null,
        context: {
          cwd: '/workspace',
          scratchpadPath: '/scratchpad',
          platform: 'linux',
          shell: 'zsh',
          timezone: 'UTC',
          username: 'test',
          fullName: null,
          git: null,
          folderStructure: '',
          agentsFile: null,
          skills: null,
        },
      })

      const before = await client.runEffect(Effect.gen(function* () {
        const projection = yield* AgentToolkitProjection.Tag
        return yield* projection.getFork(null)
      }))
      expect(before.configRevision).toBe(1)
      expect(before.toolKeys).toContain('queryImage')
      expect(before.toolKeys).not.toContain('fileView')

      await client.runEffect(Effect.gen(function* () {
        const ambient = yield* AmbientServiceTag
        yield* ambient.update(ConfigAmbient, { ...config(true, true), revision: 2 })
      }))

      const after = await client.runEffect(Effect.gen(function* () {
        const projection = yield* AgentToolkitProjection.Tag
        return yield* projection.getFork(null)
      }))
      expect(after.configRevision).toBe(2)
      expect(after.toolKeys).toContain('fileView')
      expect(after.toolKeys).not.toContain('queryImage')
    } finally {
      await client.dispose()
    }
  })
})
