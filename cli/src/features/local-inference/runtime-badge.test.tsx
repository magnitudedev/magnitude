import { renderToStaticMarkup } from 'react-dom/server'
import { Option } from 'effect'
import { describe, expect, test, vi } from 'vitest'
import {
  ModelSlotLoadingLocalModel,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  PRIMARY_SLOT_ID,
} from '@magnitudedev/sdk'
import {
  GIB,
  LOCAL_PROVIDER_ID,
  makeHardware,
  makeView,
  TEST_MEMORY_DOMAIN_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
} from './test-fixtures'
import {
  deriveInferenceRuntimeBadgeView,
  inferenceRuntimeBadgeLabel,
  inferenceRuntimeTextColors,
  InferenceRuntimeBadge,
  InferenceRuntimeBadgeOverlay,
} from './runtime-badge'

vi.mock('@opentui/react', () => ({
  useRenderer: () => ({ setMousePointer: () => {} }),
}))

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: '#ffffff',
    muted: '#888888',
    border: '#555555',
    surface: '#111111',
    success: '#00aa00',
    error: '#ff0000',
    terminalDetectedBg: '#000000',
  }),
}))

const selection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId: TEST_MODEL_ID,
  reasoningEffort: TEST_REASONING_EFFORT,
}

const withPrimary = (
  primary: ReturnType<typeof makeView>['slots']['slots']['primary'],
) => {
  const state = makeView()
  return {
    ...state,
    slots: { ...state.slots, slots: { ...state.slots.slots, primary } },
  }
}

describe('inference runtime badge view', () => {
  test('reports the complete server-published runtime allocation while resident', () => {
    const state = makeView({
      hardware: makeHardware({
        residentMemory: Option.some({
          domains: [{
            memoryDomainId: TEST_MEMORY_DOMAIN_ID,
            modelBytes: 13 * GIB,
            contextBytes: 2 * GIB,
            computeBytes: GIB,
            auxiliaryBytes: 0,
          }],
        }),
      }),
    })

    expect(deriveInferenceRuntimeBadgeView(state)).toEqual({
      status: 'running',
      memoryLabel: '16 GB',
    })
  })

  test('derives transitions globally from either local slot and hides stale memory', () => {
    const loading = withPrimary(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
      percentage: 42,
    }))
    loading.hardware = makeHardware({
      residentMemory: Option.some({
        domains: [{
          memoryDomainId: TEST_MEMORY_DOMAIN_ID,
          modelBytes: 16 * GIB,
          contextBytes: 0,
          computeBytes: 0,
          auxiliaryBytes: 0,
        }],
      }),
    })
    expect(deriveInferenceRuntimeBadgeView(loading)).toEqual({ status: 'starting', memoryLabel: null })

    const unloading = withPrimary(new ModelSlotUnloadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))
    expect(deriveInferenceRuntimeBadgeView(unloading)).toEqual({ status: 'stopping', memoryLabel: null })
  })

  test('distinguishes idle, stopped, checking, and ready-state mirror convergence', () => {
    expect(deriveInferenceRuntimeBadgeView(makeView({ ready: false })))
      .toEqual({ status: 'idle', memoryLabel: null })
    expect(deriveInferenceRuntimeBadgeView(null))
      .toEqual({ status: 'checking', memoryLabel: null })
    expect(deriveInferenceRuntimeBadgeView(makeView()))
      .toEqual({ status: 'starting', memoryLabel: null })

    const stopped = withPrimary(new ModelSlotUnloadedLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))
    stopped.hardware = makeHardware({
      runtimeFailure: Option.some({
        modelId: 'configuration_test',
        code: 'worker_exited',
        message: 'Inference worker stopped.',
        retryable: true,
      }),
    })
    expect(deriveInferenceRuntimeBadgeView(stopped))
      .toEqual({ status: 'stopped', memoryLabel: null })
  })

  test('provides full and compact one-line labels', () => {
    const running = { status: 'running' as const, memoryLabel: '16 GB' }
    expect(inferenceRuntimeBadgeLabel(running, false)).toBe('Local inference  ·  16 GB')
    expect(inferenceRuntimeBadgeLabel(running, true)).toBe('Local  ·  16 GB')
    expect(inferenceRuntimeBadgeLabel({ status: 'starting', memoryLabel: null }, false))
      .toBe('Local inference  ·  Starting…')
    expect(inferenceRuntimeBadgeLabel({ status: 'idle', memoryLabel: null }, false))
      .toBe('Local inference  ·  Idle')
  })

  test('brightens only text on hover while preserving the slate hierarchy', () => {
    expect(inferenceRuntimeTextColors(false)).toEqual({
      label: '#94a3b8',
      separator: '#64748b',
      value: '#cbd5e1',
    })
    expect(inferenceRuntimeTextColors(true)).toEqual({
      label: '#cbd5e1',
      separator: '#94a3b8',
      value: '#e2e8f0',
    })
  })

  test('renders a fully bordered badge using the detected terminal background', () => {
    const html = renderToStaticMarkup(
      <InferenceRuntimeBadge
        view={{ status: 'running', memoryLabel: '16 GB' }}
        compact={false}
        onOpenHardware={() => {}}
      />,
    )

    expect(html).toContain('border-style:single')
    expect(html).toContain('border-color:')
    expect(html).toContain('background-color:')
    expect(html).not.toContain('background-color:#111111')
    expect(html).toMatch(/span style="fg:[^"]+">●/)
    expect(html).toContain('Local inference</span><span style="fg:')
    expect(html).toContain('fg:#94a3b8"> Local inference')
    expect(html).toContain('fg:#64748b">  ·  </span>')
    expect(html).toContain('fg:#cbd5e1">16 GB</span>')
  })

  test('overlays the badge without participating in shell layout', () => {
    const html = renderToStaticMarkup(
      <InferenceRuntimeBadgeOverlay
        view={{ status: 'idle', memoryLabel: null }}
        compact={false}
        onOpenHardware={() => {}}
      />,
    )

    expect(html).toContain('position:absolute')
    expect(html).toContain('top:0')
    expect(html).toContain('right:2px')
  })
})
