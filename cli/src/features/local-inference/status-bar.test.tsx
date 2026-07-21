import { expect, test, vi } from 'vitest'
import { act } from 'react'
import { testRender } from '@opentui/react/test-utils'
import { ProviderModelIdSchema, type LocalInferenceState } from '@magnitudedev/sdk'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    primary: 'blue', secondary: 'gray', info: 'cyan', link: 'blue',
    foreground: 'white', muted: 'gray', border: 'gray', warning: 'magenta',
  }),
}))

const { LocalRuntimeStatusBar } = await import('./status-bar')
const gib = 1024 ** 3
const modelId = ProviderModelIdSchema.make('model')

const state: LocalInferenceState = {
  activeBinding: { selectionId: 'model', providerModelId: modelId, contextTokens: 32_768 },
  host: {
    _tag: 'Available',
    profile: {
      platform: 'linux', architecture: 'x86_64', topologyFingerprint: 'test',
      systemMemoryBytes: 64 * gib,
      cpuModel: 'CPU', logicalCores: 16,
      memoryDomains: [0, 1].map((index) => ({
        id: `gpu-${index}`, kind: 'physical_device' as const,
        totalCapacityBytes: 24 * gib, stableCapacityBytes: 22 * gib,
        currentFreeBytes: (6 - index * 2) * gib, sharesSystemMemory: false,
        backendNames: ['CUDA'], deviceNames: ['NVIDIA RTX 4090'], splitGroupId: null,
      })),
      residentMemory: {
        modelId: 'model', runtimeGeneration: 1,
        domains: [
          { memoryDomainId: 'gpu-0', modelBytes: 13 * gib, contextBytes: 2 * gib, computeBytes: 0.5 * gib, auxiliaryBytes: 0 },
          { memoryDomainId: 'gpu-1', modelBytes: 13 * gib, contextBytes: 4 * gib, computeBytes: 1 * gib, auxiliaryBytes: 0 },
        ],
      },
    },
  },
  choices: [{
    _tag: 'Running', choiceId: 'model', displayName: 'Qwen Test', providerModelId: modelId,
    fitClass: 'full_accelerator', availability: { _tag: 'Available' },
    fitAssessment: { _tag: 'NotAssessed' }, explanation: 'test', residency: 'loaded',
  }],
  operations: [],
  recommendationState: { _tag: 'Ready', recommendations: [] },
  warnings: [],
}

test('ready status aggregates participating GPU domains and opens hardware settings', async () => {
  const open = vi.fn()
  const view = await testRender(
    <LocalRuntimeStatusBar state={state} width={100} onOpenHardware={open} />,
    { width: 110, height: 5 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain('Qwen Test')
    expect(frame).toContain('Ready')
    expect(frame).toContain('█')
    expect(frame).toContain('Memory 38 GiB / 48 GiB')
    const line = frame.split('\n').findIndex((value) => value.includes('Memory 38 GiB'))
    const column = frame.split('\n')[line]!.indexOf('Memory 38 GiB')
    await act(async () => view.mockMouse.click(column, line))
    expect(open).toHaveBeenCalledOnce()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test('latest failed operation remains visible as the single model state', async () => {
  const failed: LocalInferenceState = {
    ...state,
    choices: [],
    operations: [{
      operationId: 'operation',
      requestId: 'request',
      kind: 'activate',
      target: { _tag: 'model', selectionId: 'model' },
      providerModelId: modelId,
      status: 'failed',
      stage: 'loading',
      failure: { code: 'runtime_start_failed', message: 'details', retryable: true },
      startedAt: '2026-07-21T00:00:00.000Z',
      updatedAt: '2026-07-21T00:00:01.000Z',
    }],
  }
  const view = await testRender(
    <LocalRuntimeStatusBar state={failed} width={100} onOpenHardware={() => {}} />,
    { width: 110, height: 5 },
  )
  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain('Failed · loading')
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
