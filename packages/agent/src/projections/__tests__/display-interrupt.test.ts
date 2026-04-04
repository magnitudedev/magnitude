import { describe, expect, test } from 'bun:test'
import { finalizeOpenToolStepsAsInterruptedInSteps } from '../display-interrupt'

describe('display interrupt finalization helper used by display projection', () => {
  test('marks unfinished tool steps as interrupted and leaves finished/non-tool steps unchanged', () => {
    const original: Array<{ id: string; type: string; toolKey?: string; state?: { phase: string; done?: unknown }; result?: { status: 'success' | 'interrupted'; output?: string }; content?: string }> = [
      { id: 'open-no-visual', type: 'tool' as const, toolKey: 'shell', state: { phase: 'executing' } },
      { id: 'open-with-visual', type: 'tool' as const, toolKey: 'shell', state: { phase: 'executing' } },
      { id: 'done', type: 'tool' as const, toolKey: 'shell', state: { phase: 'completed' }, result: { status: 'success' as const, output: 'ok' } },
      { id: 'think', type: 'thinking' as const, content: 'x' },
    ]

    const next = finalizeOpenToolStepsAsInterruptedInSteps(
      original,
      (_toolKey, visualState) => ({ ...(visualState as Record<string, unknown>), phase: 'interrupted', done: { kind: 'interrupted' } })
    )

    const openNoVisual = next.find((s: any) => s.id === 'open-no-visual')
    const openWithVisual = next.find((s: any) => s.id === 'open-with-visual')
    const done = next.find((s: any) => s.id === 'done')
    const think = next.find((s: any) => s.id === 'think')

    expect((openNoVisual as any)?.state).toEqual({ phase: 'interrupted', done: { kind: 'interrupted' } })
    expect((openWithVisual as any)?.state).toEqual({ phase: 'interrupted', done: { kind: 'interrupted' } })
    expect((done as any)?.result).toEqual({ status: 'success', output: 'ok' })
    expect(think).toEqual({ id: 'think', type: 'thinking', content: 'x' })
  })
})