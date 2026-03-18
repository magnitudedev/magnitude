import { describe, expect, it } from 'bun:test'
import type { DisplayMessage, ThinkBlockStep } from '@magnitudedev/agent'
import { selectLatestLiveActivityFromMessages, selectLatestLiveActivityFromThinkSteps } from './live-activity'

describe('live-activity selector', () => {
  it('selects latest displayable activity by recency across message types', () => {
    const messages = [
      {
        type: 'think_block',
        steps: [{ id: '1', type: 'thinking', content: 'first thought' }],
      },
      { type: 'agent_communication', preview: 'latest communication' },
    ] as any as DisplayMessage[]

    expect(selectLatestLiveActivityFromMessages(messages)).toBe('latest communication')
  })

  it('uses tool live-text semantics before fallback label', () => {
    const steps = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'navigate',
        label: 'Generic tool label',
        visualState: { label: 'Navigate to ', detail: 'https://example.com' },
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps(steps)).toBe('Navigate to https://example.com')
  })

  it('falls back to tool label when no visual live text exists', () => {
    const steps = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'unknownTool',
        label: 'Fallback label',
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps(steps)).toBe('Fallback label')
  })

  it('formats browser live text with empty detail cleanly', () => {
    const steps = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'click',
        label: 'Generic tool label',
        visualState: { label: 'Click element', detail: '   ' },
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps(steps)).toBe('Click element')
  })

  it('falls back to label for unsupported artifactCreate tool key', () => {
    const steps = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'artifactCreate',
        label: 'Creating artifact via label fallback',
        visualState: { phase: 'streaming', name: 'draft' },
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps(steps)).toBe('Creating artifact via label fallback')
  })

  it('uses progressive live text for active artifact/file tools', () => {
    const activeArtifactWrite = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'artifactWrite',
        label: 'fallback',
        visualState: { phase: 'streaming', name: 'draft' },
      },
    ] as any as ThinkBlockStep[]
    const activeFileEdit = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'fileEdit',
        label: 'fallback',
        visualState: { phase: 'running', path: 'src/app.ts' },
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps(activeArtifactWrite)).toBe('Writing artifact draft')
    expect(selectLatestLiveActivityFromThinkSteps(activeFileEdit)).toBe('Editing src/app.ts')
  })

  it('avoids awkward punctuation spacing in browser live text', () => {
    const steps = [
      {
        id: '1',
        type: 'tool',
        toolKey: 'navigate',
        label: 'Generic tool label',
        visualState: { label: 'Navigate to', detail: ' https://example.com ' },
      },
      {
        id: '2',
        type: 'tool',
        toolKey: 'evaluate',
        label: 'Generic tool label',
        visualState: { label: 'Evaluate(', detail: 'window.location)' },
      },
      {
        id: '3',
        type: 'tool',
        toolKey: 'type',
        label: 'Generic tool label',
        visualState: { label: 'Type value', detail: ': "hello"' },
      },
    ] as any as ThinkBlockStep[]

    expect(selectLatestLiveActivityFromThinkSteps([steps[0]])).toBe('Navigate to https://example.com')
    expect(selectLatestLiveActivityFromThinkSteps([steps[1]])).toBe('Evaluate(window.location)')
    expect(selectLatestLiveActivityFromThinkSteps([steps[2]])).toBe('Type value: "hello"')
  })
})