import { describe, expect, it } from 'vitest'
import { Effect, Layer, Stream } from 'effect'
import {
  createStreamingFieldParser,
  ModelStreamTerminal,
  type ProviderToolCallId,
  type ResponseStreamEvent,
  type ToolCallId,
} from '@magnitudedev/ai'
import { dispatch, type HarnessEvent } from '@magnitudedev/harness'
import { Fs, FsSearchError, type FsSearchMatch } from '../src/services/fs'
import { WorkingDirectoryTag } from '../src/execution/working-directory'
import { fsToolkit } from '../src/tools/toolkits'
import { grepTool } from '../src/tools/fs'

const toolCallId = 'grep-call' as ToolCallId
const providerToolCallId = 'grep-call' as ProviderToolCallId

function terminal(): ResponseStreamEvent {
  return {
    _tag: 'stream_end',
    terminal: ModelStreamTerminal.StreamCompleted({
      call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
      response: { status: 200, headers: [], requestId: null },
      finishReason: 'tool_calls',
      progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
      usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
    }),
  }
}

async function executeSearch(
  result: readonly FsSearchMatch[] | FsSearchError,
): Promise<readonly HarnessEvent[]> {
  const parser = createStreamingFieldParser(grepTool.definition.inputSchema)
  parser.push(JSON.stringify({ pattern: 'model', path: '.', limit: 50 }))
  parser.end()
  const events: HarnessEvent[] = []
  const modelEvents: ResponseStreamEvent[] = [
    { _tag: 'tool_call_start', toolCallId, providerToolCallId, toolName: 'grep' },
    { _tag: 'tool_call_ready', toolCallId, providerToolCallId },
    terminal(),
  ]
  const fs = {
    readFile: () => Effect.die('unused'),
    readText: () => Effect.die('unused'),
    writeFile: () => Effect.die('unused'),
    stat: () => Effect.die('unused'),
    exists: () => Effect.succeed(true),
    walk: () => Effect.die('unused'),
    search: () => result instanceof FsSearchError ? Effect.fail(result) : Effect.succeed(result),
  }
  const layer = Layer.merge(
    Layer.succeed(Fs, fs),
    Layer.succeed(WorkingDirectoryTag, { cwd: '/workspace', scratchpadPath: '/scratchpad' }),
  )

  await Effect.runPromise(dispatch({
    events: Stream.fromIterable(modelEvents),
    parsers: new Map([[toolCallId, parser]]),
    toolkit: fsToolkit,
    layer: layer as unknown as Layer.Layer<unknown>,
    emit: (event) => Effect.sync(() => { events.push(event) }),
    requestId: null,
  }))
  return events
}

describe('grep tool lifecycle', () => {
  it.each([
    ['matches', [{ file: 'models.ts', match: '1|model' }]],
    ['no matches', []],
  ] as const)('emits exactly one terminal event for %s', async (_label, result) => {
    const events = await executeSearch(result)
    const toolEvents = events.filter((event) => 'toolCallId' in event && event.toolCallId === toolCallId)
    expect(toolEvents.filter((event) => event._tag === 'ToolExecutionStarted')).toHaveLength(1)
    expect(toolEvents.filter((event) => event._tag === 'ToolExecutionEnded')).toHaveLength(1)
  })

  it('emits exactly one terminal event for a search failure', async () => {
    const events = await executeSearch(new FsSearchError({
      reason: 'timeout',
      path: '/workspace',
      message: 'Search timed out after 5s',
    }))
    const toolEvents = events.filter((event) => 'toolCallId' in event && event.toolCallId === toolCallId)
    expect(toolEvents.filter((event) => event._tag === 'ToolExecutionStarted')).toHaveLength(1)
    expect(toolEvents.filter((event) => event._tag === 'ToolExecutionEnded')).toHaveLength(1)
  })

  it('forwards only the bounded process diagnostic to the terminal tool event', async () => {
    const marker = ' … [truncated]'
    const detail = `${'x'.repeat(300 - marker.length)}${marker}`
    const message = `Ripgrep exited with code 2: ${detail}`
    const events = await executeSearch(new FsSearchError({
      reason: 'process',
      path: '/workspace',
      message,
    }))
    const ended = events.find((event) =>
      event._tag === 'ToolExecutionEnded' && event.toolCallId === toolCallId
    ) as Extract<HarnessEvent, { _tag: 'ToolExecutionEnded' }> | undefined
    expect(ended?.result).toEqual({
      _tag: 'Error',
      error: { _tag: 'FsError', message },
    })
    expect(detail).toHaveLength(300)
  })
})
