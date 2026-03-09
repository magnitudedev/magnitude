import { describe, it, expect } from 'bun:test'
import { createXmlRuntime } from '../execution/xml-runtime'
import { foldReactorState, initialReactorState } from '../execution/reactor-state'
import type { XmlRuntimeEvent, ReactorState } from '../types'
import { Stream, Effect, Layer } from 'effect'
import { Schema } from '@effect/schema'
import { createTool } from '@magnitudedev/tools'

// Minimal gather-like tool
const gatherTool = createTool({
  name: 'gather',
  group: 'default',
  description: 'gather context',
  inputSchema: Schema.Struct({
    targets: Schema.Array(Schema.Struct({
      path: Schema.String,
      query: Schema.String,
    })),
  }),
  outputSchema: Schema.String,
  argMapping: ['targets'],
  bindings: {
    xmlInput: {
      type: 'tag' as const,
      children: [{
        field: 'targets',
        tag: 'target',
        attributes: ['path'],
        body: 'query',
      }],
    },
    xmlOutput: { type: 'tag' as const },
  },
  execute: () => Effect.succeed('mock result'),
})

const GATHER_REG = {
  tool: gatherTool,
  tagName: 'gather',
  groupName: 'default',
  binding: {
    children: [{
      field: 'targets',
      tag: 'target',
      attributes: ['path'],
      body: 'query',
    }],
  },
  meta: { defKey: 'gather' },
  layerProvider: () => Effect.succeed(Layer.empty),
} as const

const LLM_OUTPUT = `<think>
Let me look at how ToolEvent is used vs TurnToolCall.
</think>
<actions>
<gather id="g1">
<target path="packages/agent/src">How is ToolEvent used?</target>
<target path="cli/src">How does the CLI consume ToolEvent?</target>
</gather>
<inspect>
<ref tool="g1" />
</inspect>
</actions>`

function makeRuntime() {
  return createXmlRuntime({
    tools: new Map([['gather', GATHER_REG]]),
  })
}

async function collectEvents(input: string, replayState?: ReactorState) {
  const events: XmlRuntimeEvent[] = []
  const runtime = makeRuntime()
  const xmlStream = Stream.fromIterable(input.split(''))
  const stream = runtime.streamWith(xmlStream, replayState ? { initialState: replayState } : undefined)

  await Effect.runPromise(
    Effect.scoped(
      stream.pipe(
        Stream.runForEach((event) => Effect.sync(() => { events.push(event) })),
      ),
    ),
  )
  return events
}

describe('think + actions tool execution', () => {
  it('executes gather after </think> on a fresh turn (no replay state)', async () => {
    const events = await collectEvents(LLM_OUTPUT)

    const toolStarted = events.filter(e => e._tag === 'ToolInputStarted')
    const toolExecEnded = events.filter(e => e._tag === 'ToolExecutionEnded')

    expect(toolStarted.length).toBe(1)
    expect(toolExecEnded.length).toBe(1)
    expect(toolExecEnded[0].result._tag).toBe('Success')
  })

  it('crash recovery: replaying same XML with same-turn replay state suppresses completed tool', async () => {
    // First run: execute normally, capture the toolCallId
    const firstRunEvents = await collectEvents(LLM_OUTPUT)
    const firstStarted = firstRunEvents.filter(e => e._tag === 'ToolInputStarted')
    expect(firstStarted.length).toBe(1)
    const gatherId = firstStarted[0].toolCallId

    // Build replay state as the real ReplayProjection would — only fold
    // ToolInputStarted, ToolInputParseError, ToolExecutionEnded (not TurnEnd)
    let replayState = initialReactorState()
    for (const event of firstRunEvents) {
      if (event._tag === 'ToolInputStarted' || event._tag === 'ToolInputParseError' || event._tag === 'ToolExecutionEnded') {
        replayState = foldReactorState(replayState, event)
      }
    }

    // Second run (crash recovery): same XML, replay state has completed gather
    const replayEvents = await collectEvents(LLM_OUTPUT, replayState)

    // Gather should be suppressed (already completed)
    const replayStarted = replayEvents.filter(e => e._tag === 'ToolInputStarted')
    expect(replayStarted.length).toBe(0)

    // But it should still yield TurnEnd
    const execEnd = replayEvents.filter(e => e._tag === 'TurnEnd')
    expect(execEnd.length).toBe(1)
  })

  it('new turn after resume: empty replay state does not suppress new tool calls', async () => {
    // This is the original bug scenario. After session resume, the replay projection
    // has been reset by turn_completed, so the runtime gets empty initial state.
    // The new gather call must execute normally.
    const events = await collectEvents(LLM_OUTPUT, initialReactorState())

    const toolStarted = events.filter(e => e._tag === 'ToolInputStarted')
    const toolExecEnded = events.filter(e => e._tag === 'ToolExecutionEnded')

    expect(toolStarted.length).toBe(1)
    expect(toolExecEnded.length).toBe(1)
    expect(toolExecEnded[0].result._tag).toBe('Success')
  })
})
