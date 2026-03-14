import type { ParseEvent, ParseStack, ParserConfig } from './types'
import { mkRootProse } from './types'
import type { TagSchema } from '../execution/binding-validator'
import { createId } from '../util'
import { getKeywords, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '../constants'
import { processChar } from './dispatch'
import { flushStack } from './flush'
import { FencePhase } from './types'
import { flushPendingWhitespace, rawEmitProse } from './prose'
import { flushPendingMessageNewline } from './message'

/** ID generator function type — injectable for replay determinism. */
export type IdGenerator = () => string

/** Default ID generator using cuid2. */
export const defaultIdGenerator: IdGenerator = createId

export interface StreamingXmlParser {
  processChunk(chunk: string): ParseEvent[]
  flush(): ParseEvent[]
}

export type RefResolver = (tag: string, recency: number, query?: string) => string | undefined

export function createStreamingXmlParser(
  knownTags: ReadonlySet<string>,
  childTagMap: ReadonlyMap<string, ReadonlySet<string>>,
  tagSchemas?: ReadonlyMap<string, TagSchema>,
  resolveRef?: RefResolver,
  generateId: IdGenerator = defaultIdGenerator,
  defaultMessageDest: string = 'user',
): StreamingXmlParser {
  const kw = getKeywords()
  const structuralTags = new Set([kw.actions, kw.think, kw.thinking, kw.lenses, 'inspect', kw.comms, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD])
  const config: ParserConfig = {
    knownTags,
    childTagMap,
    tagSchemas,
    resolveRef,
    generateId,
    defaultMessageDest,
    keywords: { actions: kw.actions, think: kw.think, thinking: kw.thinking, lenses: kw.lenses, comms: kw.comms, next: TURN_CONTROL_NEXT, yield: TURN_CONTROL_YIELD },
    structuralTags,
    actionsTags: new Set([...knownTags, ...structuralTags, 'message']),
    topLevelTags: new Set([...structuralTags, ...knownTags, 'message']),
    refTags: new Set(['ref']),
    messageTags: new Set(['message']),
  }

  const stack: ParseStack = [mkRootProse()]
  let emittedTurnControl: 'continue' | 'yield' | null = null

  function filterTurnControl(events: ParseEvent[]): ParseEvent[] {
    const filtered: ParseEvent[] = []
    for (const event of events) {
      if (event._tag !== 'TurnControl') {
        filtered.push(event)
        continue
      }
      if (emittedTurnControl === null) {
        emittedTurnControl = event.decision
        filtered.push(event)
      }
    }
    return filtered
  }

  return {
    processChunk(chunk: string): ParseEvent[] {
      const events: ParseEvent[] = []
      let i = 0
      while (i < chunk.length) {
        const frame = stack[stack.length - 1]
        if (frame?._tag === 'Prose') {
          const fence = frame.fence
          let canFastScan = fence.phase === FencePhase.Broken
          if (!canFastScan && fence.phase === FencePhase.LeadingWs && fence.buffer.length === 0) {
            const c = chunk[i]
            if (c !== ' ' && c !== '\t' && c !== '`' && c !== '\n' && c !== '<') {
              fence.phase = FencePhase.Broken
              canFastScan = true
            }
          }
          if (canFastScan) {
            const start = i
            while (i < chunk.length) {
              const c = chunk[i]
              if (c === '<' || c === '\n') break
              i++
            }
            if (i > start) {
              frame.lastCharNewline = false
              events.push(...flushPendingWhitespace(stack))
              events.push(...rawEmitProse(stack, chunk.slice(start, i)))
            }
            if (i < chunk.length) events.push(...filterTurnControl(processChar(stack, chunk[i++], config)))
            continue
          }
        } else if (frame?._tag === 'MessageBody' && !frame.pendingLt) {
          const start = i
          while (i < chunk.length) {
            const c = chunk[i]
            if (c === '<' || c === '\n') break
            i++
          }
          if (i > start) {
            events.push(...flushPendingMessageNewline(frame))
            const text = chunk.slice(start, i)
            frame.body += text
            events.push({ _tag: 'MessageBodyChunk', id: frame.id, text })
          }
          if (i < chunk.length) events.push(...filterTurnControl(processChar(stack, chunk[i++], config)))
          continue
        }
        events.push(...filterTurnControl(processChar(stack, chunk[i++], config)))
      }
      return events
    },
    flush(): ParseEvent[] {
      return filterTurnControl(flushStack(stack, config))
    },
  }
}