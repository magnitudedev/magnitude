import { createStreamingXmlParser, type ParseEvent } from '@magnitudedev/xml-act'

export interface ParsedResponse {
  agentCreates: Array<{ type: string; agentId: string }>
  proposes: Array<{ title: string }>
  userMessages: string[]
  hasThinkBlock: boolean
  hasSubmit: boolean
}

export function parseOrchestratorResponse(raw: string): ParsedResponse {
  const knownTags = new Set(['agent-create', 'propose', 'submit'])
  const childTagMap = new Map<string, ReadonlySet<string>>([
    ['agent-create', new Set(['type'])],
    ['propose', new Set()],
    ['submit', new Set()],
  ])

  const parser = createStreamingXmlParser(knownTags, childTagMap)

  const events: ParseEvent[] = [
    ...parser.processChunk(raw),
    ...parser.flush(),
  ]

  const agentCreates: Array<{ type: string; agentId: string }> = []
  const proposes: Array<{ title: string }> = []
  const userMessages: string[] = []

  let hasThinkEnd = false
  let hasUnclosedThink = false
  let hasSubmit = false

  const messageBodies = new Map<string, string>()
  const userMessageIds = new Set<string>()

  for (const event of events) {
    if (event._tag === 'ProseEnd' && event.patternId === 'think') {
      hasThinkEnd = true
      continue
    }

    if (event._tag === 'ParseError' && event.error._tag === 'UnclosedThink') {
      hasUnclosedThink = true
      continue
    }

    if (event._tag === 'TagClosed') {
      if (event.tagName === 'agent-create') {
        const typeChild = event.element.children.find(child => child.tagName === 'type')
        const type = typeChild?.body.trim() ?? ''
        const agentId = String(event.element.attributes.get('agentId') ?? '')
        agentCreates.push({ type, agentId })
      } else if (event.tagName === 'propose') {
        const title = String(event.element.attributes.get('title') ?? '')
        proposes.push({ title })
      } else if (event.tagName === 'submit') {
        hasSubmit = true
      }
      continue
    }

    if (event._tag === 'MessageTagOpen') {
      messageBodies.set(event.id, '')
      if (event.dest === 'user') userMessageIds.add(event.id)
      continue
    }

    if (event._tag === 'MessageBodyChunk') {
      messageBodies.set(event.id, (messageBodies.get(event.id) ?? '') + event.text)
      continue
    }

    if (event._tag === 'MessageTagClose') {
      if (userMessageIds.has(event.id)) {
        userMessages.push((messageBodies.get(event.id) ?? '').trim())
      }
      messageBodies.delete(event.id)
      userMessageIds.delete(event.id)
    }
  }

  return {
    agentCreates,
    proposes,
    userMessages,
    hasThinkBlock: hasThinkEnd && !hasUnclosedThink,
    hasSubmit,
  }
}