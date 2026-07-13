export interface AgentCommunicationLike {
  readonly id: string
  readonly type: 'agent_communication'
  readonly streamId?: string
  readonly direction: 'to_agent' | 'from_agent'
  readonly agentId: string
  readonly content: string
  readonly preview: string
}

export function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}

export function upsertCommunicationStream(
  messages: readonly AgentCommunicationLike[],
  streamId: string,
  base: Omit<AgentCommunicationLike, 'id' | 'type' | 'streamId' | 'content' | 'preview'>,
  textDelta: string,
  createId: () => string
): AgentCommunicationLike[] {
  const existing = messages.find(m => m.streamId === streamId)
  if (existing) {
    const content = existing.content + textDelta
    return messages.map(m => m.id === existing.id ? { ...m, content, preview: toPreview(content) } : m)
  }

  const content = textDelta
  return [
    ...messages,
    {
      id: createId(),
      type: 'agent_communication',
      streamId,
      ...base,
      content,
      preview: toPreview(content),
    }
  ]
}

export function finalizeCommunicationStream(
  messages: readonly AgentCommunicationLike[],
  streamId: string
): AgentCommunicationLike[] {
  const streamMsg = messages.find(m => m.streamId === streamId)
  if (!streamMsg) return [...messages]

  const normalized = streamMsg.content.trim()
  const duplicateSeed = normalized.length > 0
    ? messages.find(m =>
      !m.streamId
      && m.id !== streamMsg.id
      && m.direction === streamMsg.direction
      && m.agentId === streamMsg.agentId
      && m.content.trim() === normalized
    )
    : undefined

  if (duplicateSeed) {
    return messages.filter(m => m.id !== streamMsg.id)
  }

  return messages.map(m => m.id === streamMsg.id ? { ...m, preview: toPreview(m.content) } : m)
}