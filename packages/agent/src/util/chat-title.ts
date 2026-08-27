export const CHAT_TITLE_MAX_CHARACTERS = 50

/**
 * Derive a session title from the first user message without model inference.
 */
export function deriveChatTitle(userMessage: string): string | null {
  const normalized = userMessage.replace(/\s+/g, ' ').trim()
  if (!normalized) return null

  return Array.from(normalized).slice(0, CHAT_TITLE_MAX_CHARACTERS).join('')
}
