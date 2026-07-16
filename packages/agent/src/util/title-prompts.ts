export const CHAT_TITLE_PROMPT = `Generate a concise title for this conversation based on the user's first message.
The title should be 3-7 words in sentence case (capitalize only the first word and proper nouns).
Maximum 50 characters. Focus on the main task or topic.
Output only the title text with no quotes, labels, or formatting.

Examples:
Fix login button on mobile
Add OAuth authentication
Debug failing CI tests
Refactor API client error handling

Bad (too vague): Code changes
Bad (too long): Investigate and fix the issue where the login button does not respond on mobile devices
Bad (wrong case): Fix Login Button On Mobile`

const FALLBACK_TITLE_MAX_WORDS = 7
const FALLBACK_TITLE_MAX_CHARACTERS = 50

/**
 * Create an immediate, deterministic title from the first user message.
 * This keeps session naming useful even when model-based title generation is
 * unavailable, and gives the title worker a durable one-attempt guard.
 */
export function fallbackChatTitle(userMessage: string): string | null {
  const normalized = userMessage.replace(/\s+/g, ' ').trim()
  if (!normalized) return null

  const words = normalized.split(' ')
  const wordLimited = words.slice(0, FALLBACK_TITLE_MAX_WORDS).join(' ')
  const wasTruncated = words.length > FALLBACK_TITLE_MAX_WORDS
    || wordLimited.length > FALLBACK_TITLE_MAX_CHARACTERS
  const characterLimit = wasTruncated
    ? FALLBACK_TITLE_MAX_CHARACTERS - 1
    : FALLBACK_TITLE_MAX_CHARACTERS
  const characterLimited = wordLimited.length > characterLimit
    ? (() => {
        const sliced = wordLimited.slice(0, characterLimit).trimEnd()
        const withoutPartialWord = sliced.replace(/\s+\S*$/, '')
        return withoutPartialWord || sliced
      })()
    : wordLimited
  const withEllipsis = wasTruncated ? `${characterLimited.replace(/[.,;:!?-]+$/g, '')}…` : characterLimited

  return withEllipsis.charAt(0).toUpperCase() + withEllipsis.slice(1)
}
