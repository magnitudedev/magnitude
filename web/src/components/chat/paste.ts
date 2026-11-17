/** The longest message the composer holds, in characters. */
export const MESSAGE_CHARS = 1_048_576

/**
 * Whether pasting `pasted` over the selection `[start, end)` of `text` would overrun a message. The message box would
 * otherwise keep only what fits and silently drop the rest of a long log.
 */
export const pasteTooLong = (text: string, start: number, end: number, pasted: string): boolean =>
  text.length - (end - start) + pasted.length > MESSAGE_CHARS

export const PASTE_TOO_LONG = `That text is too long to paste into a message, which holds up to ${MESSAGE_CHARS.toLocaleString("en-US")} characters. Paste a shorter part of it.`

/** The notice once the message is edited, a paste that fit included: a refused paste's notice no longer applies. */
export const noticeAfterEdit = (notice: string | null): string | null => notice === PASTE_TOO_LONG ? null : notice
