import { Effect, Schema } from "effect"

export class ClipboardWriteFailed extends Schema.TaggedError<ClipboardWriteFailed>()("ClipboardWriteFailed", {}) {}

/** Copies through a hidden selection, which browsers still allow on plain-HTTP pages. */
const copyThroughSelection = (text: string) => Effect.try({
  try: () => {
    const field = document.createElement("textarea")
    field.value = text
    field.setAttribute("readonly", "")
    field.style.position = "fixed"
    field.style.opacity = "0"
    document.body.append(field)
    field.select()
    try {
      if (!document.execCommand("copy")) throw new Error("copy was refused")
    } finally {
      field.remove()
    }
  },
  catch: () => new ClipboardWriteFailed(),
})

/** Writes text to the viewer's clipboard: the Clipboard API where the page is a secure context, else the selection fallback. */
export const writeClipboardText = (text: string): Effect.Effect<void, ClipboardWriteFailed> =>
  window.isSecureContext && navigator.clipboard
    ? Effect.tryPromise({ try: () => navigator.clipboard.writeText(text), catch: () => new ClipboardWriteFailed() }).pipe(
      Effect.orElse(() => copyThroughSelection(text)))
    : copyThroughSelection(text)
