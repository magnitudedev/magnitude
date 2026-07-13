import type { ImagePart, TextPart } from "./parts"
import type { Message, UserMessage, ToolResultMessage } from "./messages"
import { Prompt } from "./prompt"

export function imagePlaceholder(part: ImagePart): string {
  const segments: string[] = ['Image placeholder: current model does not support images']
  const meta: string[] = []
  if (part.dimensions) meta.push(`${part.dimensions.width}x${part.dimensions.height}`)
  else if (part.mediaType) meta.push(part.mediaType)
  if (meta.length > 0) segments.push('—', meta.join(' '))
  return `[${segments.join(' ')}]`
}

function normalizePartsVision(
  parts: readonly (TextPart | ImagePart)[],
  format: (part: ImagePart) => string,
): readonly (TextPart | ImagePart)[] {
  let changed = false
  const result = parts.map((part) => {
    if (part._tag === "ImagePart") {
      changed = true
      return {
        _tag: "TextPart" as const,
        text: format(part),
      }
    }
    return part
  })
  return changed ? result : parts
}

/**
 * Replace all ImageParts in a prompt with placeholder TextParts.
 * Pure function — returns a new Prompt if any images were found, otherwise the original.
 */
export function normalizeVision(prompt: Prompt, format: (part: ImagePart) => string = imagePlaceholder): Prompt {
  let changed = false
  const messages = prompt.messages.map((msg): Message => {
    switch (msg._tag) {
      case "UserMessage": {
        const parts = normalizePartsVision(msg.parts, format)
        if (parts !== msg.parts) {
          changed = true
          return { ...msg, parts } as UserMessage
        }
        return msg
      }
      case "ToolResultMessage": {
        const parts = normalizePartsVision(msg.parts, format)
        if (parts !== msg.parts) {
          changed = true
          return { ...msg, parts } as ToolResultMessage
        }
        return msg
      }
      case "AssistantMessage":
        return msg
    }
  })

  if (!changed) return prompt

  return Prompt.from({
    system: prompt.system,
    messages: messages as any,
  })
}
