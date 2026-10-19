/** Parse one complete Server-Sent Events block from an OpenAI-compatible chat stream. */
export function consumeChatEvent(block: string, onText: (text: string) => void): boolean {
  let finished = false
  for (const line of block.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue
    const data = line.slice(5).trimStart()
    if (data === "[DONE]") {
      finished = true
      continue
    }
    if (!data) continue
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: unknown } }>
      error?: { message?: string }
    }
    if (event.error?.message) throw new Error(event.error.message)
    const content = event.choices?.[0]?.delta?.content
    if (typeof content === "string" && content.length > 0) onText(content)
  }
  return finished
}
