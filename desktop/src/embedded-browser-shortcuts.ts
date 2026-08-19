export type EmbeddedBrowserShortcut =
  | "close-tab"
  | "focus-location"
  | "go-back"
  | "go-forward"
  | "new-tab"
  | "next-tab"
  | "previous-tab"
  | "reload"
  | "stop"

interface BrowserKeyboardInput {
  readonly type: string
  readonly key: string
  readonly control: boolean
  readonly meta: boolean
  readonly shift: boolean
}

export function embeddedBrowserShortcut(
  input: BrowserKeyboardInput,
  platform: NodeJS.Platform,
): EmbeddedBrowserShortcut | null {
  if (input.type !== "keyDown") return null

  const key = input.key.toLowerCase()
  if (input.control && key === "tab") {
    return input.shift ? "previous-tab" : "next-tab"
  }
  if (input.key === "Escape") return "stop"

  const modifier = platform === "darwin" ? input.meta : input.control
  if (!modifier) return null
  switch (key) {
    case "l": return "focus-location"
    case "t": return "new-tab"
    case "w": return "close-tab"
    case "r": return "reload"
    case "[": return "go-back"
    case "]": return "go-forward"
    default: return null
  }
}
