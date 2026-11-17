/** The longest run of non-space characters a table cell keeps whole; past it a token (a URL, a hash) may wrap anywhere. */
export const LONG_TOKEN_LENGTH = 24

interface HastNode {
  readonly type: string
  readonly value?: string
  readonly tagName?: string
  readonly properties?: Record<string, unknown>
  children?: HastNode[]
}

const longToken = new RegExp(`(\\S{${LONG_TOKEN_LENGTH + 1},})`)

/** A text node split so each overlong token sits in its own span that may break anywhere. */
const splitText = (value: string): HastNode[] => value.split(longToken).flatMap((part, index): HastNode[] =>
  part === "" ? []
    : index % 2 === 1 ? [{ type: "element", tagName: "span", properties: { className: ["chat-long-token"] }, children: [{ type: "text", value: part }] }]
    : [{ type: "text", value: part }])

const markLongTokens = (node: HastNode): void => {
  if (!node.children) return
  node.children = node.children.flatMap(child => {
    if (child.type === "text") return splitText(child.value ?? "")
    markLongTokens(child)
    return [child]
  })
}

const visitCells = (node: HastNode): void => {
  if (node.type === "element" && (node.tagName === "td" || node.tagName === "th")) return markLongTokens(node)
  node.children?.forEach(visitCells)
}

/**
 * Lets only overlong tokens in table cells wrap. A cell's narrowest width is its longest unbreakable run, so a table
 * with a long URL or hash can't fit its column without scrolling; letting every word break anywhere instead lets the
 * table squeeze its other columns until short words split. Marking only the long tokens breakable keeps every
 * ordinary word whole and lets the long one wrap.
 */
export const rehypeLongTokens = () => (tree: HastNode) => { visitCells(tree) }
