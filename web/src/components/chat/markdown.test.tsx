import { renderToStaticMarkup } from "react-dom/server"
import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ChatGeneration,
  ChatSettings,
  defaultChatSettings,
} from "@magnitudedev/sdk"
import { ChatResponse } from "./response"
import { codeLanguageLabel, highlightCode } from "./code-highlight"
const render = (content: string, status = "completed") =>
  renderToStaticMarkup(
    <ChatResponse
      generation={Schema.decodeUnknownSync(ChatGeneration)({
        id: "g",
        commandId: "c",
        messageId: "m",
        modelId: "local",
        settings: Schema.encodeSync(ChatSettings)(defaultChatSettings),
        status,
        createdAt: 0,
        updatedAt: 1,
        content,
        reasoning: "",
        metrics: {},
      })}
      notify={() => {}}
    />
  )
const codeContents = (html: string) =>
  html.match(/<pre[^>]*><code>([\s\S]*?)<\/code><\/pre>/g) ?? []
const text = (html: string) => html.replace(/<[^>]+>/g, "")
describe("Chat Markdown code blocks", () => {
  it("gives each fenced block its own language label and copy control", () => {
    const html = render(
      "```python\ndef fib(n):\n    return n\n```\n\n```js\nconst n = 1\n```"
    )
    expect(html.match(/aria-label="Copy code"/g)).toHaveLength(2)
    expect(html).toContain('aria-label="Python code"')
    expect(html).toContain('aria-label="JavaScript code"')
    expect(html).toContain('class="hljs-keyword"')
    expect(text(codeContents(html)[0]!)).toBe("def fib(n):\n    return n")
  })
  it("keeps inline code simple, without a copy button", () => {
    const html = render("Use `README.md` and `n` here.")
    expect(html).toContain("<code>README.md</code>")
    expect(html).not.toContain('aria-label="Copy code"')
    expect(html).not.toContain("chat-code-block")
  })
  it("labels unlabelled and indented code as Text without guessing", () => {
    for (const content of ["```\nx = 1\n```", "    x = 1"]) {
      const html = render(content)
      expect(html).toContain('aria-label="Text code"')
      expect(html).toContain('aria-label="Copy code"')
      expect(html).not.toContain("hljs-")
    }
  })
  it("preserves exact code whitespace and literal Markdown while omitting fences", () => {
    const code = "# Heading\n\n- Item  \n\n"
    const html = render("```markdown\n" + code + "\n```")
    expect(text(codeContents(html)[0]!)).toBe(code)
    expect(html).not.toContain("<h1>")
  })
  it("keeps unknown languages readable and copyable", () => {
    const html = render("```unfamiliar-lang\n  x = 1\n```")
    expect(html).toContain('aria-label="unfamiliar-lang code"')
    expect(html).toContain('aria-label="Copy code"')
    expect(html).toContain("<code>  x = 1</code>")
  })
  it("escapes executable HTML in syntax-highlighted code", () => {
    const html = render(
      '```html\n<script>alert("x")</script>\n<img src=x onerror=alert(1)>\n```'
    )
    expect(html).not.toContain("<script>")
    expect(html).not.toContain("<img")
    expect(html).toContain("&lt;")
    expect(html).toContain("hljs-")
  })
  it("supports empty fences and unfinished fences while streaming", () => {
    expect(render("```python\n\n```")).toContain('aria-label="Copy code"')
    const html = render("```python\ndef fib(n):\n    return ", "running")
    expect(html).toContain('aria-label="Copy code"')
    expect(html).not.toContain('aria-label="Copy response"')
    expect(text(codeContents(html)[0]!)).toBe("def fib(n):\n    return ")
  })
  it("handles a code block nested in a quote or list", () => {
    const html = render(
      '> ```json\n> {"x": 1}\n> ```\n\n- Example\n\n  ```sh\n  echo hello\n  ```'
    )
    expect(html.match(/aria-label="Copy code"/g)).toHaveLength(2)
    expect(html).toContain('aria-label="JSON code"')
    expect(html).toContain('aria-label="Shell code"')
  })
  it.each([
    ["py", "Python"],
    ["js", "JavaScript"],
    ["ts", "TypeScript"],
    ["md", "Markdown"],
    ["yml", "YAML"],
    ["", "Text"],
    ["HTML", "HTML"],
  ])("labels the %s language alias as %s", (language, label) => {
    expect(codeLanguageLabel(language!)).toBe(label)
  })
  it.each([
    ["python", "def f():\n    return 42"],
    ["js", "const f = () => 42"],
    ["ts", "const n: number = 42"],
    ["tsx", "const x = <div>hi</div>"],
    ["bash", "if true; then echo hello; fi"],
    ["json", '{"x": true}'],
    ["rust", "fn main() { let x = 42; }"],
    ["go", "func main() { return }"],
    ["yaml", "name: hello"],
    ["sql", "SELECT * FROM users"],
  ])("colors %s without changing its text", (language, code) => {
    const html = renderToStaticMarkup(
      <code>{highlightCode(code!, language!)}</code>
    )
    expect(html).toContain("hljs-")
    // Decode XML escapes using the same React renderer for a fair comparison.
    const plain = renderToStaticMarkup(<code>{code}</code>)
    expect(html.replace(/<\/?span[^>]*>/g, "")).toBe(plain)
  })
  it("leaves large blocks as text to keep streaming responsive", () => {
    const code = "const x = 1;\n".repeat(5000)
    expect(highlightCode(code, "js")).toBe(code)
  })
})
describe("Chat Markdown structure", () => {
  it("retains all heading levels and paragraphs inside lists and quotes", () => {
    const html = render(
      "# One\n\n## Two\n\n### Three\n\n#### Four\n\n##### Five\n\n###### Six\n\n> First\n>\n> Second\n\n- First\n\n  Second\n\n  - Nested"
    )
    for (let i = 1; i <= 6; i++) expect(html).toContain(`<h${i}>`)
    expect(html).toContain("<blockquote>\n<p>First</p>\n<p>Second</p>")
    expect(html).toContain("<li>\n<p>First</p>\n<p>Second</p>\n<ul>")
  })
})
