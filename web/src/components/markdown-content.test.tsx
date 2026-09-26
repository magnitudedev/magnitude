import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { MarkdownContent, allowlistedLinkHref } from "./markdown-content"

const render = (content: string) =>
  renderToStaticMarkup(<MarkdownContent content={content} />)

describe("markdown link protocols", () => {
  it("keeps http, https, mailto, and relative targets clickable", () => {
    expect(allowlistedLinkHref("https://magnitude.dev/docs")).toBe(
      "https://magnitude.dev/docs"
    )
    expect(allowlistedLinkHref("http://localhost:5173")).toBe(
      "http://localhost:5173"
    )
    expect(allowlistedLinkHref("mailto:dev@magnitude.dev")).toBe(
      "mailto:dev@magnitude.dev"
    )
    expect(allowlistedLinkHref("docs/getting-started.md")).toBe(
      "docs/getting-started.md"
    )
    expect(allowlistedLinkHref("#installation")).toBe("#installation")
    expect(allowlistedLinkHref(" /docs/guide?tab=cli ")).toBe(
      "/docs/guide?tab=cli"
    )

    const html = render(
      "[docs](https://magnitude.dev) [mail](mailto:dev@magnitude.dev) [top](#top)"
    )

    expect(html).toContain('href="https://magnitude.dev"')
    expect(html).toContain('href="mailto:dev@magnitude.dev"')
    expect(html).toContain('href="#top"')
    expect(html.match(/rel="noopener noreferrer"/g)).toHaveLength(3)
  })

  it("rejects every non-allowlisted protocol however it is spelled", () => {
    // Browsers ignore ASCII whitespace and control characters inside a
    // protocol, so the scheme has to be read the same way to be compared.
    const nulObfuscated = `java${String.fromCharCode(0)}script:alert(1)`
    const spaceObfuscated = "java script:alert(1)"

    for (const href of [
      "javascript:alert(1)",
      "JavaScript:alert(1)",
      " javascript:alert(1)",
      "java\tscript:alert(1)",
      "java\nscript:alert(1)",
      nulObfuscated,
      spaceObfuscated,
      "vbscript:msgbox(1)",
      "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
      "file:///etc/passwd",
      "blob:https://magnitude.dev/2b5d",
      "about:blank",
    ]) {
      expect(allowlistedLinkHref(href)).toBeUndefined()
    }

    expect(allowlistedLinkHref(undefined)).toBeUndefined()
    expect(allowlistedLinkHref("   ")).toBeUndefined()
  })
})

describe("markdown link rendering", () => {
  it("renders a rejected protocol as text rather than a link", () => {
    const html = render("[run it](javascript:alert(1))")

    expect(html).not.toContain("<a")
    expect(html).not.toContain("javascript:")
    expect(html).toContain("run it")
  })

  it("does not leave an empty href that reloads the app when clicked", () => {
    const html = render(
      "[script](javascript:alert(1)) [doc](data:text/plain,hi) [script file](file:///etc/passwd)"
    )

    expect(html).not.toContain("<a")
    expect(html).not.toContain('href=""')
  })

  it("keeps the link text of a rejected protocol, including its formatting", () => {
    const html = render("[**bold** claim](javascript:alert(1))")

    expect(html).not.toContain("<a")
    expect(html).toContain(">bold</strong> claim")
  })
})
