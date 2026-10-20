import { describe, expect, it } from "vitest"
import { normalizeLocalEmbedUrl } from "./open-webui-url"

describe("normalizeLocalEmbedUrl", () => {
  it("accepts local HTTP and HTTPS URLs", () => {
    expect(normalizeLocalEmbedUrl("http://127.0.0.1:3000")).toBe("http://127.0.0.1:3000/")
    expect(normalizeLocalEmbedUrl("http://localhost:3000/")).toBe("http://localhost:3000/")
    expect(normalizeLocalEmbedUrl("https://[::1]:3000/chat")).toBe("https://[::1]:3000/chat")
  })

  it("rejects remote hosts and non-web protocols", () => {
    expect(normalizeLocalEmbedUrl("https://example.com")).toBeNull()
    expect(normalizeLocalEmbedUrl("file:///etc/passwd")).toBeNull()
    expect(normalizeLocalEmbedUrl("javascript:alert(1)")).toBeNull()
  })

  it("rejects credentials and malformed values", () => {
    expect(normalizeLocalEmbedUrl("http://user:pass@localhost:3000")).toBeNull()
    expect(normalizeLocalEmbedUrl("not a URL")).toBeNull()
    expect(normalizeLocalEmbedUrl("")).toBeNull()
  })

  it("strips fragments before embedding", () => {
    expect(normalizeLocalEmbedUrl("http://localhost:3000/#token")).toBe("http://localhost:3000/")
  })
})
