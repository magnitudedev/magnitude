import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  BrowserTabIdSchema,
  BrowserViewportRectSchema,
  BrowserWorkspaceStateSchema,
} from "./embedded-browser"

describe("embedded browser wire schemas", () => {
  it("round-trips a workspace snapshot", () => {
    const id = BrowserTabIdSchema.make("tab-1")
    const value = {
      revision: 4,
      focusLocationRevision: 2,
      activeTabId: id,
      tabs: [{
        id,
        title: "Magnitude",
        url: "https://magnitude.run/",
        pendingUrl: null,
        faviconUrl: null,
        phase: "ready" as const,
        canGoBack: false,
        canGoForward: true,
        error: null,
        insecureUrl: null,
      }],
      permissionRequest: null,
      downloads: [],
    }

    const encoded = Schema.encodeSync(BrowserWorkspaceStateSchema)(value)
    expect(Schema.decodeUnknownSync(BrowserWorkspaceStateSchema)(encoded)).toEqual(value)
  })

  it("rejects empty IDs and negative viewport dimensions", () => {
    expect(() => Schema.decodeUnknownSync(BrowserTabIdSchema)("")).toThrow()
    expect(() => Schema.decodeUnknownSync(BrowserViewportRectSchema)({
      x: 0,
      y: 0,
      width: -1,
      height: 300,
    })).toThrow()
  })
})
