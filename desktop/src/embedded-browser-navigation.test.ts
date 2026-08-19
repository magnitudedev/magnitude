import { describe, expect, it } from "vitest"
import {
  browserNavigationFailureMessage,
  isAllowedBrowserNavigation,
  isLoopbackHostname,
  resolveBrowserNavigation,
} from "./embedded-browser-navigation"

describe("embedded browser navigation", () => {
  it("normalizes secure URLs and bare hostnames", () => {
    expect(resolveBrowserNavigation("https://example.com/docs")).toEqual({
      _tag: "navigate",
      url: "https://example.com/docs",
    })
    expect(resolveBrowserNavigation("example.com/docs")).toEqual({
      _tag: "navigate",
      url: "https://example.com/docs",
    })
  })

  it("allows loopback HTTP and identifies non-loopback HTTP as insecure", () => {
    expect(resolveBrowserNavigation("localhost:5173/test")).toEqual({
      _tag: "navigate",
      url: "http://localhost:5173/test",
    })
    expect(resolveBrowserNavigation("http://example.com/")).toEqual({
      _tag: "insecure",
      url: "http://example.com/",
    })
    expect(isLoopbackHostname("127.99.2.3")).toBe(true)
    expect(isLoopbackHostname("::1")).toBe(true)
    expect(isLoopbackHostname("app.localhost")).toBe(true)
  })

  it("searches ordinary text", () => {
    expect(resolveBrowserNavigation("effect ts documentation")).toEqual({
      _tag: "navigate",
      url: "https://www.google.com/search?q=effect%20ts%20documentation",
    })
    expect(resolveBrowserNavigation("C++ unicode 日本語")).toEqual({
      _tag: "navigate",
      url: "https://www.google.com/search?q=C%2B%2B%20unicode%20%E6%97%A5%E6%9C%AC%E8%AA%9E",
    })
  })

  it("rejects privileged and executable schemes", () => {
    for (const value of [
      "file:///tmp/secret",
      "javascript:alert(1)",
      "data:text/html,test",
      "chrome://settings",
      "devtools://devtools",
      "magnitude://internal",
    ]) {
      expect(resolveBrowserNavigation(value)._tag).toBe("invalid")
      expect(isAllowedBrowserNavigation(value)).toBe(false)
    }
  })

  it("presents network failures without Chromium internals", () => {
    expect(browserNavigationFailureMessage(-105)).toBe("The server could not be found.")
    expect(browserNavigationFailureMessage(-102)).toBe("The server refused the connection.")
    expect(browserNavigationFailureMessage(-202)).toBe("This site’s identity could not be verified.")
    expect(browserNavigationFailureMessage(-999)).toBe("This page could not be loaded.")
  })
})
