import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { ApplicationPage } from "./contracts"
import { pageFromPath, pagePath } from "./router"

describe("application routes", () => {
  it("round-trips every page through its URL path", () => {
    for (const page of ApplicationPage.literals) expect(pageFromPath(pagePath(page))).toEqual(Option.some(page))
  })
  it("treats the root and trailing slashes as the same route", () => {
    expect(pageFromPath("/")).toEqual(Option.some("discover"))
    expect(pageFromPath("/catalog/")).toEqual(Option.some("catalog"))
  })
  it("rejects paths that are not pages", () => {
    expect(pageFromPath("/inference/v1")).toEqual(Option.none())
    expect(pageFromPath("/settings/network")).toEqual(Option.none())
  })
})
