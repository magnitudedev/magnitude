import { Option } from "effect"
import { describe, expect, it, vi } from "vitest"

vi.mock("node:path", async () => (await vi.importActual<typeof import("node:path")>("node:path")).win32)

const { webAppFilePath } = await import("./web-app")

describe("browser app paths on Windows", () => {
  it("keeps the forward-slash keys the embedded app is stored under", () => {
    expect(webAppFilePath("/")).toEqual(Option.some("index.html"))
    expect(webAppFilePath("/assets/index-abc.js")).toEqual(Option.some("assets/index-abc.js"))
    expect(webAppFilePath("/favicon.svg")).toEqual(Option.some("favicon.svg"))
  })
})
