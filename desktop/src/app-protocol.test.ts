import * as nodePath from "node:path"
import { describe, expect, it } from "vitest"
import { resolveMagnitudeAppAssetPath } from "./app-protocol"

describe("Magnitude app protocol", () => {
  const rendererRoot = nodePath.resolve("out/renderer")

  it("maps app URLs to renderer assets", () => {
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://app/"))
      .toBe(nodePath.join(rendererRoot, "index.html"))
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://app/assets/app.js"))
      .toBe(nodePath.join(rendererRoot, "assets", "app.js"))
  })

  it("rejects other hosts and paths outside the renderer root", () => {
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://other/index.html")).toBeNull()
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://app:8080/index.html")).toBeNull()
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://app/%2F..%2Fsecret.txt")).toBeNull()
    expect(resolveMagnitudeAppAssetPath(rendererRoot, "magnitude://app/%E0%A4%A")).toBeNull()
  })
})
