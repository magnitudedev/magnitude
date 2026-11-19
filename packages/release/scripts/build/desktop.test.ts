import { describe, expect, it } from "vitest"
import { desktopPackagerOptions } from "./desktop"

const options = (platform: "darwin" | "linux" | "win32") => desktopPackagerOptions({
  app: "/stage/app", outputDirectory: "/out", target: platform === "win32" ? { platform, arch: "x64" } : { platform, arch: "arm64" },
  electronVersion: "43.7.0", version: "1.2.3", revision: 7, extraResource: ["/stage/resources/magnitude"],
})

describe("desktop packaging", () => {
  it.each(["darwin", "linux", "win32"] as const)("registers no link scheme on %s", platform => {
    expect(options(platform).protocols).toBeUndefined()
  })
  it("keeps the bundle identity and versions", () => {
    expect(options("darwin")).toMatchObject({ appBundleId: "dev.magnitude.desktop", buildVersion: "7", appVersion: "1.2.3" })
    expect(options("win32")).toMatchObject({ buildVersion: "1.2.3.7", executableName: "Magnitude" })
  })
})
