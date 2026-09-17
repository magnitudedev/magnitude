import { FileSystem } from "@effect/platform"
import { NodePath } from "@effect/platform-node"
import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { CliBinaryResolver, cliBinaryResolverLayer, type DesktopLocation } from "./cli-binary-resolver"

const resolve = (platform: string, files: string[], application?: string, executable = true) => {
  const location: DesktopLocation = { platform, home: "/Users/test", application: Option.fromNullable(application), localAppData: Option.some("C:\\Users\\test\\AppData\\Local") }
  const fs = FileSystem.makeNoop({
    stat: file => Effect.succeed({ type: files.includes(file) ? "File" : "Directory", mode: executable ? 0o755 : 0o644 } as FileSystem.File.Info),
    access: () => Effect.void,
  })
  return Effect.runPromise(CliBinaryResolver.pipe(Effect.flatMap(resolver => resolver.resolve), Effect.either,
    Effect.provide(cliBinaryResolverLayer(location).pipe(Layer.provide([
      Layer.succeed(FileSystem.FileSystem, fs), platform === "win32" ? NodePath.layerWin32 : NodePath.layerPosix,
    ])))))
}

describe("desktop CLI locations", () => {
  it("finds a Mac app in the user's Applications directory", async () => {
    const app = "/Users/test/Applications/Magnitude.app"
    expect(await resolve("darwin", [`${app}/Contents/MacOS/Magnitude`, `${app}/Contents/Resources/magnitude`])).toMatchObject({ right: { application: app } })
  })
  it("keeps the Linux guarded desktop entry point", async () => {
    expect(await resolve("linux", ["/usr/bin/magnitude-desktop", "/usr/lib/magnitude-desktop/resources/magnitude"])).toMatchObject({ right: {
      application: "/usr/bin/magnitude-desktop", executable: "/usr/lib/magnitude-desktop/resources/magnitude",
    } })
  })
  it("finds the Windows per-user installation", async () => {
    const app = "C:\\Users\\test\\AppData\\Local\\Programs\\Magnitude"
    expect(await resolve("win32", [`${app}\\Magnitude.exe`, `${app}\\resources\\magnitude.exe`])).toMatchObject({ right: { executable: `${app}\\resources\\magnitude.exe` } })
  })
  it("honors a custom installation with spaces", async () => {
    expect(await resolve("linux", ["/custom app/magnitude", "/custom app/resources/magnitude"], "/custom app/magnitude")).toMatchObject({ right: { application: "/custom app/magnitude" } })
  })
  it("does not fall back from a missing explicit application", async () => {
    expect((await resolve("linux", ["/usr/bin/magnitude-desktop", "/usr/lib/magnitude-desktop/resources/magnitude"], "/missing"))._tag).toBe("Left")
  })
  it("rejects relative paths, missing CLI, non-executable files and unsupported platforms", async () => {
    for (const result of await Promise.all([
      resolve("linux", [], "relative"), resolve("linux", ["/usr/bin/magnitude-desktop"]),
      resolve("linux", ["/usr/bin/magnitude-desktop", "/usr/lib/magnitude-desktop/resources/magnitude"], undefined, false), resolve("freebsd", []),
    ])) expect(result._tag).toBe("Left")
  })
})
