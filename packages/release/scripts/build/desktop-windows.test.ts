import { describe, expect, it } from "vitest"
import { Cause, Effect, Exit, Schema } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as NodeContext from "@effect/platform-node/NodeContext"
import { join } from "node:path"
import { WindowsInstallerInput, WindowsPayloadPath, buildWindowsDesktopInstaller, renderWindowsInstaller } from "./desktop-windows"
const template = "@PAYLOAD_FILES@\n@REMOVE_FILES@\n@REMOVE_DIRECTORIES@"
const input: typeof WindowsInstallerInput.Encoded = {
  version: "1.2.3-alpha.1", revision: 36, files: ["Magnitude.exe", "resources/app.asar"],
}
describe("Windows installer file contract", () => {
  it.each(["../outside", "/absolute", "a//b", "a/./b", "C:/outside", "a\\b", "CON.txt", "x/LPT1", "file.", "file ", "a\nfile", "a:file"])('rejects unsafe name %j', path => {
    expect(Schema.decodeUnknownEither(WindowsPayloadPath)(path)._tag).toBe("Left")
  })
  it.each([
    ["a", "A"], ["a", "a/b"], ["Resources/a", "resources/b"], ["Uninstall Magnitude.exe"], ["Uninstall Magnitude.exe/child"],
  ])("rejects case/ownership collisions %j", async (...files) => {
    await expect(Effect.runPromise(renderWindowsInstaller(template, { ...input, files: files as [string, ...string[]] }))).rejects.toThrow()
  })
  it("uses one payload list for extraction and handle-relative removal", async () => {
    const result = await Effect.runPromise(renderWindowsInstaller(template, input))
    expect(result).toContain('File "/oname=app.asar" "payload/000001.bin"')
    expect(result).toContain('RemovePayload(w "resources\\app.asar", i 0)')
    expect(result).toContain('RemovePayload(w "resources", i 1)')
    expect(result).toContain('VIProductVersion "1.2.3.36"')
    expect(result).toContain('!define MAGNITUDE_VERSION "1.2.3-alpha.1"')
    expect(result).not.toContain('Delete "$INSTDIR')
  })
  it("escapes both NSIS quoting contexts and interpolation", async () => {
    const result = await Effect.runPromise(renderWindowsInstaller(template, { ...input, files: ["resources/O'Brien$.txt"] }))
    expect(result).toContain("O$\\'Brien$$.txt")
    expect(result).not.toContain("O'Brien$.txt")
  })
  it.each(["01.2.3", "1.2.3\n!system bad", "v1.2.3", "65536.2.3"])("rejects invalid native release version %j", async version => {
    await expect(Effect.runPromise(renderWindowsInstaller(template, { ...input, version }))).rejects.toThrow()
  })
  it("refuses missing or duplicated template sections", async () => {
    await expect(Effect.runPromise(renderWindowsInstaller(template + "@REMOVE_FILES@", input))).rejects.toThrow()
    await expect(Effect.runPromise(renderWindowsInstaller("@PAYLOAD_FILES@", input))).rejects.toThrow()
  })
})


describe("Windows installer source boundary", () => {
  it.each(["redirected payload", "x64 helper"] as const)("rejects %s before compiler execution", async scenario => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const fixture = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-installer-input-" })
      const app = join(fixture, "app")
      yield* fs.makeDirectory(join(app, "resources"), { recursive: true })
      for (const name of ["Magnitude.exe", "resources/app.asar", "resources/magnitude-service.exe", "resources/desktop-host.node", "resources/magnitude-task-query.exe", "resources/Magnitude-LICENSE.txt"]) {
        yield* fs.writeFileString(join(app, name), "fixture")
      }
      const guard = join(fixture, "guard.dll")
      const bytes = new Uint8Array(96)
      const header = new DataView(bytes.buffer)
      header.setUint16(0, 0x5a4d, true)
      header.setUint32(60, 64, true)
      header.setUint32(64, 0x4550, true)
      header.setUint16(68, 0x8664, true)
      header.setUint16(86, 0x2000, true)
      yield* fs.writeFile(guard, bytes)
      if (scenario === "redirected payload") {
        const outside = join(fixture, "outside.txt")
        yield* fs.writeFileString(outside, "outside payload")
        yield* fs.symlink(outside, join(app, "redirected.txt"))
      }
      const output = join(fixture, "output")
      const result = yield* Effect.exit(buildWindowsDesktopInstaller({
        app, guard, makensis: join(fixture, "compiler-must-not-run"), output, version: "1.2.3", revision: 1,
      }))
      expect(Exit.isFailure(result)).toBe(true)
      if (Exit.isFailure(result)) expect(Cause.pretty(result.cause)).toContain(scenario === "redirected payload" ? "redirected path" : "x86 PE DLL")
      expect(yield* fs.exists(output)).toBe(false)
      if (scenario === "redirected payload") expect(yield* fs.readFileString(join(fixture, "outside.txt"))).toBe("outside payload")
    })).pipe(Effect.provide(NodeContext.layer)))
  })
})
