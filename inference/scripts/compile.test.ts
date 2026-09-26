import { hostById } from "@magnitudedev/release/targets"
import { Effect, Stream } from "effect"
import { resolve } from "node:path"
import { describe, expect, test } from "vitest"
import { cargoExecutables, InferenceBuildFailed, readCargoMessages, releaseRustflags } from "./compile"
import { installationEnvironment } from "./smoke"

const lines = (...values: readonly string[]) => Stream.fromIterable(values)

describe("inference compilation", () => {
  test("retains Cargo messages, forwards rendered diagnostics and finds the service executable", async () => {
    const forwarded: string[] = []
    const { messages, diagnostics } = await Effect.runPromise(readCargoMessages(
      lines(
        JSON.stringify({ reason: "compiler-message", message: { rendered: "warning: unused\n" } }),
        "",
        JSON.stringify({ reason: "compiler-artifact", target: { name: "magnitude_engine" }, executable: null }),
        JSON.stringify({ reason: "compiler-artifact", target: { name: "magnitude-inference" }, executable: "/t/magnitude-inference" }),
        JSON.stringify({ reason: "build-finished", success: true }),
      ),
      (rendered) => Effect.sync(() => { forwarded.push(rendered) }),
    ))
    expect(forwarded).toEqual(["warning: unused\n"])
    expect(diagnostics).toEqual(["warning: unused\n"])
    expect(cargoExecutables(messages, "magnitude-inference")).toEqual(["/t/magnitude-inference"])
    expect(cargoExecutables(messages, "magnitude_engine")).toEqual([])
  })

  test("rejects a malformed Cargo message", async () => {
    const error = await Effect.runPromise(readCargoMessages(lines("{not json"), () => Effect.void).pipe(Effect.flip))
    expect(error).toBeInstanceOf(InferenceBuildFailed)
  })

  test("release builds fix rustflags: baseline CPU everywhere and the ../runtime rpath on Linux", () => {
    expect(releaseRustflags(hostById("linux-x64-gnu"))).toBe("-Clink-arg=-Wl,-rpath,$ORIGIN/../runtime")
    expect(releaseRustflags(hostById("linux-arm64-gnu"))).toBe("-Clink-arg=-Wl,-rpath,$ORIGIN/../runtime")
    for (const host of ["darwin-arm64", "darwin-x64", "windows-x64-msvc"] as const) {
      expect(releaseRustflags(hostById(host))).toBe("")
    }
  })
})

describe("installation environment", () => {
  test("clears inherited Unix loader paths and prepends runtime/ to the Windows PATH", () => {
    expect(installationEnvironment("/i", "linux").LD_LIBRARY_PATH).toBe("")
    expect(installationEnvironment("/i", "darwin").DYLD_LIBRARY_PATH).toBe("")
    expect(installationEnvironment("/i", "win32").PATH?.startsWith(resolve("/i", "runtime"))).toBe(true)
  })
})
