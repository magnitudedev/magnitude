import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { isServerProfileActive, SERVER_MARKER_PATH } from "./server-profile"

type Entry = { readonly type: "File" | "Directory"; readonly uid: number; readonly mode: number; readonly real?: string }
const run = (entries: Record<string, Entry>, platform = "linux") => Effect.runPromise(isServerProfileActive(platform).pipe(
  Effect.provideService(FileSystem.FileSystem, FileSystem.makeNoop({
    stat: path => entries[path] ? Effect.succeed({ type: entries[path]!.type, uid: Option.some(entries[path]!.uid), mode: entries[path]!.mode } as FileSystem.File.Info)
      : Effect.fail(new (class extends Error { _tag = "SystemError" as const })() as never),
    realPath: path => Effect.succeed(entries[path]?.real ?? path),
  }))))
const valid = { [SERVER_MARKER_PATH]: { type: "File", uid: 0, mode: 0o100644 }, "/etc/magnitude": { type: "Directory", uid: 0, mode: 0o40755 } } as const

describe("server profile marker", () => {
  it("is active only for a root-owned marker in a root-owned directory", async () => {
    expect(await run(valid)).toBe(true)
  })
  it("is never active outside Linux", async () => {
    expect(await run(valid, "darwin")).toBe(false)
  })
  it.each([
    ["absent", {}],
    ["owned by another user", { ...valid, [SERVER_MARKER_PATH]: { type: "File", uid: 1000, mode: 0o100644 } }],
    ["group-writable", { ...valid, [SERVER_MARKER_PATH]: { type: "File", uid: 0, mode: 0o100664 } }],
    ["a directory", { ...valid, [SERVER_MARKER_PATH]: { type: "Directory", uid: 0, mode: 0o40755 } }],
    ["reached through a symbolic link", { ...valid, [SERVER_MARKER_PATH]: { type: "File", uid: 0, mode: 0o100644, real: "/tmp/marker" } }],
    ["in a writable directory", { ...valid, "/etc/magnitude": { type: "Directory", uid: 0, mode: 0o41777 } }],
    ["in a foreign-owned directory", { ...valid, "/etc/magnitude": { type: "Directory", uid: 1000, mode: 0o40755 } }],
  ] as const)("is inactive when the marker is %s", async (_, entries) => {
    expect(await run(entries as Record<string, Entry>)).toBe(false)
  })
})
