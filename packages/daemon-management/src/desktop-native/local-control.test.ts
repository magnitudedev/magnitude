import { chmod, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option, Redacted } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { LOCAL_CONTROL_TOKEN_FILE, issueLocalControlToken, readLocalControlToken } from "./local-control"

let directory: string
beforeEach(async () => { directory = await mkdtemp(join(tmpdir(), "magnitude-local-control-")) })
afterEach(async () => {
  await chmod(directory, 0o700).catch(() => undefined)
  await rm(directory, { recursive: true, force: true })
})

const read = (group = Option.none<string>()) => Effect.runPromise(readLocalControlToken({ stateDirectory: directory, group, user: "nobody-here" }).pipe(
  Effect.map(token => Option.getOrNull(Option.map(token, Redacted.value))), Effect.either, Effect.map(result =>
    result._tag === "Right" ? { token: result.right, denied: undefined } : { denied: result.left })))

describe.skipIf(process.platform === "win32")("local control token", () => {
  it("issues a fresh token for each service start, readable only by the owner or also by its group", async () => {
    const first = await Effect.runPromise(issueLocalControlToken({ stateDirectory: directory, localControl: "Token", groupAccess: false }))
    const path = join(directory, LOCAL_CONTROL_TOKEN_FILE)
    expect(((await stat(path)).mode & 0o777).toString(8)).toBe("600")
    expect(Option.map(first, Redacted.value)).toEqual(Option.some((await readFile(path, "utf8")).trim()))
    expect(await read()).toEqual({ token: Redacted.value(Option.getOrThrow(first)) })
    const second = await Effect.runPromise(issueLocalControlToken({ stateDirectory: directory, localControl: "Token", groupAccess: true }))
    expect(((await stat(path)).mode & 0o777).toString(8)).toBe("640")
    expect(Option.getOrThrow(second).pipe(Redacted.value)).not.toBe(Option.getOrThrow(first).pipe(Redacted.value))
  })

  it("leaves no token where loopback control is open, removing one an earlier service left", async () => {
    await writeFile(join(directory, LOCAL_CONTROL_TOKEN_FILE), "stale\n")
    expect(Option.isNone(await Effect.runPromise(issueLocalControlToken({ stateDirectory: directory, localControl: "Open", groupAccess: false })))).toBe(true)
    expect(await read()).toEqual({ token: null })
  })

  it.skipIf(process.getuid?.() === 0)("denies an account that cannot read the token and is not listed in the group", async () => {
    await Effect.runPromise(issueLocalControlToken({ stateDirectory: directory, localControl: "Token", groupAccess: true }))
    await chmod(directory, 0o000)
    const result = await read(Option.some("magnitude"))
    expect(result.denied?._tag).toBe("LocalControlDenied")
    expect(result.denied?.message).toContain("sudo usermod -aG magnitude nobody-here")
  })
})
