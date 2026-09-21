import { expect, test } from "vitest"
import { Effect } from "effect"
import { parseArguments, selectionFromOptions } from "../src/cli"
test("keeps source paths literal and enables Spark only by explicit flag", async () => {
  const args = await Effect.runPromise(parseArguments(["run", "--source", "/tmp/source with spaces", "--target", "ubuntu-24.04-x64-cpu-intel", "--allow-spark"]))
  expect(args.options.get("source")).toBe("/tmp/source with spaces")
  expect(args.options.has("allow-spark")).toBe(true)
  expect((await Effect.runPromise(parseArguments(["run", "--source", "."]))).options.has("allow-spark")).toBe(false)
})

test("custom selections preserve explicit targets, suites and harnesses", async () => {
  const selected = await Effect.runPromise(selectionFromOptions(new Map([["target", "ubuntu-24.04-x64-cpu-intel,windows-server-2025-x64-cuda-a10"], ["suite", "harness, cli"], ["harness", "pi,hermes"]])))
  expect(selected).toMatchObject({ kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel", "windows-server-2025-x64-cuda-a10"], suites: ["harness", "cli"], harnesses: ["pi", "hermes"] })
})
test("quick accepts a harness override without becoming a custom suite selection", async () => {
  const selected = await Effect.runPromise(selectionFromOptions(new Map([["profile", "quick"], ["harness", "opencode,hermes"]])))
  expect(selected).toMatchObject({ kind: "profile", profile: "quick", harnesses: { _tag: "Some", value: ["opencode", "hermes"] } })
})
test.each([
  [["suite", "app"]],
  [["suite", "app"], ["target", "ubuntu-24.04-x64-cpu-intel"], ["profile", "pr"]],
  [["suite", "app,app"], ["target", "ubuntu-24.04-x64-cpu-intel"]],
  [["suite", "app,"], ["target", "ubuntu-24.04-x64-cpu-intel"]],
  [["harness", "pi"], ["profile", "full"]],
].map(options => ({ options })))("rejects ambiguous or incomplete custom selections $options", async ({ options }) => {
  expect((await Effect.runPromise(selectionFromOptions(new Map(options as [string, string][])).pipe(Effect.either)))._tag).toBe("Left")
})
test("report files require a completed result", async () => {
  for (const args of [["run", "--source", ".", "--no-wait", "--junit", "out.xml"], ["status", "--json", "out.json"]]) {
    expect((await Effect.runPromise(parseArguments(args).pipe(Effect.either)))._tag).toBe("Left")
  }
  expect((await Effect.runPromise(parseArguments(["results", "--run", "run-id", "--junit", "out.xml"]))).options.get("junit")).toBe("out.xml")
})
test.each([["run", "--source"], ["run", "--source", "--target", "x"], ["run", "--budget", "10", "--budget", "20"], ["run", "--allow-sparkk"]])("rejects missing, duplicate and unknown arguments", async (...args) => {
  expect((await Effect.runPromise(parseArguments(args).pipe(Effect.either)))._tag).toBe("Left")
})

test("requires exactly one source or artifact input", async () => {
  const parsed = await Effect.runPromise(parseArguments(["run", "--artifacts", "/tmp/private release/release.json"]))
  expect(parsed.options.get("artifacts")).toBe("/tmp/private release/release.json")
  for (const args of [["run"], ["run", "--source", ".", "--artifacts", "release.json"]]) {
    expect((await Effect.runPromise(parseArguments(args).pipe(Effect.either)))._tag).toBe("Left")
  }
})

test("rejects unfinished historical migration before snapshotting or uploading inputs", async () => {
  expect((await Effect.runPromise(parseArguments(["status", "--update-from", "old.json"]).pipe(Effect.either)))._tag).toBe("Left")
  for (const input of ["--source", "--artifacts"]) {
    expect(await Effect.runPromise(parseArguments(["run", input, "candidate", "--update-from", "/tmp/old release/release.json"]).pipe(Effect.either))).toMatchObject({ _tag: "Left", left: { message: expect.stringContaining("not implemented") } })
  }
})

test("rejects unsupported execution modes before source preparation", async () => {
  expect((await Effect.runPromise(parseArguments(["run", "--source", ".", "--mode", "verify"]))).options.get("mode")).toBe("verify")
  for (const mode of ["iterate", "unknown"]) expect(await Effect.runPromise(parseArguments(["run", "--source", ".", "--mode", mode]).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
})
