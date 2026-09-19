import { expect, test } from "vitest"
import { Effect } from "effect"
import { parseArguments } from "../src/cli"
test("keeps source paths literal and enables Spark only by explicit flag", async () => {
  const args = await Effect.runPromise(parseArguments(["run", "--source", "/tmp/source with spaces", "--target", "ubuntu-24.04-x64-cpu-intel", "--allow-spark"]))
  expect(args.options.get("source")).toBe("/tmp/source with spaces")
  expect(args.options.has("allow-spark")).toBe(true)
  expect((await Effect.runPromise(parseArguments(["run", "--source", "."]))).options.has("allow-spark")).toBe(false)
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
