import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { parse, stringify } from "yaml"
import { expect, test } from "vitest"
import { connectionFixture } from "../src/harnesses/connection-fixture"

for (const harness of ["pi", "opencode", "hermes"] as const) test(`${harness} checks unrelated config, endpoint and removal`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-connection-" })
  const endpoint = "http://127.0.0.1:11319/inference/v1"
  const fixture = yield* connectionFixture(root, harness, endpoint)
  const file = join(root, harness === "pi" ? ".pi/agent/models.json" : harness === "opencode" ? ".config/opencode/opencode.json" : ".hermes/config.yaml")
  const key = harness === "opencode" ? "provider" : "providers"
  const document = parse(yield* fs.readFileString(file))
  const write = () => fs.writeFileString(file, stringify(document))
  yield* fixture.inspect(false)
  const absent = yield* fixture.inspect(true).pipe(Effect.either)
  expect(absent._tag === "Left" && absent.left._tag).toBe("AssertionFailure")
  document[key].magnitude = harness === "pi" ? { baseUrl: endpoint } : harness === "hermes" ? { base_url: endpoint } : { options: { baseURL: endpoint } }
  yield* write()
  yield* fixture.inspect(true)
  expect((yield* fixture.inspect(false).pipe(Effect.either))._tag).toBe("Left")
  const saved = document[key].lab_unrelated
  delete document[key].lab_unrelated
  yield* write()
  expect((yield* fixture.inspect(true).pipe(Effect.either))._tag).toBe("Left")
  document[key].lab_unrelated = saved
  document[key].magnitude = { baseUrl: "http://wrong.invalid", base_url: "http://wrong.invalid", options: { baseURL: "http://wrong.invalid" } }
  yield* write()
  expect((yield* fixture.inspect(true).pipe(Effect.either))._tag).toBe("Left")
  expect((yield* connectionFixture(root, harness, endpoint).pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))
