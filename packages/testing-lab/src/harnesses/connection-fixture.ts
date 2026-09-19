import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { dirname, join } from "node:path"
import { isDeepStrictEqual } from "node:util"
import { parse } from "yaml"
import { AssertionFailure, Digest, Harness, InfrastructureFailure } from "../domain"
import { DesktopDriver } from "../desktop-driver"
import { sha256 } from "../snapshot"

const fail = (message: string) => new AssertionFailure({ message })
const ObjectValue = Schema.Record({ key: Schema.String, value: Schema.Unknown })
const object = (value: unknown) => Schema.decodeUnknown(ObjectValue)(value).pipe(Effect.mapError(() => fail("Harness configuration is missing an expected object")))
export const ConnectionReceipt = Schema.Struct({ harness: Harness, endpoint: Schema.String, configurationDigest: Digest })
/** Seed only the isolated product-owned harness home; never write the developer's configuration. */
export const connectionFixture = (isolatedHome: string, harness: typeof Harness.Type, endpoint: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const file = join(isolatedHome, harness === "pi" ? ".pi/agent/models.json" : harness === "opencode" ? ".config/opencode/opencode.json" : ".hermes/config.yaml")
  const key = harness === "opencode" ? "provider" : "providers"
  const unrelated = harness === "pi" ? { baseUrl: "http://127.0.0.1:1/v1", api: "openai-completions", apiKey: "lab-unused", models: [] }
    : harness === "opencode" ? { npm: "@ai-sdk/openai-compatible", name: "Lab unrelated", options: { baseURL: "http://127.0.0.1:1/v1", apiKey: "lab-unused" }, models: {} }
    : { name: "Lab unrelated", base_url: "http://127.0.0.1:1/v1", api_key: "lab-unused", transport: "chat_completions" }
  if (yield* fs.exists(file)) return yield* fail("Connection fixture requires fresh isolated configuration")
  yield* fs.makeDirectory(dirname(file), { recursive: true, mode: 0o700 })
  yield* fs.writeFileString(file, yield* Schema.encode(Schema.parseJson(ObjectValue))({ [key]: { lab_unrelated: unrelated } }), { flag: "wx", mode: 0o600 })
  const inspect = (connected: boolean) => Effect.gen(function* () {
    const wire = yield* fs.readFileString(file)
    const document = yield* Effect.try({ try: () => parse(wire) as unknown, catch: () => fail("Harness configuration is not valid JSON/YAML") }).pipe(Effect.flatMap(object))
    const providers = yield* object(document[key])
    if (!isDeepStrictEqual(providers.lab_unrelated, unrelated)) return yield* fail(`${harness} connection clobbered an unrelated provider`)
    if (!connected) {
      if (Object.hasOwn(providers, "magnitude")) return yield* fail(`${harness} disconnect left its provider configuration behind`)
      return
    }
    const magnitude = yield* object(providers.magnitude)
    const actual = harness === "pi" ? magnitude.baseUrl : harness === "hermes" ? magnitude.base_url
      : (yield* object(magnitude.options)).baseURL
    if (actual !== endpoint) return yield* fail(`${harness} connection points at a different endpoint`)
  })
  const checkedInspect = (connected: boolean) => inspect(connected).pipe(Effect.mapError(error => error._tag === "AssertionFailure" ? error : new InfrastructureFailure({ operation: "connection-config", message: error.message })))
  const exercise = Effect.gen(function* () {
    const desktop = yield* DesktopDriver
    yield* desktop.connect(harness)
    yield* inspect(true)
    // The connected action is the product's refresh/sync action, addressed by the same stable ID.
    yield* desktop.connect(harness)
    yield* inspect(true)
    yield* desktop.disconnect(harness)
    yield* inspect(false)
    yield* desktop.connect(harness)
    yield* inspect(true)
    return ConnectionReceipt.make({ harness, endpoint, configurationDigest: sha256(yield* fs.readFile(file)) })
  })
  return { harness, exercise, inspect: checkedInspect }
})
