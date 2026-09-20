import { FileSystem } from "@effect/platform"
import { Config, Context, Effect, Layer, Option, Schema } from "effect"
import { isAbsolute, join } from "node:path"
import { AssertionFailure, Harness, InfrastructureFailure } from "../domain"
import { command } from "../process"
import { fileFixture } from "./file-fixture"
import { piSession } from "./pi"
import { openCode } from "./opencode"
import { hermes } from "./hermes"
import { hermesInstallation } from "./installation"

export interface HarnessTools { readonly executable: (harness: Harness) => Effect.Effect<string, InfrastructureFailure> }
export const HarnessTools = Context.GenericTag<HarnessTools>("@magnitudedev/testing-lab/HarnessTools")
export const configuredHarnessTools = Layer.effect(HarnessTools, Effect.gen(function* () {
  const paths = new Map<Harness, string>()
  for (const harness of ["pi", "opencode", "hermes"] as const) {
    const path = yield* Config.option(Config.string(`LAB_${harness.toUpperCase()}_EXECUTABLE`))
    if (Option.isSome(path) && isAbsolute(path.value)) paths.set(harness, path.value)
  }
  return { executable: harness => Effect.fromNullable(paths.get(harness)).pipe(Effect.mapError(() =>
    new InfrastructureFailure({ operation: "harness-tools", message: `Worker requires an explicit absolute LAB_${harness.toUpperCase()}_EXECUTABLE path` }))) } satisfies HarnessTools
}))
export const HarnessTurn = Schema.Struct({ sessionId: Schema.NonEmptyString, text: Schema.NonEmptyString,
  streamed: Schema.Boolean, tools: Schema.Array(Schema.String) })
const fail = (message: string) => new AssertionFailure({ message })
const versions = { pi: "0.85.1", opencode: "1.18.31", hermes: hermesInstallation.version } as const
export const harnessSuite = (harness: Harness, model: string, root: string, home: string, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executable = yield* (yield* HarnessTools).executable(harness)
  const evidence = join(root, "events")
  yield* fs.makeDirectory(evidence, { recursive: true })
  const env = { ...environment, HOME: home, USERPROFILE: home, XDG_CONFIG_HOME: join(home, ".config"),
    XDG_DATA_HOME: join(home, ".local", "share"), XDG_CACHE_HOME: join(home, ".cache"), XDG_STATE_HOME: join(home, ".local", "state"),
    PI_CODING_AGENT_DIR: join(home, ".pi", "agent"), HERMES_HOME: join(home, ".hermes") }
  const version = yield* command(executable, ["--version"], { env, inheritEnv: false, timeoutMs: 30_000 })
  if (version.exitCode !== 0 || !new RegExp(`(?:^|[^0-9.])${versions[harness].replaceAll(".", "\\.")}(?:$|[^0-9.])`).test(version.stdout)) {
    return yield* new InfrastructureFailure({ operation: "harness-tools", message: `Expected pinned ${harness} ${versions[harness]}; worker tool is missing or differs` })
  }
  yield* fs.writeFileString(join(evidence, "version.txt"), version.stdout)
  const controls: Record<string, string> = harness === "opencode" ? { "opencode.json": yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ $schema: "https://opencode.ai/config.json", permission: { "*": "deny", read: "allow", edit: "allow" } }) } : {}
  const fixture = yield* fileFixture(root, controls)
  const config = { executable, cwd: fixture.directory, environment: env, evidence, model }
  const rpc = harness === "opencode" ? yield* openCode(config) : harness === "hermes" ? yield* hermes(config) : undefined
  let turnIndex = 0
  const prompt = (message: string, previous: Option.Option<string>) => Effect.scoped(Effect.gen(function* () {
    if (harness !== "pi") {
      const turn = yield* rpc!.prompt(message, previous)
      return HarnessTurn.make({ ...turn, streamed: "streamed" in turn && turn.streamed === true })
    }
    const label = `turn-${++turnIndex}`
    const session = yield* piSession({ executable, args: ["--mode", "rpc", "--provider", "magnitude", "--model", model, "--thinking", "off", "--tools", "read,write,edit", "--offline", "--session", join(root, "session.jsonl")],
      cwd: fixture.directory, environment: env, stdoutLog: join(evidence, `${label}.jsonl`), stderrLog: join(evidence, `${label}.stderr.log`) }, model)
    const selected = yield* session.state()
    if (Option.exists(previous, id => id !== selected.sessionId)) return yield* fail("Pi resumed a different persisted session")
    const turn = yield* session.prompt(message)
    return HarnessTurn.make({ ...turn, tools: turn.tools.map(tool => tool.name) })
  }))
  const token = `lab-${crypto.randomUUID()}`
  const initial = yield* Effect.cached(prompt(`Remember this identifier for our conversation: ${token}. Briefly acknowledge it. For this reply only, do not invoke tools.`, Option.none()))
  const recall = Effect.gen(function* () {
    const first = yield* initial
    const turn = yield* prompt("Repeat the identifier I gave you earlier. For this reply only, do not invoke tools.", Option.some(first.sessionId))
    if (!turn.text.includes(token)) return yield* fail(`${harness} did not retain the conversation reference code`)
    return turn
  })
  const tools = Effect.gen(function* () {
    const first = yield* initial
    const read = harness === "hermes" ? "read_file" : "read", edit = harness === "hermes" ? "patch" : "edit"
    const file = join(fixture.directory, "message.txt")
    const quotedFile = yield* Schema.encode(Schema.parseJson(Schema.String))(file)
    const replacement = harness === "pi" ? 'oldText="before", newText="after"'
      : harness === "opencode" ? 'oldString="before", newString="after"' : 'old_string="before", new_string="after"'
    const turn = yield* prompt(`The existing fixture file is ${quotedFile}. Use the ${read} tool to read that absolute path, then execute the ${edit} tool on the same path with ${replacement}. Line numbers and JSON fields displayed by the read tool are metadata, not file contents; do not include them in the replacement. Preserve the final newline. Do not change any other file. You must execute the tools, not describe the steps. Reply DONE only after the edit succeeds.`, Option.some(first.sessionId))
    if (!turn.tools.includes(read) || !turn.tools.includes(edit)) return yield* fail(`${harness} omitted the required read/edit tools`)
    yield* fixture.verify
    return turn
  })
  return { initial, recall, tools }
})
