import { Command, CommandExecutor } from "@effect/platform"
import { GuardedCommand } from "@magnitudedev/daemon-management/desktop-native"
import { Effect, HashMap, Option, Schema } from "effect"
import { randomUUID } from "node:crypto"
import { userInfo } from "node:os"

const Environment = Schema.Record({ key: Schema.String, value: Schema.String })
type Environment = typeof Environment.Type
class ShellEnvironmentFailed extends Schema.TaggedError<ShellEnvironmentFailed>()("ShellEnvironmentFailed", {}) {}

const probe = (shell: string, flags: string, environment: Environment, timeout: number) => Effect.gen(function* () {
  const runner = yield* GuardedCommand
  const marker = `__MAGNITUDE_ENV_${randomUUID()}__`
  const { code, stdout: output } = yield* runner.run(shell, [flags, `printf '${marker}\\n'; /usr/bin/env -0; printf '${marker}\\n'`], {
    ...environment, ELECTRON_RUN_AS_NODE: "1", MAGNITUDE_RESOLVING_ENV: "1",
  }).pipe(Effect.timeout(timeout))
  const start = output.indexOf(`${marker}\n`)
  const end = output.lastIndexOf(marker)
  if (code !== 0 || start < 0 || end <= start) return yield* new ShellEnvironmentFailed()
  const entries = output.slice(start + marker.length + 1, end).split("\0").filter(Boolean).map(entry => {
    const separator = entry.indexOf("=")
    return [entry.slice(0, separator), entry.slice(separator + 1)] as const
  })
  if (entries.some(([key]) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key))) return yield* new ShellEnvironmentFailed()
  return yield* Schema.decodeUnknown(Environment)(Object.fromEntries(entries.filter(([key]) => !key.startsWith("ELECTRON_") && key !== "MAGNITUDE_RESOLVING_ENV")))
})

/** A bounded, application-scoped observation for harnesses; never mutates process.env. */
export const resolveHarnessEnvironment = (options: {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly platform?: NodeJS.Platform
  readonly timeoutMilliseconds?: number
} = {}) => Effect.gen(function* () {
  const environment = Object.fromEntries(Object.entries(options.environment ?? process.env).filter((entry): entry is [string, string] => entry[1] !== undefined))
  const platform = options.platform ?? process.platform
  if (platform === "win32" || environment.MAGNITUDE_SHELL_ENV_INHERITED) return environment
  const shell = environment.SHELL ?? (yield* Effect.try(() => userInfo().shell).pipe(Effect.option, Effect.map(Option.getOrUndefined))) ?? (platform === "darwin" ? "/bin/zsh" : "/bin/bash")
  if (/[/\\]nu(?:\.exe)?$/.test(shell)) return environment
  const found = yield* probe(shell, "-ilc", environment, options.timeoutMilliseconds ?? 2500).pipe(
    Effect.orElse(() => probe(shell, "-lc", environment, options.timeoutMilliseconds ?? 2500)), Effect.option,
  )
  if (Option.isNone(found)) return environment
  // Preserve explicit launch overrides; login PATH supplies tools absent from GUI-launch PATH.
  return { ...found.value, ...environment, PATH: found.value.PATH ?? environment.PATH ?? "" }
})

/** Child commands inherit the resolved harness environment; command-specific overrides win. */
export const harnessCommandExecutor = (environment: Environment) => Effect.map(CommandExecutor.CommandExecutor, executor => {
  const apply = (command: Command.Command): Command.Command => command._tag === "PipedCommand"
    ? Command.pipeTo(apply(command.left), apply(command.right))
    : Command.env(command, { ...environment, ...Object.fromEntries(HashMap.toEntries(command.env)) })
  return CommandExecutor.makeExecutor(command => executor.start(apply(command)))
})
