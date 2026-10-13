import { execFile } from "node:child_process"
import { randomBytes, randomUUID } from "node:crypto"
import { chmod, open, readFile, rename, rm } from "node:fs/promises"
import { join } from "node:path"
import { promisify } from "node:util"
import { Effect, Option, Redacted, Schema } from "effect"

/**
 * How loopback callers are admitted to application control. `Open` admits every loopback caller, as
 * on a personal machine; `Token` (a system service, which every local account can reach) admits only
 * callers presenting the token the owner issued for the current service start.
 */
export type LocalControl = "Open" | "Token"

export const LOCAL_CONTROL_TOKEN_FILE = "control-token"

export class LocalControlFailed extends Schema.TaggedError<LocalControlFailed>()("LocalControlFailed", { message: Schema.String }) {}
export class LocalControlDenied extends Schema.TaggedError<LocalControlDenied>()("LocalControlDenied", { message: Schema.String }) {}

const tokenPath = (stateDirectory: string) => join(stateDirectory, LOCAL_CONTROL_TOKEN_FILE)
const parseToken = (content: string) => {
  const token = content.trim()
  return token.length === 0 ? Option.none() : Option.some(Redacted.make(token))
}

/**
 * Issues a fresh token for one service start in the owner's state directory, replacing the previous
 * one atomically. Only the owner's account can read it, and with `groupAccess` also its group (the
 * server's `magnitude` group), which is what makes OS group membership the boundary for control.
 * `Open` removes any token an earlier service left behind.
 */
export const issueLocalControlToken = (options: {
  readonly stateDirectory: string; readonly localControl: LocalControl; readonly groupAccess: boolean
}) => Effect.tryPromise({
  try: async () => {
    const path = tokenPath(options.stateDirectory)
    if (options.localControl === "Open") {
      await rm(path, { force: true })
      return Option.none<Redacted.Redacted<string>>()
    }
    const token = randomBytes(32).toString("base64url")
    const mode = options.groupAccess ? 0o640 : 0o600
    const temporary = join(options.stateDirectory, `.${LOCAL_CONTROL_TOKEN_FILE}-${randomUUID()}.tmp`)
    const file = await open(temporary, "wx", mode)
    try {
      await file.writeFile(`${token}\n`)
    } finally {
      await file.close()
    }
    await chmod(temporary, mode)
    await rename(temporary, path)
    return Option.some(Redacted.make(token))
  },
  catch: error => new LocalControlFailed({ message: `Could not write the local control token: ${String(error)}` }),
})

const errorCode = (error: unknown) => typeof error === "object" && error !== null && "code" in error ? String(error.code) : undefined
const run = promisify(execFile)
const SAFE_PATH = /^\/[A-Za-z0-9._/-]+$/

const listedMember = (group: string, user: string) => Effect.tryPromise(() => run("/usr/bin/getent", ["group", group], { env: { PATH: "/usr/bin:/bin", LC_ALL: "C" } })).pipe(
  Effect.map(({ stdout }) => (stdout.trim().split(":")[3] ?? "").split(",").includes(user)),
  Effect.orElseSucceed(() => false))

/** `sg` grants a listed member the group for one command, without logging in again. */
const readAsMember = (group: string, path: string) => Effect.tryPromise(() =>
  run("/usr/bin/sg", [group, "-c", `/bin/cat -- ${path}`], { env: { PATH: "/usr/bin:/bin", LC_ALL: "C" }, timeout: 5_000 })).pipe(
  Effect.map(({ stdout }) => parseToken(stdout)))

/**
 * The token this account presents to the service on this machine, or none when the service admits
 * loopback callers openly. Where control is shared with `group`, an account listed in it but not
 * yet carrying it in this login (it was just added) reads the token through `sg`; anyone else is
 * denied.
 */
export const readLocalControlToken = (options: {
  readonly stateDirectory: string; readonly group: Option.Option<string>; readonly user: string
}) => Effect.gen(function* () {
  const path = tokenPath(options.stateDirectory)
  const direct = yield* Effect.tryPromise({ try: () => readFile(path, "utf8"), catch: errorCode }).pipe(Effect.either)
  if (direct._tag === "Right") return parseToken(direct.right)
  if (direct.left === "ENOENT") return Option.none<Redacted.Redacted<string>>()
  if (Option.isSome(options.group) && SAFE_PATH.test(path) && (yield* listedMember(options.group.value, options.user))) {
    const viaGroup = yield* readAsMember(options.group.value, path).pipe(Effect.option)
    if (Option.isSome(viaGroup)) return viaGroup.value
  }
  return yield* new LocalControlDenied({ message: Option.match(options.group, {
    onSome: group => `Magnitude runs as a server on this machine, and only members of the ${group} group can control it. `
      + `Ask an administrator to run \`sudo usermod -aG ${group} ${options.user}\`.`,
    onNone: () => `Could not read Magnitude's local control token at ${path}.`,
  }) })
})
