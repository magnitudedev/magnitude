import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { randomUUID } from "node:crypto"
import { dirname, join } from "node:path"
import { SERVER_DATA_DIRECTORY, SERVER_MARKER_CONTENT, SERVER_MARKER_PATH, SERVER_USER } from "./server-profile"

export class ServerInstallationFailed extends Schema.TaggedError<ServerInstallationFailed>()("ServerInstallationFailed", {
  message: Schema.String,
}) {}
const failed = (message: string) => new ServerInstallationFailed({ message })

/** The installed CLI each platform's root step must run from; sudo invokes it by this exact path. */
export const installedServerCli = (platform: string) => Option.fromNullable(({
  linux: "/usr/lib/magnitude-desktop/resources/magnitude",
  darwin: "/Applications/Magnitude.app/Contents/Resources/magnitude",
} as Record<string, string>)[platform])

/** A POSIX login name; never root or the service account itself. */
const UserName = Schema.String.pipe(Schema.pattern(/^[a-z_][a-z0-9_.-]{0,31}$/), Schema.filter(name => name !== "root" && name !== SERVER_USER))

/**
 * The hidden root commands accept exactly `_server-install <user>`, `_server-install` or `_server-remove`.
 * The user must be the account that ran sudo, so the root step cannot be pointed at anyone else. Without
 * a user it is root's own setup, run directly from a root login rather than through sudo.
 */
export const parseServerInstallation = (argv: readonly string[], environment: Readonly<Record<string, string | undefined>>) => Effect.gen(function* () {
  const [command, ...rest] = argv
  if (command === "_server-remove" && rest.length === 0) return { _tag: "Remove" as const }
  if (command === "_server-install" && rest.length === 0) {
    if (environment.SUDO_USER !== undefined || environment.SUDO_UID !== undefined) return yield* failed("Run `magnitude server setup` as yourself; it asks sudo for the root step.")
    return { _tag: "Install" as const, user: Option.none<{ readonly name: string; readonly uid: number }>() }
  }
  if (command !== "_server-install" || rest.length !== 1) return yield* failed("Usage: magnitude server setup")
  const user = yield* Schema.decodeUnknown(UserName)(rest[0]).pipe(Effect.mapError(() => failed("The server user name is invalid.")))
  if (environment.SUDO_USER !== user || !/^[1-9][0-9]*$/.test(environment.SUDO_UID ?? "")) {
    return yield* failed("Run `magnitude server setup` as yourself; it asks sudo for the root step.")
  }
  return { _tag: "Install" as const, user: Option.some({ name: user, uid: Number(environment.SUDO_UID) }) }
})

/** Refuses unless the process is root and is the installed CLI, reached without symbolic links. */
export const requireInstalledRoot = (platform: string, executable: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const expected = installedServerCli(platform)
  if (Option.isNone(expected) || process.getuid?.() !== 0) return yield* failed("Run `magnitude server setup`; this step only runs as root from the installed Magnitude.")
  const real = yield* fs.realPath(executable).pipe(Effect.orElseSucceed(() => ""))
  if (real !== expected.value) return yield* failed("Run `magnitude server setup`; this step only runs as root from the installed Magnitude.")
  return expected.value
})

const run = (description: string, command: string, ...args: string[]) => Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const code = yield* executor.exitCode(Command.make(command, ...args).pipe(
    Command.env({ PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }), Command.stdout("inherit"), Command.stderr("inherit"),
  )).pipe(Effect.orElseSucceed(() => -1))
  if (code !== 0) return yield* failed(`Could not ${description}.`)
})
const succeeds = (command: string, ...args: string[]) => Effect.flatMap(CommandExecutor.CommandExecutor, executor =>
  executor.exitCode(Command.make(command, ...args).pipe(Command.env({ PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }))).pipe(
    Effect.map(code => code === 0), Effect.orElseSucceed(() => false)))
const commandOutput = (command: string, ...args: string[]) => Effect.flatMap(CommandExecutor.CommandExecutor, executor =>
  executor.string(Command.make(command, ...args).pipe(Command.env({ PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }))))

/** A root-owned directory reached without symbolic links; created when absent. */
const rootDirectory = (path: string, mode: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!(yield* fs.exists(path))) yield* fs.makeDirectory(path, { mode })
  const info = yield* fs.stat(path)
  if (info.type !== "Directory" || Option.getOrUndefined(info.uid) !== 0 || (yield* fs.realPath(path)) !== path) {
    return yield* failed(`${path} must be a directory owned by root.`)
  }
  yield* fs.chmod(path, mode)
})

/** Our fixed root-owned file, replaced atomically so it is never observed partially written. */
const writeRootFile = (path: string, content: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = join(dirname(path), `.${randomUUID()}.tmp`)
  yield* fs.writeFileString(temporary, content, { flag: "wx", mode: 0o644 })
  yield* fs.chmod(temporary, 0o644)
  yield* fs.rename(temporary, path)
})

const systemctl = "/usr/bin/systemctl"

/** Linux root step: the service account, its data directory, group membership, marker and unit. */
export const installLinuxServer = (user: Option.Option<{ readonly name: string; readonly uid: number }>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!(yield* fs.exists("/run/systemd/system"))) return yield* failed("Server setup needs systemd. Run `magnitude serve` instead.")
  if (Option.isSome(user)) {
    const userUid = (yield* commandOutput("/usr/bin/id", "-u", "--", user.value.name).pipe(Effect.orElseSucceed(() => ""))).trim()
    if (userUid !== String(user.value.uid)) return yield* failed("The account that ran sudo could not be identified.")
  }
  if (!(yield* succeeds("/usr/bin/getent", "group", SERVER_USER))) yield* run("create the magnitude group", "/usr/sbin/groupadd", "--system", SERVER_USER)
  if (!(yield* succeeds("/usr/bin/getent", "passwd", SERVER_USER))) {
    yield* run("create the magnitude user", "/usr/sbin/useradd", "--system", "--gid", SERVER_USER, "--home-dir", SERVER_DATA_DIRECTORY,
      "--no-create-home", "--shell", "/usr/sbin/nologin", "--comment", "Magnitude server", SERVER_USER)
  }
  const serviceUid = (yield* commandOutput("/usr/bin/id", "-u", "--", SERVER_USER)).trim()
  // A kept data directory from an earlier setup may belong to a previous magnitude account.
  if (!(yield* fs.exists(SERVER_DATA_DIRECTORY))) yield* fs.makeDirectory(SERVER_DATA_DIRECTORY, { mode: 0o750 })
  const data = yield* fs.stat(SERVER_DATA_DIRECTORY)
  if (data.type !== "Directory" || (yield* fs.realPath(SERVER_DATA_DIRECTORY)) !== SERVER_DATA_DIRECTORY) {
    return yield* failed(`${SERVER_DATA_DIRECTORY} must be a directory.`)
  }
  if (String(Option.getOrUndefined(data.uid)) !== serviceUid) {
    yield* run(`give ${SERVER_DATA_DIRECTORY} to the magnitude user`, "/usr/bin/chown", "-R", "-h", "--", `${SERVER_USER}:${SERVER_USER}`, SERVER_DATA_DIRECTORY)
  }
  yield* fs.chmod(SERVER_DATA_DIRECTORY, 0o750)
  if (Option.isSome(user)) yield* run(`add ${user.value.name} to the magnitude group`, "/usr/sbin/usermod", "-aG", SERVER_USER, "--", user.value.name)
  yield* rootDirectory(dirname(SERVER_MARKER_PATH), 0o755)
  yield* writeRootFile(SERVER_MARKER_PATH, SERVER_MARKER_CONTENT)
  yield* run("reload systemd", systemctl, "daemon-reload")
  yield* run("start magnitude.service", systemctl, "enable", "--now", "magnitude.service")
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server setup could not finish.")))

/** Linux root step: stops the service and removes the account; the data directory is kept. */
export const removeLinuxServer = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (yield* fs.exists("/run/systemd/system")) {
    // The unit may already be gone with the package; disabling it then has nothing to do.
    if (yield* succeeds(systemctl, "cat", "magnitude.service")) yield* run("stop magnitude.service", systemctl, "disable", "--now", "magnitude.service")
  }
  yield* fs.remove(SERVER_MARKER_PATH, { force: true })
  const markerDirectory = dirname(SERVER_MARKER_PATH)
  if ((yield* fs.exists(markerDirectory)) && (yield* fs.readDirectory(markerDirectory)).length === 0) yield* fs.remove(markerDirectory)
  if (yield* succeeds("/usr/bin/getent", "passwd", SERVER_USER)) yield* run("remove the magnitude user", "/usr/sbin/userdel", SERVER_USER)
  if (yield* succeeds("/usr/bin/getent", "group", SERVER_USER)) yield* run("remove the magnitude group", "/usr/sbin/groupdel", SERVER_USER)
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server removal could not finish.")))
