import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Console, Effect, Either, Option, Schedule, Schema, Stream } from "effect"
import { randomUUID } from "node:crypto"
import { dirname, join } from "node:path"
import { MAC_SERVER_LABEL, MAC_SERVER_PLIST, SERVER_DATA_DIRECTORY, SERVER_MARKER_CONTENT, SERVER_MARKER_PATH, SERVER_SUDOERS_PATH, SERVER_SUDOERS_RULE,
  SERVER_USER, isServerProfileActive } from "./server-profile"

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
 * The hidden root commands accept exactly `_server-install <user>`, `_server-install`, `_server-remove` or
 * `_server-restore-grant`, which package installation runs.
 * The user must be the account that ran sudo, so the root step cannot be pointed at anyone else. Without
 * a user it is root's own setup, run directly from a root login rather than through sudo.
 */
export const parseServerInstallation = (argv: readonly string[], environment: Readonly<Record<string, string | undefined>>) => Effect.gen(function* () {
  const [command, ...rest] = argv
  if (command === "_server-remove" && rest.length === 0) return { _tag: "Remove" as const }
  if (command === "_server-restore-grant" && rest.length === 0) return { _tag: "RestoreGrant" as const }
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

type Services = FileSystem.FileSystem | CommandExecutor.CommandExecutor
const environment = { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }
const text = <E>(stream: Stream.Stream<Uint8Array, E>) => stream.pipe(Stream.decodeText(), Stream.runFold("", (all, chunk) => all + chunk))

/** Runs a command quietly: its output appears only in the failure, so a tool's routine notes never mislead. */
const run = (description: string, command: string, ...args: string[]) => Effect.scoped(Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const child = yield* executor.start(Command.make(command, ...args).pipe(Command.env(environment)))
  const [code, stdout, stderr] = yield* Effect.all([child.exitCode, text(child.stdout), text(child.stderr)], { concurrency: "unbounded" })
  return { code: Number(code), output: `${stdout}${stderr}`.trim() }
})).pipe(
  Effect.orElseSucceed(() => ({ code: -1, output: "" })),
  Effect.flatMap(({ code, output }) => code === 0 ? Effect.void : Effect.fail(failed(`Could not ${description}.${output === "" ? "" : `\n${output}`}`))))
const succeeds = (command: string, ...args: string[]) => Effect.flatMap(CommandExecutor.CommandExecutor, executor =>
  executor.exitCode(Command.make(command, ...args).pipe(Command.env(environment))).pipe(
    Effect.map(code => code === 0), Effect.orElseSucceed(() => false)))
const commandOutput = (command: string, ...args: string[]) => Effect.flatMap(CommandExecutor.CommandExecutor, executor =>
  executor.string(Command.make(command, ...args).pipe(Command.env(environment))))
const describe = (error: unknown) => error instanceof ServerInstallationFailed ? error.message : "Server setup could not finish."

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

export type ServerAccountState = "Absent" | "Ours" | "Foreign"
export interface ServerAccounts {
  readonly user: ServerAccountState
  readonly group: ServerAccountState
}

const loginDefinition = (definitions: string, key: string) => {
  const match = new RegExp(`^\\s*${key}\\s+(\\d+)\\s*$`, "m").exec(definitions)
  return match ? Number(match[1]) : undefined
}

/** The range `useradd --system` and `groupadd --system` allocate from: /etc/login.defs, else shadow's defaults. */
export const systemIdRange = (definitions: string, kind: "UID" | "GID") => ({
  min: loginDefinition(definitions, `SYS_${kind}_MIN`) ?? 101,
  max: loginDefinition(definitions, `SYS_${kind}_MAX`) ?? (loginDefinition(definitions, `${kind}_MIN`) ?? 1000) - 1,
})

const noLoginShells = new Set(["/usr/sbin/nologin", "/sbin/nologin", "/usr/bin/nologin", "/bin/false", "/usr/bin/false"])

/**
 * Only an account setup itself could have made is the service account: a system-range uid with the
 * system-range magnitude group as its primary group, home /var/lib/magnitude and no login shell.
 * Anything else named magnitude, such as a person's account, is never reused, granted or deleted.
 */
export const classifyServerAccounts = (entries: { readonly passwd: string; readonly group: string; readonly loginDefinitions: string }): ServerAccounts => {
  const within = (value: number, range: { readonly min: number; readonly max: number }) => Number.isInteger(value) && value >= range.min && value <= range.max
  const groupFields = entries.group.trim().split(":")
  const gid = Number(groupFields[2])
  const group: ServerAccountState = entries.group.trim() === "" ? "Absent"
    : groupFields.length === 4 && groupFields[0] === SERVER_USER && within(gid, systemIdRange(entries.loginDefinitions, "GID")) ? "Ours" : "Foreign"
  const [name, , uid, primaryGid, , home, shell] = entries.passwd.trim().split(":")
  const user: ServerAccountState = entries.passwd.trim() === "" ? "Absent"
    : name === SERVER_USER && within(Number(uid), systemIdRange(entries.loginDefinitions, "UID")) && group === "Ours"
      && Number(primaryGid) === gid && home === SERVER_DATA_DIRECTORY && noLoginShells.has(shell ?? "") ? "Ours" : "Foreign"
  return { user, group }
}

/** Looks up the `magnitude` user and group; readable without root. */
export const inspectServerAccounts = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const lookup = (database: string) => commandOutput("/usr/bin/getent", database, SERVER_USER).pipe(Effect.orElseSucceed(() => ""))
  return classifyServerAccounts({
    passwd: yield* lookup("passwd"),
    group: yield* lookup("group"),
    loginDefinitions: yield* fs.readFileString("/etc/login.defs").pipe(Effect.orElseSucceed(() => "")),
  })
})

const foreignAccount = `An account or group named ${SERVER_USER} already exists, and it isn't the system account Magnitude creates `
  + `(home ${SERVER_DATA_DIRECTORY}, no login shell). Server setup won't use it. Nothing was changed.`

const systemctl = "/usr/bin/systemctl"
const SYSTEMD = "/run/systemd/system"
const UNIT = "magnitude.service"
const visudo = (fs: FileSystem.FileSystem) => Effect.findFirst(["/usr/sbin/visudo", "/usr/bin/visudo"], path => fs.exists(path).pipe(Effect.orElseSucceed(() => false)))

/**
 * Leaves no symbolic link to follow, so ownership and mode changes reach only our directory.
 * Kept data with no service account goes to root, private, until setup gives it back.
 */
const giveDataToRoot = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!(yield* fs.exists(SERVER_DATA_DIRECTORY)) || (yield* fs.realPath(SERVER_DATA_DIRECTORY)) !== SERVER_DATA_DIRECTORY) return
  yield* run(`give ${SERVER_DATA_DIRECTORY} to root`, "/usr/bin/chown", "-R", "-h", "--", "root:root", SERVER_DATA_DIRECTORY)
  yield* fs.chmod(SERVER_DATA_DIRECTORY, 0o700)
})

/**
 * Writes the rule under a name sudo ignores (it contains a dot), and moves it into place only once
 * visudo accepts it. Without sudo there is nothing to grant, and the service only notices updates.
 */
const writeSudoersRule = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const checker = yield* visudo(fs)
  const directory = dirname(SERVER_SUDOERS_PATH)
  if (Option.isNone(checker) || !(yield* fs.exists(directory))) return false
  const temporary = join(directory, `.${randomUUID()}.tmp`)
  yield* Effect.gen(function* () {
    yield* fs.writeFileString(temporary, SERVER_SUDOERS_RULE, { flag: "wx", mode: 0o440 })
    yield* fs.chown(temporary, 0, 0)
    yield* fs.chmod(temporary, 0o440)
    yield* run("validate the sudoers rule", checker.value, "-cf", temporary)
    yield* fs.rename(temporary, SERVER_SUDOERS_PATH)
  }).pipe(Effect.onError(() => fs.remove(temporary, { force: true }).pipe(Effect.ignore)))
  return true
})

/**
 * Every local account may pass through the data directory to the control socket, as loopback HTTP
 * already admits them; none can list it, and the unit's UMask keeps everything inside it private.
 */
const SERVER_DATA_DIRECTORY_MODE = 0o711

/**
 * Linux root step, in an order that leaves nothing half set up: the account, its data directory,
 * the sudoers rule and the unit, and the marker last, since it switches every CLI
 * command to the server profile. On a failure it undoes what this run created, so running setup
 * again starts clean, and a re-run over a complete or partial setup finishes the remaining steps.
 */
export const installLinuxServer = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!(yield* fs.exists(SYSTEMD))) return yield* failed("Server setup needs systemd. Run `magnitude serve` instead.")
  const accounts = yield* inspectServerAccounts
  if (accounts.user === "Foreign" || accounts.group === "Foreign") return yield* failed(foreignAccount)
  const hadSudoersRule = yield* fs.exists(SERVER_SUDOERS_PATH)
  const wasEnabled = yield* succeeds(systemctl, "is-enabled", "--quiet", UNIT)
  const undo: Array<Effect.Effect<void, unknown, Services>> = []
  yield* Effect.gen(function* () {
    if (accounts.group === "Absent") {
      yield* run("create the magnitude group", "/usr/sbin/groupadd", "--system", SERVER_USER)
      undo.push(run("remove the magnitude group", "/usr/sbin/groupdel", SERVER_USER))
    }
    if (accounts.user === "Absent") {
      yield* run("create the magnitude user", "/usr/sbin/useradd", "--system", "--gid", SERVER_USER, "--home-dir", SERVER_DATA_DIRECTORY,
        "--no-create-home", "--shell", "/usr/sbin/nologin", "--comment", "Magnitude server", SERVER_USER)
      undo.push(run("remove the magnitude user", "/usr/sbin/userdel", SERVER_USER))
    }
    const serviceUid = (yield* commandOutput("/usr/bin/id", "-u", "--", SERVER_USER)).trim()
    if (!(yield* fs.exists(SERVER_DATA_DIRECTORY))) {
      yield* fs.makeDirectory(SERVER_DATA_DIRECTORY, { mode: SERVER_DATA_DIRECTORY_MODE })
      // Only while still empty; anything the service already wrote stays, given to root.
      undo.push(fs.remove(SERVER_DATA_DIRECTORY).pipe(Effect.ignore))
    }
    const data = yield* fs.stat(SERVER_DATA_DIRECTORY)
    if (data.type !== "Directory" || (yield* fs.realPath(SERVER_DATA_DIRECTORY)) !== SERVER_DATA_DIRECTORY) {
      return yield* failed(`${SERVER_DATA_DIRECTORY} must be a directory.`)
    }
    // Data kept by `server remove` belongs to root until a service account takes it back.
    if (String(Option.getOrUndefined(data.uid)) !== serviceUid) {
      yield* run(`give ${SERVER_DATA_DIRECTORY} to the magnitude user`, "/usr/bin/chown", "-R", "-h", "--", `${SERVER_USER}:${SERVER_USER}`, SERVER_DATA_DIRECTORY)
      undo.push(giveDataToRoot)
    }
    yield* fs.chmod(SERVER_DATA_DIRECTORY, SERVER_DATA_DIRECTORY_MODE)
    if (!hadSudoersRule) undo.push(fs.remove(SERVER_SUDOERS_PATH, { force: true }))
    const updates = yield* writeSudoersRule
    yield* run("reload systemd", systemctl, "daemon-reload")
    if (!wasEnabled) undo.push(run(`stop ${UNIT}`, systemctl, "disable", "--now", UNIT))
    yield* run(`start ${UNIT}`, systemctl, "enable", "--now", UNIT)
    yield* rootDirectory(dirname(SERVER_MARKER_PATH), 0o755)
    yield* writeRootFile(SERVER_MARKER_PATH, SERVER_MARKER_CONTENT)
    if (!updates) yield* Console.log("sudo isn't installed, so the server won't install Magnitude updates by itself. Install sudo, then run `magnitude server setup` again.")
  }).pipe(Effect.catchAll(error => Effect.gen(function* () {
    const undone = yield* Effect.forEach(undo.reverse(), step => Effect.either(step))
    return yield* failed(`${describe(error)}\n${undone.every(Either.isRight)
      ? "Setup undid the changes it had made. Run `magnitude server setup` again once that's fixed."
      : "Setup could not undo all of its changes; `magnitude server remove` removes what is left."}`)
  })))
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server setup could not finish.")))

/**
 * Linux root step: removes whatever of the server exists, so it can always be run again, and the
 * marker last, so a failure leaves the server visible to another `server remove`. Only Magnitude's own
 * system account and group are deleted; the data directory is kept, owned by root.
 */
export const removeLinuxServer = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  // The unit may already be gone with the package; disabling it then has nothing to do.
  if ((yield* fs.exists(SYSTEMD)) && (yield* succeeds(systemctl, "cat", UNIT))) yield* run(`stop ${UNIT}`, systemctl, "disable", "--now", UNIT)
  yield* fs.remove(SERVER_SUDOERS_PATH, { force: true })
  const accounts = yield* inspectServerAccounts
  if (accounts.user === "Ours") yield* run("remove the magnitude user", "/usr/sbin/userdel", SERVER_USER)
  if (accounts.user === "Foreign") yield* Console.log(`Kept the account ${SERVER_USER}: it isn't the system account Magnitude created.`)
  // userdel may already have removed the group with its user.
  const group = (yield* inspectServerAccounts).group
  if (group === "Ours") yield* run("remove the magnitude group", "/usr/sbin/groupdel", SERVER_USER)
  if (group === "Foreign" && accounts.user !== "Foreign") yield* Console.log(`Kept the group ${SERVER_USER}: it isn't the group Magnitude created.`)
  if (accounts.user !== "Foreign") yield* giveDataToRoot
  yield* fs.remove(SERVER_MARKER_PATH, { force: true })
  const markerDirectory = dirname(SERVER_MARKER_PATH)
  if ((yield* fs.exists(markerDirectory)) && (yield* fs.readDirectory(markerDirectory)).length === 0) yield* fs.remove(markerDirectory, { recursive: true })
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server removal could not finish.")))

/**
 * Package installation step: removing the package deletes the sudoers rule with the helper it names,
 * so installing over a configured server writes the current rule again. Only a server setup made,
 * marker and system account both, gets it; another account named magnitude never does.
 */
export const restoreLinuxServerGrant = Effect.gen(function* () {
  if (!(yield* isServerProfileActive("linux"))) return
  const accounts = yield* inspectServerAccounts
  if (accounts.user !== "Ours" || accounts.group !== "Ours") return
  yield* writeSudoersRule
}).pipe(Effect.mapError(() => failed("The Magnitude server's update permission could not be restored. Run `magnitude server setup` again.")))

/** Whether anything `server setup` leaves on Linux exists, so `server remove` can finish a partial setup or removal. */
export const hasLinuxServerRemnants = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const accounts = yield* inspectServerAccounts
  return accounts.user === "Ours" || accounts.group === "Ours"
    || (yield* fs.exists(SERVER_MARKER_PATH).pipe(Effect.orElseSucceed(() => false)))
    || (yield* fs.exists(SERVER_SUDOERS_PATH).pipe(Effect.orElseSucceed(() => false)))
    || (yield* succeeds(systemctl, "is-enabled", "--quiet", UNIT))
})

const launchctl = "/bin/launchctl"
const xml = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")

/** The LaunchDaemon that serves as the person who set it up, at boot and after logout. */
export const macServerPlist = (options: { readonly cli: string; readonly user: string; readonly home: string }) => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${MAC_SERVER_LABEL}</string>
  <key>AssociatedBundleIdentifiers</key><string>dev.magnitude.desktop</string>
  <key>ProgramArguments</key>
  <array><string>${xml(options.cli)}</string><string>serve</string></array>
  <key>UserName</key><string>${xml(options.user)}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>${xml(options.home)}</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>WorkingDirectory</key><string>${xml(options.home)}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>${xml(options.home)}/.magnitude/logs/server.log</string>
  <key>StandardErrorPath</key><string>${xml(options.home)}/.magnitude/logs/server.log</string>
</dict>
</plist>
`

/**
 * macOS root step: a root-owned LaunchDaemon that runs the app's CLI as the person, loaded into the
 * system domain. Re-running it replaces the registration; a first setup that cannot start the daemon
 * removes it again, so nothing is left half set up.
 */
export const installMacServer = (user: { readonly name: string; readonly uid: number }, cli: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const userUid = (yield* commandOutput("/usr/bin/id", "-u", "--", user.name).pipe(Effect.orElseSucceed(() => ""))).trim()
  if (userUid !== String(user.uid)) return yield* failed("The account that ran sudo could not be identified.")
  const record = yield* commandOutput("/usr/bin/dscl", ".", "-read", `/Users/${user.name}`, "NFSHomeDirectory")
  const home = /^NFSHomeDirectory:\s*(\/\S.*)$/m.exec(record)?.[1]?.trim()
  if (home === undefined || !home.startsWith("/") || home.includes("\n")) return yield* failed("Your home folder could not be found.")
  // The log folder and file belong to the person, so launchd never creates them as root, and only they can read the log.
  const logs = join(home, ".magnitude", "logs")
  const log = join(logs, "server.log")
  yield* run("create the log folder", "/usr/bin/sudo", "-u", user.name, "--", "/bin/mkdir", "-p", "-m", "700", logs)
  yield* run("create the server log", "/usr/bin/sudo", "-u", user.name, "--", "/usr/bin/touch", log)
  yield* run("protect the server log", "/usr/bin/sudo", "-u", user.name, "--", "/bin/chmod", "600", log)
  const registered = yield* fs.exists(MAC_SERVER_PLIST)
  const temporary = join(dirname(MAC_SERVER_PLIST), `.${randomUUID()}.tmp`)
  yield* fs.writeFileString(temporary, macServerPlist({ cli, user: user.name, home }), { flag: "wx", mode: 0o644 })
  yield* run("protect the LaunchDaemon", "/usr/sbin/chown", "root:wheel", temporary).pipe(
    Effect.onError(() => fs.remove(temporary, { force: true }).pipe(Effect.ignore)))
  yield* fs.chmod(temporary, 0o644)
  yield* fs.rename(temporary, MAC_SERVER_PLIST)
  // Replace a previous registration; a job switched off in Login Items is switched back on by setup.
  yield* succeeds(launchctl, "bootout", `system/${MAC_SERVER_LABEL}`)
  yield* succeeds(launchctl, "enable", `system/${MAC_SERVER_LABEL}`)
  // A job that was just booted out can briefly refuse a new bootstrap while it finishes exiting.
  yield* run("start the Magnitude server", launchctl, "bootstrap", "system", MAC_SERVER_PLIST).pipe(
    Effect.retry(Schedule.spaced("1 second").pipe(Schedule.intersect(Schedule.recurs(5)))),
    Effect.catchAll(error => registered ? Effect.fail(error) : fs.remove(MAC_SERVER_PLIST, { force: true }).pipe(Effect.ignore,
      Effect.zipRight(Effect.fail(failed(`${error.message}\nSetup removed the LaunchDaemon again. Run \`magnitude server setup\` again once that's fixed.`))))))
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server setup could not finish.")))

/** macOS root step: unloads and removes the LaunchDaemon; the person's data stays in ~/.magnitude. */
export const removeMacServer = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* succeeds(launchctl, "bootout", `system/${MAC_SERVER_LABEL}`)
  yield* fs.remove(MAC_SERVER_PLIST, { force: true })
}).pipe(Effect.mapError(error => error instanceof ServerInstallationFailed ? error : failed("Server removal could not finish.")))
