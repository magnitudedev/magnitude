import { execFile } from "node:child_process"
import { chmod, chown, copyFile, lstat, mkdir, readdir, readlink, rename, rm, symlink, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { promisify } from "node:util"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { isLegacyServerMarked, legacyServerUnit, LEGACY_SERVER_HOME, LEGACY_SERVER_MARKER, LEGACY_SERVER_SUDOERS, LEGACY_SERVER_UNIT,
  LEGACY_SERVER_USER, LEGACY_SERVER_WRAPPER } from "@magnitudedev/daemon-management/desktop-native"
import { CLI_VERSION } from "../version"

class ServerConversionFailed extends Schema.TaggedError<ServerConversionFailed>()("ServerConversionFailed", { message: Schema.String }) {}

const PACKAGE_RESOURCES = "/usr/lib/magnitude-desktop/resources"
const PROFILE = join(LEGACY_SERVER_HOME, ".magnitude")
const INSTALLATION = join(PROFILE, "cli")
const COMMAND = join(PROFILE, "bin", "magnitude")
const PAYLOAD: ReadonlyArray<readonly [source: string, target: string, required: boolean]> = [
  ["magnitude", "magnitude", true], ["magnitude-service", "magnitude-service", true], ["desktop-host.node", "desktop-host.node", true],
  ["magnitude-command", "magnitude-command", false], ["update-configuration.json", "update-configuration.json", true],
  ["update-trust.json", "update-trust.json", true], ["LICENSE", "LICENSE", false],
]

const run = promisify(execFile)
const step = <A>(message: string, operation: () => Promise<A>) => Effect.tryPromise({
  try: operation, catch: error => new ServerConversionFailed({ message: `${message}: ${error instanceof Error ? error.message : String(error)}` }),
})
const exists = (path: string) => step(`Could not inspect ${path}`, () => lstat(path).then(() => true, () => false))
const output = (command: string, ...args: string[]) => step(`${command} failed`, () => run(command, args)).pipe(Effect.map(result => result.stdout.trim()))

/** The account's ids, so everything moved or written for the service belongs to it. */
const account = Effect.all([output("id", "-u", LEGACY_SERVER_USER), output("id", "-g", LEGACY_SERVER_USER)]).pipe(
  Effect.map(([uid, gid]) => ({ uid: Number(uid), gid: Number(gid) })))

const chownTree = (path: string, owner: { readonly uid: number; readonly gid: number }) =>
  output("chown", "-R", "--no-dereference", `${owner.uid}:${owner.gid}`, path)

/** The server's data sits directly in the account's home; its profile becomes `~/.magnitude`. Home dotfiles stay. */
const moveData = (owner: { readonly uid: number; readonly gid: number }) => Effect.gen(function* () {
  yield* step("Could not create the profile", () => mkdir(PROFILE, { recursive: true, mode: 0o700 }))
  yield* step("Could not own the profile", () => chown(PROFILE, owner.uid, owner.gid))
  for (const name of yield* step("Could not list the server data", () => readdir(LEGACY_SERVER_HOME))) {
    if (name.startsWith(".")) continue
    const target = join(PROFILE, name)
    if (yield* exists(target)) yield* step(`Could not replace ${target}`, () => rm(target, { recursive: true, force: true }))
    yield* step(`Could not move ${name}`, () => rename(join(LEGACY_SERVER_HOME, name), target))
  }
})

/** The package's own files, laid out as a CLI-only installation owned by the account, made current atomically. */
const installCli = (owner: { readonly uid: number; readonly gid: number }) => Effect.gen(function* () {
  const version = join(INSTALLATION, CLI_VERSION)
  const staging = join(INSTALLATION, `.staging-${CLI_VERSION}`)
  yield* step("Could not create the installation", () => mkdir(INSTALLATION, { recursive: true, mode: 0o755 }))
  yield* step("Could not clear the staging folder", () => rm(staging, { recursive: true, force: true }))
  yield* step("Could not create the staging folder", () => mkdir(staging, { mode: 0o755 }))
  for (const [source, target, required] of PAYLOAD) {
    const from = join(PACKAGE_RESOURCES, source)
    if (!(yield* exists(from))) {
      if (required) return yield* new ServerConversionFailed({ message: `The package is missing ${from}.` })
      continue
    }
    yield* step(`Could not copy ${source}`, () => copyFile(from, join(staging, target)))
    yield* step(`Could not set the mode of ${target}`, () => chmod(join(staging, target), target.endsWith(".json") || target === "LICENSE" ? 0o644 : 0o755))
  }
  yield* chownTree(staging, owner)
  yield* step("Could not replace the version folder", () => rm(version, { recursive: true, force: true }))
  yield* step("Could not move the installation into place", () => rename(staging, version))
  const pointer = join(INSTALLATION, `.current-${process.pid}`)
  yield* step("Could not link the current version", () => rm(pointer, { force: true }).then(() => symlink(CLI_VERSION, pointer)))
  yield* step("Could not switch the current version", () => rename(pointer, join(INSTALLATION, "current")))
  yield* step("Could not create the command folder", () => mkdir(join(PROFILE, "bin"), { recursive: true, mode: 0o755 }))
  const current = yield* step("Could not read the command link", () => readlink(COMMAND).then(target => target, () => ""))
  if (current !== join(INSTALLATION, "current", "magnitude")) {
    const link = `${COMMAND}.${process.pid}`
    yield* step("Could not link the command", () => rm(link, { force: true }).then(() => symlink(join(INSTALLATION, "current", "magnitude"), link)))
    yield* step("Could not place the command", () => rename(link, COMMAND))
  }
  yield* chownTree(join(PROFILE, "bin"), owner)
  yield* step("Could not own the installation", () => chown(INSTALLATION, owner.uid, owner.gid))
})

const wrapper = `#!/bin/sh
# Written by the Magnitude desktop package: runs the Magnitude server's command as its account.
exec sudo -u ${LEGACY_SERVER_USER} -H ${COMMAND} "$@"
`

/** Which command removes the desktop package on this distribution. */
const removalCommand = Effect.gen(function* () {
  for (const [tool, command] of [["apt-get", "DEBIAN_FRONTEND=noninteractive apt-get remove -y magnitude-desktop"],
    ["dnf", "dnf remove -y magnitude-desktop"], ["pacman", "pacman -R --noconfirm magnitude-desktop"]] as const) {
    if (yield* output("sh", "-c", `command -v ${tool}`).pipe(Effect.as(true), Effect.orElseSucceed(() => false))) return command
  }
  return undefined
})

/**
 * Headless machines (no display manager) no longer need the desktop package: admins get a wrapper for the
 * service's command, and the package is removed in a transient unit outside the service, once the package
 * manager is free. Machines with a display manager keep the package, whose app takes the service over while open.
 */
const retireDesktopPackage = Effect.gen(function* () {
  // Server images often default to graphical.target without any display, so a display manager is the sign
  // that someone may use the desktop app here; anything unclear keeps the package.
  const displayManager = yield* output("systemctl", "show", "--property=LoadState", "--value", "display-manager.service").pipe(
    Effect.map(state => state !== "not-found"), Effect.orElseSucceed(() => true))
  if (displayManager) return "kept (graphical machine)"
  // The unit's UMask=0007 would leave the wrapper root-only, so its mode is set after writing.
  yield* step("Could not write the command wrapper", () => writeFile(LEGACY_SERVER_WRAPPER, wrapper).then(() => chmod(LEGACY_SERVER_WRAPPER, 0o755)))
  const removal = yield* removalCommand
  if (removal === undefined) return "kept (no supported package manager found)"
  yield* output("systemd-run", "--unit=magnitude-desktop-removal", "--collect", "--on-active=60", "/bin/sh", "-c", removal)
  return "scheduled for removal"
})

/**
 * Converts a server set up before the CLI-only install into a CLI-only installation under the same account,
 * keeping its configuration, network key and port. It runs as root before each start of the converted unit
 * and does nothing once converted. The marker goes last, so a conversion that stops anywhere is completed by
 * the next start; until then the CLI-only `serve` refuses to run.
 */
export const convertServer = Effect.gen(function* () {
  if (!(yield* isLegacyServerMarked(process.platform))) return
  if (process.getuid?.() !== 0) return yield* new ServerConversionFailed({ message: "The server conversion runs as root from the Magnitude service." })
  const owner = yield* account
  yield* moveData(owner)
  yield* installCli(owner)
  const desktop = yield* retireDesktopPackage
  yield* step("Could not remove the old update permission", () => rm(LEGACY_SERVER_SUDOERS, { force: true }))
  yield* step("Could not remove the server marker", () => rm(LEGACY_SERVER_MARKER, { force: true }))
  yield* step("Could not remove the server folder", () => rm("/etc/magnitude", { recursive: true, force: true }))
  yield* Effect.sync(() => process.stdout.write(`Converted the Magnitude server into a CLI-only install at ${INSTALLATION} (version ${CLI_VERSION}); `
    + `its data is now in ${PROFILE}, and the desktop package is ${desktop}.\n`))
}).pipe(Effect.provide(BunContext.layer))

/**
 * The package's install step: write and enable the converted unit, which overrides the packaged one and
 * survives package removal. It may run inside the server's own update, so it stops and moves nothing.
 */
export const writeConvertedServerUnit = Effect.gen(function* () {
  if (!(yield* isLegacyServerMarked(process.platform))) return
  if (process.getuid?.() !== 0) return yield* new ServerConversionFailed({ message: "The converted server unit is written as root by the Magnitude package." })
  const temporary = `${LEGACY_SERVER_UNIT}.${process.pid}`
  yield* step("Could not write the converted server unit", () => writeFile(temporary, legacyServerUnit, { mode: 0o644 }).then(() => rename(temporary, LEGACY_SERVER_UNIT)))
  yield* output("systemctl", "daemon-reload").pipe(Effect.ignore)
  yield* output("systemctl", "reenable", "magnitude.service").pipe(Effect.ignore)
}).pipe(Effect.provide(BunContext.layer))

export const runConvertServer = (unitOnly: boolean) => Effect.runPromise((unitOnly ? writeConvertedServerUnit : convertServer).pipe(Effect.catchAll(error => Effect.sync(() => {
  process.stderr.write(`${error.message}\n`)
  process.exitCode = 1
}))))
