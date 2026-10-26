import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { dirname } from "node:path"
import { SERVER_MARKER_PATH, SERVER_PORT } from "./server-profile"

const firewallCmd = "/usr/bin/firewall-cmd"
const firewallOfflineCmd = "/usr/bin/firewall-offline-cmd"
/** Setup records the port it opened, so removal closes only that and never an administrator's own rule. */
export const SERVER_FIREWALL_RECORD_PATH = `${dirname(SERVER_MARKER_PATH)}/firewalld-port`
const port = `${SERVER_PORT}/tcp`

const ufw = "/usr/sbin/ufw"
const environment = { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }

/** Whether `ufw status` lists a rule that allows the server's port in. */
export const ufwAllowsServerPort = (status: string) => status.split("\n").some(line =>
  new RegExp(`^${SERVER_PORT}(/tcp)?\\s+ALLOW(\\s+IN)?\\s`).test(line.trim() + " "))

/**
 * An active ufw (Ubuntu, Debian) that doesn't allow the server's port drops other computers' connections.
 * ufw is the administrator's own policy, so setup only says how to allow the port.
 */
const checkUfw = (executor: CommandExecutor.CommandExecutor) => Effect.gen(function* () {
  const status = yield* executor.string(Command.make(ufw, "status").pipe(Command.env(environment))).pipe(Effect.orElseSucceed(() => ""))
  if (!/^Status: active$/m.test(status) || ufwAllowsServerPort(status)) return Option.none<string>()
  return Option.some(`ufw blocks TCP port ${SERVER_PORT}. To let other computers reach the server, run: sudo ufw allow ${port}`)
})

/**
 * firewalld (Fedora, RHEL) drops inbound connections to the server's port by default, so the Linux root
 * step opens it in the default zone, now and permanently. An active ufw gets the command to allow it;
 * other firewalls are left as they are. It never fails setup; it returns what to tell the person, if anything,
 * and whether it opened the port, for removal to close it again.
 */
export const openServerFirewall = Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const firewall = (...args: string[]) => executor.exitCode(Command.make(firewallCmd, ...args).pipe(
    Command.env(environment))).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false))
  if (!(yield* firewall("--state"))) return { message: yield* checkUfw(executor), opened: false }
  if ((yield* firewall(`--query-port=${port}`)) && (yield* firewall("--permanent", `--query-port=${port}`))) return { message: Option.none<string>(), opened: false }
  if ((yield* firewall(`--add-port=${port}`)) && (yield* firewall("--permanent", `--add-port=${port}`))) {
    return { message: Option.some(`Opened TCP port ${SERVER_PORT} in firewalld's default zone, so other computers can reach the server.`), opened: true }
  }
  return { message: Option.some(`firewalld blocks TCP port ${SERVER_PORT}, and it could not be opened. To let other computers reach the server, run: `
    + `sudo firewall-cmd --permanent --add-port=${port} && sudo firewall-cmd --reload`), opened: false }
})

export const recordOpenedServerFirewall = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.writeFileString(SERVER_FIREWALL_RECORD_PATH, `${port}\n`, { mode: 0o644 })
})

/**
 * Removal step: closes the port only if setup recorded opening it, in the running firewall and its
 * permanent configuration, or the permanent configuration alone while firewalld is stopped.
 */
export const closeServerFirewall = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!(yield* fs.exists(SERVER_FIREWALL_RECORD_PATH))) return Option.none<string>()
  const executor = yield* CommandExecutor.CommandExecutor
  const succeeds = (command: string, ...args: string[]) => executor.exitCode(Command.make(command, ...args).pipe(
    Command.env(environment))).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false))
  const closed = (yield* succeeds(firewallCmd, "--state"))
    ? (yield* succeeds(firewallCmd, `--remove-port=${port}`)) && (yield* succeeds(firewallCmd, "--permanent", `--remove-port=${port}`))
    : yield* succeeds(firewallOfflineCmd, `--remove-port=${port}`)
  if (!closed) return Option.some(`TCP port ${SERVER_PORT}, which setup opened in firewalld, could not be closed. To close it, run: `
    + `sudo firewall-cmd --permanent --remove-port=${port} && sudo firewall-cmd --reload`)
  yield* fs.remove(SERVER_FIREWALL_RECORD_PATH, { force: true })
  return Option.some(`Closed TCP port ${SERVER_PORT} in firewalld, which setup had opened.`)
})
