import { Command, CommandExecutor } from "@effect/platform"
import { Effect, Option } from "effect"
import { SERVER_PORT } from "./server-profile"

const firewallCmd = "/usr/bin/firewall-cmd"
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
 * other firewalls are left as they are. It never fails setup; it returns what to tell the person, if anything.
 */
export const openServerFirewall = Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const firewall = (...args: string[]) => executor.exitCode(Command.make(firewallCmd, ...args).pipe(
    Command.env(environment))).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false))
  if (!(yield* firewall("--state"))) return yield* checkUfw(executor)
  if ((yield* firewall(`--query-port=${port}`)) && (yield* firewall("--permanent", `--query-port=${port}`))) return Option.none<string>()
  if ((yield* firewall(`--add-port=${port}`)) && (yield* firewall("--permanent", `--add-port=${port}`))) {
    return Option.some(`Opened TCP port ${SERVER_PORT} in firewalld's default zone, so other computers can reach the server.`)
  }
  return Option.some(`firewalld blocks TCP port ${SERVER_PORT}, and it could not be opened. To let other computers reach the server, run: `
    + `sudo firewall-cmd --permanent --add-port=${port} && sudo firewall-cmd --reload`)
})
