import { Command, CommandExecutor } from "@effect/platform"
import { Effect, Option } from "effect"
import { SERVER_PORT } from "./server-profile"

const firewallCmd = "/usr/bin/firewall-cmd"
const port = `${SERVER_PORT}/tcp`

/**
 * firewalld (Fedora, RHEL) drops inbound connections to the server's port by default, so the Linux root
 * step opens it in the default zone, now and permanently. Other firewalls are left as they are. It never
 * fails setup; it returns what to tell the person, if anything.
 */
export const openServerFirewall = Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const firewall = (...args: string[]) => executor.exitCode(Command.make(firewallCmd, ...args).pipe(
    Command.env({ PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }))).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false))
  if (!(yield* firewall("--state"))) return Option.none<string>()
  if ((yield* firewall(`--query-port=${port}`)) && (yield* firewall("--permanent", `--query-port=${port}`))) return Option.none<string>()
  if ((yield* firewall(`--add-port=${port}`)) && (yield* firewall("--permanent", `--add-port=${port}`))) {
    return Option.some(`Opened TCP port ${SERVER_PORT} in firewalld's default zone, so other computers can reach the server.`)
  }
  return Option.some(`firewalld blocks TCP port ${SERVER_PORT}, and it could not be opened. To let other computers reach the server, run: `
    + `sudo firewall-cmd --permanent --add-port=${port} && sudo firewall-cmd --reload`)
})
