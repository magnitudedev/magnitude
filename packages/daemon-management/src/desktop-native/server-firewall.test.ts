import { CommandExecutor } from "@effect/platform"
import type { Command } from "@effect/platform/Command"
import { SystemError } from "@effect/platform/Error"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { openServerFirewall } from "./server-firewall"

const run = (succeeds: (args: string) => boolean | "missing", ufwStatus = "") => {
  const commands: string[] = []
  const executor: CommandExecutor.CommandExecutor = { ...CommandExecutor.makeExecutor(() => Effect.die("unexpected start")),
    exitCode: (command: Command) => Effect.suspend(() => {
      if (command._tag !== "StandardCommand") return Effect.die("unexpected command")
      const args = command.args.join(" ")
      commands.push(`${command.command} ${args}`)
      const result = succeeds(args)
      return result === "missing"
        ? Effect.fail(new SystemError({ reason: "NotFound", module: "Command", method: "spawn", pathOrDescriptor: command.command })) : Effect.succeed(CommandExecutor.ExitCode(result ? 0 : 1))
    }),
    string: (command: Command) => Effect.suspend(() => {
      if (command._tag !== "StandardCommand") return Effect.die("unexpected command")
      commands.push(`${command.command} ${command.args.join(" ")}`)
      return ufwStatus === "" ? Effect.fail(new SystemError({ reason: "NotFound", module: "Command", method: "spawn", pathOrDescriptor: command.command }))
        : Effect.succeed(ufwStatus)
    }),
  }
  return Effect.runPromise(openServerFirewall.pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor)))
    .then(message => ({ message, commands }))
}

describe("server firewall", () => {
  it("does nothing without firewalld or ufw, or when they are not running", async () => {
    const commands = ["/usr/bin/firewall-cmd --state", "/usr/sbin/ufw status"]
    expect(await run(() => "missing")).toEqual({ message: Option.none(), commands })
    expect(await run(() => false)).toEqual({ message: Option.none(), commands })
    expect(await run(() => false, "Status: inactive\n")).toEqual({ message: Option.none(), commands })
  })
  const ufwActive = (...rules: string[]) => ["Status: active", "", "To                         Action      From",
    "--                         ------      ----", "22/tcp                     ALLOW       Anywhere", ...rules, ""].join("\n")
  it("gives the exact command when an active ufw doesn't allow the server port, and changes nothing", async () => {
    const result = await run(() => false, ufwActive())
    expect(Option.getOrThrow(result.message)).toBe("ufw blocks TCP port 10100. To let other computers reach the server, run: sudo ufw allow 10100/tcp")
    expect(result.commands).toEqual(["/usr/bin/firewall-cmd --state", "/usr/sbin/ufw status"])
  })
  it.each([
    "10100/tcp                  ALLOW       Anywhere",
    "10100                      ALLOW IN    Anywhere",
    "10100/tcp                  ALLOW       10.0.0.0/8",
  ])("is silent when ufw already allows the server port: %s", async rule => {
    expect((await run(() => false, ufwActive(rule))).message).toEqual(Option.none())
  })
  it("still warns when ufw only denies or allows another port", async () => {
    expect(Option.isSome((await run(() => false, ufwActive("10100/tcp                  DENY        Anywhere", "10101/tcp                  ALLOW       Anywhere"))).message)).toBe(true)
  })
  it("opens the server port in the default zone, now and permanently, and says so", async () => {
    const result = await run(args => !args.includes("--query-port"))
    expect(result.commands).toEqual([
      "/usr/bin/firewall-cmd --state", "/usr/bin/firewall-cmd --query-port=10100/tcp",
      "/usr/bin/firewall-cmd --add-port=10100/tcp", "/usr/bin/firewall-cmd --permanent --add-port=10100/tcp",
    ])
    expect(Option.getOrThrow(result.message)).toBe("Opened TCP port 10100 in firewalld's default zone, so other computers can reach the server.")
  })
  it("is silent when the port is already open", async () => {
    const result = await run(args => !args.includes("--add-port"))
    expect(result.message).toEqual(Option.none())
    expect(result.commands.some(command => command.includes("--add-port"))).toBe(false)
  })
  it("gives the exact command when the port cannot be opened", async () => {
    const result = await run(args => args === "--state")
    expect(Option.getOrThrow(result.message)).toContain("sudo firewall-cmd --permanent --add-port=10100/tcp && sudo firewall-cmd --reload")
  })
})
