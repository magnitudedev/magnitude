import { CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Either, Option } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { installLinuxServer, parseServerInstallation, removeLinuxServer, requireInstalledRoot } from "./server-installation"

const sudo = { SUDO_USER: "ada", SUDO_UID: "1000" }
const parse = (argv: readonly string[], environment: Record<string, string | undefined> = sudo) =>
  Effect.runPromise(Effect.either(parseServerInstallation(argv, environment)))

describe("hidden root command grammar", () => {
  it("accepts exactly the user that ran sudo, or removal without arguments", async () => {
    expect(await parse(["_server-install", "ada"])).toEqual(Either.right({ _tag: "Install", user: Option.some({ name: "ada", uid: 1000 }) }))
    expect(await parse(["_server-remove"])).toEqual(Either.right({ _tag: "Remove" }))
  })
  it.each([
    ["an extra argument", ["_server-install", "ada", "extra"]],
    ["an option", ["_server-install", "--user", "ada"]],
    ["an option after the user", ["_server-install", "ada", "--force"]],
    ["a leading program option", ["-v", "_server-install", "ada"]],
    ["arguments to removal", ["_server-remove", "ada"]],
    ["another user", ["_server-install", "grace"]],
    ["root", ["_server-install", "root"]],
    ["the service account", ["_server-install", "magnitude"]],
    ["a path", ["_server-install", "../ada"]],
    ["shell syntax", ["_server-install", "ada;id"]],
    ["an empty user", ["_server-install", ""]],
    ["a newline", ["_server-install", "ada\nroot"]],
    ["an over-long name", ["_server-install", "a".repeat(33)]],
    ["another command", ["_install-application-update", "ada"]],
  ])("refuses %s", async (_, argv) => {
    expect(Either.isLeft(await parse(argv))).toBe(true)
  })
  it("accepts root's own setup only from a root login, not through sudo", async () => {
    expect(await parse(["_server-install"], {})).toEqual(Either.right({ _tag: "Install", user: Option.none() }))
    expect(Either.isLeft(await parse(["_server-install"]))).toBe(true)
  })
  it.each([
    ["without sudo", {}],
    ["when sudo names another user", { SUDO_USER: "grace", SUDO_UID: "1001" }],
    ["when sudo ran as root", { SUDO_USER: "ada", SUDO_UID: "0" }],
    ["with a malformed uid", { SUDO_USER: "ada", SUDO_UID: "1000x" }],
  ])("refuses %s", async (_, environment) => {
    expect(Either.isLeft(await parse(["_server-install", "ada"], environment))).toBe(true)
  })
})

describe("root step admission", () => {
  afterEach(() => vi.restoreAllMocks())
  const fs = (real: string) => FileSystem.makeNoop({ realPath: () => Effect.succeed(real) })
  const admit = (platform: string, real: string) => Effect.runPromise(Effect.either(requireInstalledRoot(platform, "/usr/bin/magnitude").pipe(
    Effect.provideService(FileSystem.FileSystem, fs(real)))))
  it("refuses an unprivileged caller", async () => {
    vi.spyOn(process as { getuid: () => number }, "getuid").mockReturnValue(1000)
    expect(Either.isLeft(await admit("linux", "/usr/lib/magnitude-desktop/resources/magnitude"))).toBe(true)
  })
  it("refuses a copy outside the installed path, and Windows", async () => {
    vi.spyOn(process as { getuid: () => number }, "getuid").mockReturnValue(0)
    expect(Either.isLeft(await admit("linux", "/tmp/magnitude"))).toBe(true)
    expect(Either.isLeft(await admit("win32", "C:\\magnitude.exe"))).toBe(true)
  })
  it("admits root running the installed CLI", async () => {
    vi.spyOn(process as { getuid: () => number }, "getuid").mockReturnValue(0)
    expect(await admit("linux", "/usr/lib/magnitude-desktop/resources/magnitude")).toEqual(Either.right("/usr/lib/magnitude-desktop/resources/magnitude"))
    expect(await admit("darwin", "/Applications/Magnitude.app/Contents/Resources/magnitude")).toEqual(Either.right("/Applications/Magnitude.app/Contents/Resources/magnitude"))
  })
})

describe("Linux root step", () => {
  const machine = (options: { readonly existing: boolean; readonly dataUid: number }) => {
    const commands: string[] = []
    const files = new Map<string, { type: "File" | "Directory"; uid: number }>([
      ["/run/systemd/system", { type: "Directory", uid: 0 }],
      ...(options.existing ? [["/var/lib/magnitude", { type: "Directory", uid: options.dataUid }] as const] : []),
    ])
    const written: string[] = []
    const fs = FileSystem.makeNoop({
      exists: path => Effect.succeed(files.has(path)),
      stat: path => Effect.succeed({ type: files.get(path)!.type, uid: Option.some(files.get(path)!.uid), mode: 0 } as FileSystem.File.Info),
      realPath: path => Effect.succeed(path),
      makeDirectory: path => Effect.sync(() => { files.set(path, { type: "Directory", uid: path === "/var/lib/magnitude" ? 0 : 0 }); written.push(`mkdir ${path}`) }),
      chmod: (path, mode) => Effect.sync(() => { written.push(`chmod ${mode.toString(8)} ${path}`) }),
      writeFileString: path => Effect.sync(() => { written.push(`write ${path.replace(/\.[^/]*\.tmp$/, "<tmp>")}`) }),
      rename: (_, to) => Effect.sync(() => { written.push(`rename ${to}`) }),
      remove: path => Effect.sync(() => { files.delete(path); written.push(`remove ${path}`) }),
      readDirectory: () => Effect.succeed([]),
    })
    const accounts = new Set(options.existing ? ["group", "passwd"] : [])
    const executor = CommandExecutor.makeExecutor(() => Effect.die("unexpected start"))
    const command = (value: import("@effect/platform/Command").Command) => value._tag === "StandardCommand" ? [value.command, ...value.args].join(" ") : ""
    const fake: CommandExecutor.CommandExecutor = { ...executor,
      exitCode: value => Effect.sync(() => {
        const line = command(value)
        commands.push(line)
        const lookup = /^\/usr\/bin\/getent (group|passwd) magnitude$/.exec(line)
        if (lookup) return CommandExecutor.ExitCode(accounts.has(lookup[1]!) ? 0 : 2)
        return CommandExecutor.ExitCode(0)
      }),
      string: value => Effect.sync(() => {
        const line = command(value)
        commands.push(line)
        return line.endsWith("-- ada") ? "1000\n" : "990\n"
      }),
    }
    return { commands, written, provide: <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor>) =>
      effect.pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(CommandExecutor.CommandExecutor, fake)) }
  }

  it("creates the account, data directory, group membership and marker, then enables the unit", async () => {
    const m = machine({ existing: false, dataUid: 0 })
    await Effect.runPromise(m.provide(installLinuxServer(Option.some({ name: "ada", uid: 1000 }))))
    expect(m.commands).toEqual([
      "/usr/bin/id -u -- ada",
      "/usr/bin/getent group magnitude", "/usr/sbin/groupadd --system magnitude",
      "/usr/bin/getent passwd magnitude",
      "/usr/sbin/useradd --system --gid magnitude --home-dir /var/lib/magnitude --no-create-home --shell /usr/sbin/nologin --comment Magnitude server magnitude",
      "/usr/bin/id -u -- magnitude",
      "/usr/bin/chown -R -h -- magnitude:magnitude /var/lib/magnitude",
      "/usr/sbin/usermod -aG magnitude -- ada",
      "/usr/bin/systemctl daemon-reload", "/usr/bin/systemctl enable --now magnitude.service",
    ])
    expect(m.written).toEqual(expect.arrayContaining(["mkdir /var/lib/magnitude", "chmod 750 /var/lib/magnitude", "mkdir /etc/magnitude", "chmod 755 /etc/magnitude", "rename /etc/magnitude/server"]))
  })
  it("reuses a kept data directory that already belongs to the service account", async () => {
    const m = machine({ existing: true, dataUid: 990 })
    await Effect.runPromise(m.provide(installLinuxServer(Option.some({ name: "ada", uid: 1000 }))))
    expect(m.commands.some(line => line.includes("useradd") || line.includes("groupadd") || line.includes("chown"))).toBe(false)
  })
  it("adds no one to the group for a root login", async () => {
    const m = machine({ existing: false, dataUid: 0 })
    await Effect.runPromise(m.provide(installLinuxServer(Option.none())))
    expect(m.commands.some(line => line.includes("usermod") || line.startsWith("/usr/bin/id -u -- ada"))).toBe(false)
    expect(m.commands.at(-1)).toBe("/usr/bin/systemctl enable --now magnitude.service")
  })
  it("refuses when the sudo uid does not belong to the named user", async () => {
    const m = machine({ existing: false, dataUid: 0 })
    expect(Either.isLeft(await Effect.runPromise(Effect.either(m.provide(installLinuxServer(Option.some({ name: "ada", uid: 1001 }))))))).toBe(true)
    expect(m.commands).toEqual(["/usr/bin/id -u -- ada"])
  })
  it("removal stops the unit and removes the account and marker, keeping the data directory", async () => {
    const m = machine({ existing: true, dataUid: 990 })
    await Effect.runPromise(m.provide(removeLinuxServer))
    expect(m.commands).toEqual([
      "/usr/bin/systemctl cat magnitude.service", "/usr/bin/systemctl disable --now magnitude.service",
      "/usr/bin/getent passwd magnitude", "/usr/sbin/userdel magnitude",
      "/usr/bin/getent group magnitude", "/usr/sbin/groupdel magnitude",
    ])
    expect(m.written).not.toContain("remove /var/lib/magnitude")
    expect(m.written).toContain("remove /etc/magnitude/server")
  })
})
