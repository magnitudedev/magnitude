import { CommandExecutor, FileSystem } from "@effect/platform"
import { SystemError } from "@effect/platform/Error"
import { Effect, Either, Fiber, Option, Sink, Stream, TestClock, TestContext } from "effect"
import { spawnSync } from "node:child_process"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it, vi } from "vitest"
import { classifyServerAccounts, hasLinuxServerRemnants, installLinuxServer, installMacServer, macServerPlist, parseServerInstallation, removeLinuxServer, requireInstalledRoot,
  systemIdRange } from "./server-installation"
import { SERVER_SUDOERS_RULE } from "./server-profile"

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

const tmp = (path: string) => path.replace(/\.[^/]*\.tmp$/, "<tmp>")
/** Records each command line (temporary names shown as <tmp>) and answers it with an exit code, stdout and stderr. */
const fakeExecutor = (commands: string[], respond: (line: string) => readonly [number, string, string]) => CommandExecutor.makeExecutor(value => Effect.sync(() => {
  const line = value._tag === "StandardCommand" ? [value.command, ...value.args].map(tmp).join(" ") : ""
  commands.push(line)
  const [code, stdout, stderr] = respond(line)
  return {
    [CommandExecutor.ProcessTypeId]: CommandExecutor.ProcessTypeId, pid: 1,
    exitCode: Effect.succeed(CommandExecutor.ExitCode(code)), isRunning: Effect.succeed(false), kill: () => Effect.void,
    stdout: Stream.make(new TextEncoder().encode(stdout)), stderr: Stream.make(new TextEncoder().encode(stderr)), stdin: Sink.drain,
  } as unknown as CommandExecutor.Process
}))

const ours = { passwd: "magnitude:x:990:990:Magnitude server:/var/lib/magnitude:/usr/sbin/nologin", group: "magnitude:x:990:ada" }
const human = { passwd: "magnitude:x:1000:1000:Magnitude:/home/magnitude:/bin/bash", group: "magnitude:x:1000:" }
const debianDefinitions = "UID_MIN 1000\nUID_MAX 60000\n#SYS_UID_MIN 100\n#SYS_UID_MAX 999\nGID_MIN 1000\n"

describe("service account criteria", () => {
  const classify = (entries: { passwd: string; group: string }, loginDefinitions = debianDefinitions) => classifyServerAccounts({ ...entries, loginDefinitions })
  it("recognizes only the system account setup creates", () => {
    expect(classify(ours)).toEqual({ user: "Ours", group: "Ours" })
    expect(classify({ passwd: "", group: "" })).toEqual({ user: "Absent", group: "Absent" })
    expect(classify({ passwd: "", group: ours.group })).toEqual({ user: "Absent", group: "Ours" })
  })
  it.each([
    ["a person's account", human],
    ["a person's account with a system uid", { ...ours, passwd: "magnitude:x:990:990::/home/magnitude:/bin/bash" }],
    ["a login shell", { ...ours, passwd: "magnitude:x:990:990::/var/lib/magnitude:/bin/bash" }],
    ["another home", { ...ours, passwd: "magnitude:x:990:990::/opt/magnitude:/usr/sbin/nologin" }],
    ["another primary group", { ...ours, passwd: "magnitude:x:990:27::/var/lib/magnitude:/usr/sbin/nologin" }],
    ["a uid above the system range", { ...ours, passwd: "magnitude:x:1001:990::/var/lib/magnitude:/usr/sbin/nologin" }],
    ["root's uid", { ...ours, passwd: "magnitude:x:0:990::/var/lib/magnitude:/usr/sbin/nologin" }],
  ])("treats %s as foreign", (_, entries) => {
    expect(classify(entries).user).toBe("Foreign")
  })
  it("treats a group outside the system range as foreign, and the account with it", () => {
    expect(classify({ ...ours, group: "magnitude:x:1000:" })).toEqual({ user: "Foreign", group: "Foreign" })
  })
  it("reads the system range from /etc/login.defs, else shadow's defaults", () => {
    expect(systemIdRange("SYS_UID_MIN 201\nSYS_UID_MAX 499\n", "UID")).toEqual({ min: 201, max: 499 })
    expect(systemIdRange("UID_MIN 500\n", "UID")).toEqual({ min: 101, max: 499 })
    expect(systemIdRange("", "GID")).toEqual({ min: 101, max: 999 })
    expect(classify(ours, "SYS_UID_MAX 499\nSYS_GID_MAX 499\n")).toEqual({ user: "Foreign", group: "Foreign" })
  })
})

describe("Linux root step", () => {
  afterEach(() => vi.restoreAllMocks())
  type Entry = { type: "File" | "Directory"; uid: number }
  const machine = (options: {
    readonly accounts?: { passwd: string; group: string }
    readonly setUp?: boolean
    readonly dataUid?: number
    readonly sudo?: boolean
    readonly fail?: RegExp
  } = {}) => {
    const commands: string[] = []
    const written: string[] = []
    const printed: string[] = []
    vi.spyOn(console, "log").mockImplementation((...values: unknown[]) => { printed.push(values.join(" ")) })
    const accounts = { ...(options.accounts ?? (options.setUp ? ours : { passwd: "", group: "" })) }
    let enabled = options.setUp ?? false
    const files = new Map<string, Entry>([
      ["/run/systemd/system", { type: "Directory", uid: 0 }],
      ["/etc/login.defs", { type: "File", uid: 0 }],
      ...(options.sudo === false ? [] : [["/usr/sbin/visudo", { type: "File", uid: 0 }] as const, ["/etc/sudoers.d", { type: "Directory", uid: 0 }] as const]),
      ...(options.setUp ? [["/var/lib/magnitude", { type: "Directory", uid: options.dataUid ?? 990 }] as const, ["/etc/magnitude", { type: "Directory", uid: 0 }] as const,
        ["/etc/magnitude/server", { type: "File", uid: 0 }] as const, ["/etc/sudoers.d/magnitude", { type: "File", uid: 0 }] as const] : []),
      ...(!options.setUp && options.dataUid !== undefined ? [["/var/lib/magnitude", { type: "Directory", uid: options.dataUid }] as const] : []),
    ])
    const fs = FileSystem.makeNoop({
      exists: path => Effect.succeed(files.has(path)),
      stat: path => Effect.succeed({ type: files.get(path)!.type, uid: Option.some(files.get(path)!.uid), mode: 0 } as FileSystem.File.Info),
      realPath: path => Effect.succeed(path),
      readFileString: path => path === "/etc/login.defs" ? Effect.succeed(debianDefinitions) : Effect.die(path),
      makeDirectory: path => Effect.sync(() => { files.set(path, { type: "Directory", uid: 0 }); written.push(`mkdir ${path}`) }),
      chmod: (path, mode) => Effect.sync(() => { written.push(`chmod ${mode.toString(8)} ${tmp(path)}`) }),
      chown: (path, uid, gid) => Effect.sync(() => { written.push(`chown ${uid}:${gid} ${tmp(path)}`) }),
      writeFileString: (path, _, writeOptions) => Effect.sync(() => {
        files.set(path, { type: "File", uid: 0 })
        written.push(`write ${(writeOptions?.mode ?? 0).toString(8)} ${tmp(path)}`)
      }),
      rename: (from, to) => Effect.sync(() => { files.delete(from); files.set(to, { type: "File", uid: 0 }); written.push(`rename ${to}`) }),
      // Like Node's fs.rm, a directory needs `recursive`.
      remove: (path, removeOptions) => files.get(path)?.type === "Directory" && !removeOptions?.recursive && [...files.keys()].some(file => file.startsWith(`${path}/`))
        ? Effect.fail(new SystemError({ reason: "Unknown", module: "FileSystem", method: "remove", pathOrDescriptor: path, description: "ENOTEMPTY" }))
        : Effect.sync(() => { if (files.delete(path) || !removeOptions?.force) written.push(`remove ${tmp(path)}`) }),
      readDirectory: path => Effect.succeed([...files.keys()].filter(file => file.startsWith(`${path}/`)).map(file => file.slice(path.length + 1))),
    })
    const respond = (line: string): readonly [number, string, string] => {
      if (options.fail?.test(line)) return [1, "", `${line.split(" ")[0]}: refused`]
      const lookup = /^\/usr\/bin\/getent (group|passwd) magnitude$/.exec(line)
      if (lookup) return accounts[lookup[1] as "group" | "passwd"] === "" ? [2, "", ""] : [0, `${accounts[lookup[1] as "group" | "passwd"]}\n`, ""]
      if (line === "/usr/bin/id -u -- ada") return [0, "1000\n", ""]
      if (line === "/usr/bin/id -u -- magnitude") return [0, "990\n", ""]
      if (line.startsWith("/usr/sbin/groupadd")) accounts.group = "magnitude:x:990:"
      if (line.startsWith("/usr/sbin/useradd")) accounts.passwd = ours.passwd
      if (line === "/usr/sbin/userdel magnitude") {
        accounts.passwd = ""
        return [0, "", "userdel: group magnitude not removed because it has other members.\n"]
      }
      if (line === "/usr/sbin/groupdel magnitude") accounts.group = ""
      if (line.endsWith("enable --now magnitude.service")) enabled = true
      if (line.endsWith("disable --now magnitude.service")) enabled = false
      if (line.endsWith("is-enabled --quiet magnitude.service")) return [enabled ? 0 : 1, "", ""]
      return [0, "", ""]
    }
    const executor = fakeExecutor(commands, respond)
    const changes = () => commands.filter(line => !/getent|id -u|is-enabled|systemctl cat/.test(line))
    return { commands, changes, written, printed, files, accounts, enabled: () => enabled,
      run: <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor>) => Effect.runPromise(Effect.either(
        effect.pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(CommandExecutor.CommandExecutor, executor)))) }
  }

  it("creates the account, data, validated sudoers rule and unit, adds no one to the group, and writes the marker last", async () => {
    const m = machine()
    expect(Either.isRight(await m.run(installLinuxServer))).toBe(true)
    expect(m.changes()).toEqual([
      "/usr/sbin/groupadd --system magnitude",
      "/usr/sbin/useradd --system --gid magnitude --home-dir /var/lib/magnitude --no-create-home --shell /usr/sbin/nologin --comment Magnitude server magnitude",
      "/usr/bin/chown -R -h -- magnitude:magnitude /var/lib/magnitude",
      "/usr/sbin/visudo -cf /etc/sudoers.d/<tmp>",
      "/usr/bin/systemctl daemon-reload", "/usr/bin/systemctl enable --now magnitude.service",
    ])
    expect(m.commands.some(line => line.includes("usermod"))).toBe(false)
    expect(m.written).toEqual([
      "mkdir /var/lib/magnitude", "chmod 711 /var/lib/magnitude",
      "write 440 /etc/sudoers.d/<tmp>", "chown 0:0 /etc/sudoers.d/<tmp>", "chmod 440 /etc/sudoers.d/<tmp>", "rename /etc/sudoers.d/magnitude",
      "mkdir /etc/magnitude", "chmod 755 /etc/magnitude", "write 644 /etc/magnitude/<tmp>", "chmod 644 /etc/magnitude/<tmp>", "rename /etc/magnitude/server",
    ])
  })
  it("refuses a person's account named magnitude and changes nothing", async () => {
    const m = machine({ accounts: human })
    const result = await m.run(installLinuxServer)
    expect(Either.isLeft(result) && result.left.message).toContain("isn't the system account Magnitude creates")
    expect(m.changes()).toEqual([])
    expect(m.written).toEqual([])
  })
  it("refuses a foreign group named magnitude without an account", async () => {
    const m = machine({ accounts: { passwd: "", group: "magnitude:x:1000:" } })
    expect(Either.isLeft(await m.run(installLinuxServer))).toBe(true)
    expect(m.changes()).toEqual([])
  })
  it("finishes a complete or partial setup without recreating the account or retaking the data", async () => {
    const m = machine({ setUp: true })
    expect(Either.isRight(await m.run(installLinuxServer))).toBe(true)
    expect(m.changes().some(line => /useradd|groupadd|chown/.test(line))).toBe(false)
    expect(m.written.at(-1)).toBe("rename /etc/magnitude/server")
  })
  it("gives root-owned kept data back to the new service account", async () => {
    const m = machine({ dataUid: 0 })
    expect(Either.isRight(await m.run(installLinuxServer))).toBe(true)
    expect(m.changes()).toContain("/usr/bin/chown -R -h -- magnitude:magnitude /var/lib/magnitude")
    expect(m.written).not.toContain("mkdir /var/lib/magnitude")
  })
  it("grants nothing without sudo, says so, and still sets up the service", async () => {
    const m = machine({ sudo: false })
    expect(Either.isRight(await m.run(installLinuxServer))).toBe(true)
    expect(m.written.some(line => line.includes("sudoers"))).toBe(false)
    expect(m.printed.join("\n")).toContain("sudo isn't installed")
  })
  it("undoes everything it created when the service cannot start, and writes no marker", async () => {
    const m = machine({ fail: /enable --now/ })
    const result = await m.run(installLinuxServer)
    expect(Either.isLeft(result) && result.left.message).toBe("Could not start magnitude.service.\n/usr/bin/systemctl: refused\nSetup undid the changes it had made. Run `magnitude server setup` again once that's fixed.")
    expect(m.changes().slice(-5)).toEqual([
      "/usr/bin/systemctl enable --now magnitude.service", "/usr/bin/systemctl disable --now magnitude.service",
      "/usr/bin/chown -R -h -- root:root /var/lib/magnitude", "/usr/sbin/userdel magnitude", "/usr/sbin/groupdel magnitude",
    ])
    expect(m.written).toContain("remove /etc/sudoers.d/magnitude")
    expect(m.written).toContain("remove /var/lib/magnitude")
    expect(m.files.has("/etc/magnitude/server")).toBe(false)
    expect(m.accounts).toEqual({ passwd: "", group: "" })
  })
  it("never installs a sudoers rule visudo rejects", async () => {
    const m = machine({ fail: /visudo/ })
    expect(Either.isLeft(await m.run(installLinuxServer))).toBe(true)
    expect(m.written).not.toContain("rename /etc/sudoers.d/magnitude")
    expect(m.written).toContain("remove /etc/sudoers.d/<tmp>")
    expect(m.changes().some(line => line.includes("enable"))).toBe(false)
    expect(m.files.has("/etc/magnitude/server")).toBe(false)
  })
  it("keeps an earlier complete setup's rule and unit when a re-run fails", async () => {
    const m = machine({ setUp: true, fail: /daemon-reload/ })
    expect(Either.isLeft(await m.run(installLinuxServer))).toBe(true)
    expect(m.changes().some(line => /disable|userdel|groupdel/.test(line))).toBe(false)
    expect(m.files.has("/etc/sudoers.d/magnitude")).toBe(true)
  })

  it("removes the unit, rule, account and group, hands kept data to root, and the marker last", async () => {
    const m = machine({ setUp: true })
    expect(Either.isRight(await m.run(removeLinuxServer))).toBe(true)
    expect(m.changes()).toEqual([
      "/usr/bin/systemctl disable --now magnitude.service", "/usr/sbin/userdel magnitude", "/usr/sbin/groupdel magnitude",
      "/usr/bin/chown -R -h -- root:root /var/lib/magnitude",
    ])
    expect(m.written).toEqual(["remove /etc/sudoers.d/magnitude", "chmod 700 /var/lib/magnitude", "remove /etc/magnitude/server", "remove /etc/magnitude"])
    expect(m.printed).toEqual([])
  })
  it("keeps a person's account and group named magnitude, and says so", async () => {
    const m = machine({ setUp: true, accounts: human })
    expect(Either.isRight(await m.run(removeLinuxServer))).toBe(true)
    expect(m.changes()).toEqual(["/usr/bin/systemctl disable --now magnitude.service"])
    expect(m.accounts).toEqual(human)
    expect(m.printed).toEqual(["Kept the account magnitude: it isn't the system account Magnitude created."])
    expect(m.written).toEqual(["remove /etc/sudoers.d/magnitude", "remove /etc/magnitude/server", "remove /etc/magnitude"])
  })
  it("keeps the marker when the account cannot be removed, so removal can run again and finish", async () => {
    const m = machine({ setUp: true, fail: /userdel/ })
    const result = await m.run(removeLinuxServer)
    expect(Either.isLeft(result) && result.left.message).toBe("Could not remove the magnitude user.\n/usr/sbin/userdel: refused")
    expect(m.files.has("/etc/magnitude/server")).toBe(true)
    const again = machine({ accounts: ours })
    expect(Either.isRight(await again.run(removeLinuxServer))).toBe(true)
    expect(again.changes()).toEqual(["/usr/bin/systemctl disable --now magnitude.service", "/usr/sbin/userdel magnitude", "/usr/sbin/groupdel magnitude"])
  })
  it("only disables the packaged unit on a machine without a server", async () => {
    const m = machine()
    expect(Either.isRight(await m.run(removeLinuxServer))).toBe(true)
    expect(m.changes()).toEqual(["/usr/bin/systemctl disable --now magnitude.service"])
    expect(m.written).toEqual([])
  })
  it("finds a partial setup's remnants without the marker", async () => {
    expect(await machine({ accounts: ours }).run(hasLinuxServerRemnants)).toEqual(Either.right(true))
    expect(await machine({ accounts: human }).run(hasLinuxServerRemnants)).toEqual(Either.right(false))
    expect(await machine().run(hasLinuxServerRemnants)).toEqual(Either.right(false))
  })
})

describe("macOS LaunchDaemon", () => {
  const plist = macServerPlist({ cli: "/Applications/Magnitude.app/Contents/Resources/magnitude", user: "ada", home: "/Users/Ada & Co" })
  it("runs the app's CLI serve as the person at load, kept alive, and named for the app", () => {
    expect(plist).toContain("<key>Label</key><string>dev.magnitude.server</string>")
    expect(plist).toContain("<array><string>/Applications/Magnitude.app/Contents/Resources/magnitude</string><string>serve</string></array>")
    expect(plist).toContain("<key>UserName</key><string>ada</string>")
    expect(plist).toContain("<key>RunAtLoad</key><true/>")
    expect(plist).toContain("<key>KeepAlive</key><true/>")
    expect(plist).toContain("<key>AssociatedBundleIdentifiers</key><string>dev.magnitude.desktop</string>")
    expect(plist).not.toContain("dev.magnitude.acn")
  })
  it("logs to the person's ~/.magnitude/logs and escapes their home folder", () => {
    expect(plist).toContain("<key>StandardOutPath</key><string>/Users/Ada &amp; Co/.magnitude/logs/server.log</string>")
    expect(plist).not.toContain("Ada & Co")
  })
})

describe("macOS root step", () => {
  const machine = (options: { readonly registered: boolean; readonly bootstrap: boolean }) => {
    const commands: string[] = []
    const files = new Set(options.registered ? ["/Library/LaunchDaemons/dev.magnitude.server.plist"] : [])
    const fs = FileSystem.makeNoop({
      exists: path => Effect.succeed(files.has(path)),
      writeFileString: path => Effect.sync(() => { files.add(path) }),
      chmod: () => Effect.void,
      rename: (from, to) => Effect.sync(() => { files.delete(from); files.add(to) }),
      remove: path => Effect.sync(() => { files.delete(path) }),
    })
    const executor = fakeExecutor(commands, line => line === "/usr/bin/id -u -- ada" ? [0, "501\n", ""]
      : line.startsWith("/usr/bin/dscl") ? [0, "NFSHomeDirectory: /Users/ada\n", ""]
      : line.startsWith("/bin/launchctl bootstrap") && !options.bootstrap ? [5, "", "Bootstrap failed: 5: Input/output error"]
      : [0, "", ""])
    const run = Effect.gen(function* () {
      const fiber = yield* Effect.fork(Effect.either(installMacServer({ name: "ada", uid: 501 }, "/Applications/Magnitude.app/Contents/Resources/magnitude")))
      yield* TestClock.adjust("1 minute")
      return yield* Fiber.join(fiber)
    }).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(CommandExecutor.CommandExecutor, executor), Effect.provide(TestContext.TestContext))
    return { commands, files, run: () => Effect.runPromise(run) }
  }
  it("creates the server log private to the person", async () => {
    const m = machine({ registered: false, bootstrap: true })
    expect(Either.isRight(await m.run())).toBe(true)
    expect(m.commands).toContain("/usr/bin/sudo -u ada -- /bin/chmod 600 /Users/ada/.magnitude/logs/server.log")
    expect(m.files.has("/Library/LaunchDaemons/dev.magnitude.server.plist")).toBe(true)
  })
  it("removes a first registration that cannot start, so setup can run again", async () => {
    const m = machine({ registered: false, bootstrap: false })
    const result = await m.run()
    expect(Either.isLeft(result) && result.left.message).toContain("Setup removed the LaunchDaemon again.")
    expect(m.files.size).toBe(0)
  })
  it("keeps an earlier registration when a re-run cannot start it", async () => {
    const m = machine({ registered: true, bootstrap: false })
    expect(Either.isLeft(await m.run())).toBe(true)
    expect(m.files.has("/Library/LaunchDaemons/dev.magnitude.server.plist")).toBe(true)
  })
})

describe("sudoers rule", () => {
  it("grants only the service account the installed update helper", () => {
    expect(SERVER_SUDOERS_RULE.split("\n").filter(line => line !== "" && !line.startsWith("#")))
      .toEqual(["magnitude ALL=(root) NOPASSWD: /usr/lib/magnitude-desktop/resources/magnitude _install-application-update *"])
  })
  it("passes visudo", () => {
    const visudo = ["/usr/sbin/visudo", "/usr/bin/visudo"].find(path => spawnSync(path, ["-V"]).status === 0)
    if (!visudo) return
    const directory = mkdtempSync(join(tmpdir(), "magnitude-sudoers-"))
    writeFileSync(join(directory, "magnitude"), SERVER_SUDOERS_RULE, { mode: 0o440 })
    expect(spawnSync(visudo, ["-cf", join(directory, "magnitude")]).status).toBe(0)
  })
})
