import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { readFileSync, statSync } from "node:fs"
import { parse } from "yaml"
import { expect, test } from "vitest"
import { LinuxInitialization, linuxInitialization } from "../src/providers/linux-initialization"
import { checkedCommand, ProcessExecutorLive } from "../src/process"

const download = { url: "https://example.com/runtime?cap=fixture-secret", sha256: "a".repeat(64), bytes: 1024 }
const config = { distribution: { os: "ubuntu", version: "24.04" }, adminUsername: "labworker", architecture: "x64", runtime: download, node: download, rustup: download }
test("renders private cloud-init inputs and syntax-valid native setup without interpolating capabilities into shell", async () => {
  const output = await Effect.runPromise(linuxInitialization(Schema.decodeUnknownSync(LinuxInitialization)(config)).pipe(Effect.provide(BunContext.layer)))
  expect(output.startsWith("#cloud-config\n")).toBe(true)
  const parsed = parse(output)
  expect(parsed.runcmd).toEqual([["/bin/bash", "/opt/magnitude-lab-initialize.sh"]])
  expect(parsed.write_files[0].permissions).toBe("0600")
  expect(JSON.parse(parsed.write_files[0].content).runtime).toEqual(download)
  expect(parsed.write_files[1].content).not.toContain("fixture-secret")
  expect(parsed.write_files[1].content).toContain("--frozen-lockfile")
  expect(parsed.write_files[1].content).toContain("outward-worker.ts")
  await Effect.runPromise(checkedCommand("/bin/bash", ["-n", "-c", parsed.write_files[1].content]).pipe(Effect.provide(ProcessExecutorLive)))
  // Check the embedded provisioning Python without executing it on the developer's host.
  const script = readFileSync(new URL("../infra/linux-worker.sh", import.meta.url), "utf8")
  const python = script.split("python3 - <<'PY'\n")[1]!.split("\nPY\n")[0]!
  await Effect.runPromise(checkedCommand("python3", ["-c", "import ast,sys; ast.parse(sys.argv[1])", python]).pipe(Effect.provide(ProcessExecutorLive)))
})
test("rejects non-HTTPS or credential-bearing download origins and unbounded inputs", () => {
  for (const url of ["http://example.com", "https://user:pass@example.com", "https://example.com/#fragment"])
    expect(() => Schema.decodeUnknownSync(LinuxInitialization)({ ...config, runtime: { ...download, url } })).toThrow()
  expect(() => Schema.decodeUnknownSync(LinuxInitialization)({ ...config, runtime: { ...download, bytes: 1024 ** 3 + 1 } })).toThrow()
  expect(() => Schema.decodeUnknownSync(LinuxInitialization)({ ...config, adminUsername: "root; command" })).toThrow()
  expect(() => Schema.decodeUnknownSync(LinuxInitialization)({ ...config, distribution: { os: "debian", version: "24.04" } })).toThrow()
  expect(() => Schema.decodeUnknownSync(LinuxInitialization)({ ...config, distribution: { os: "alpine", version: "13" } })).toThrow()
})

for (const [os, version] of [["debian", "13"], ["fedora", "44"], ["redhat", "10"]] as const) test(`${os} preparation preserves its identity and rejects an incorrectly supplied guest`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-distribution-" })
  const settings = yield* Schema.decodeUnknown(LinuxInitialization)({ ...config, distribution: { os, version } })
  const rendered = parse(yield* linuxInitialization(settings))
  const prepared = yield* Schema.decodeUnknown(Schema.parseJson(LinuxInitialization))(rendered.write_files[0].content)
  expect(prepared.distribution).toEqual({ os, version })
  yield* fs.writeFileString(`${root}/config.json`, rendered.write_files[0].content)
  const check = rendered.write_files[1].content.split("<<'CHECK'\n")[1].split("\nCHECK\n")[0]
    .replace("'/etc/magnitude-lab-initialization.json'", "sys.argv[3]").replace("sys.argv[1:]", "sys.argv[1:3]")
  for (const [guestOs, guestVersion, expected] of [[os === "redhat" ? "rhel" : os, version, "Right"], [os === "redhat" ? "rhel" : os, `${version}.2`, os === "redhat" ? "Right" : "Left"], ["ubuntu", "24.04", "Left"], [os, "12", "Left"]]) {
    const result = yield* checkedCommand("python3", ["-c", check, guestOs!, guestVersion!, `${root}/config.json`]).pipe(Effect.either)
    expect(result._tag).toBe(expected)
  }
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))

test("the generated guest launcher preserves package directory permissions", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-launcher-permissions-" })
  // Execute only the actual launcher expression from provisioning. Stub the display boundary,
  // so this checks the inherited modes without installing tools or starting a desktop locally.
  const script = readFileSync(new URL("../infra/linux-worker.sh", import.meta.url), "utf8")
  const python = script.split("python3 - <<'PY'\n")[1]!.split("\nPY\n")[0]!
  const rendered = yield* checkedCommand("python3", ["-c", `import ast,pathlib,shlex,sys
tree=ast.parse(sys.argv[1])
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('display_command','launcher') for t in n.targets)]
config={'distribution':{'os':'ubuntu'}}
root=home=workspace=pathlib.Path(sys.argv[2])
path=str(root)+':/usr/bin:/bin'
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<launcher>','exec'))
print(launcher)`, python, root])
  yield* fs.writeFileString(`${root}/xvfb-run`, "#!/bin/sh\nmkdir package-control\numask\n", { mode: 0o755 })
  const observed = yield* checkedCommand("/bin/bash", ["-c", `umask 077\n${rendered.stdout}`])
  expect(observed.stdout.trim()).toBe("0022")
  expect(statSync(`${root}/package-control`).mode & 0o777).toBe(0o755)
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
