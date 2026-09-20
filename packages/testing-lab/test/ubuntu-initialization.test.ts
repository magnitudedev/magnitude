import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { readFileSync, statSync } from "node:fs"
import { parse } from "yaml"
import { expect, test } from "vitest"
import { UbuntuInitialization, ubuntuInitialization } from "../src/providers/ubuntu-initialization"
import { checkedCommand, ProcessExecutorLive } from "../src/process"

const download = { url: "https://example.com/runtime?cap=fixture-secret", sha256: "a".repeat(64), bytes: 1024 }
const config = { adminUsername: "labworker", architecture: "x64", runtime: download, node: download, rustup: download }
test("renders private cloud-init inputs and syntax-valid native setup without interpolating capabilities into shell", async () => {
  const output = await Effect.runPromise(ubuntuInitialization(Schema.decodeUnknownSync(UbuntuInitialization)(config)).pipe(Effect.provide(BunContext.layer)))
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
  const script = readFileSync(new URL("../infra/ubuntu-worker.sh", import.meta.url), "utf8")
  const python = script.split("python3 - <<'PY'\n")[1]!.split("\nPY\n")[0]!
  await Effect.runPromise(checkedCommand("python3", ["-c", "import ast,sys; ast.parse(sys.argv[1])", python]).pipe(Effect.provide(ProcessExecutorLive)))
})
test("rejects non-HTTPS or credential-bearing download origins and unbounded inputs", () => {
  for (const url of ["http://example.com", "https://user:pass@example.com", "https://example.com/#fragment"])
    expect(() => Schema.decodeUnknownSync(UbuntuInitialization)({ ...config, runtime: { ...download, url } })).toThrow()
  expect(() => Schema.decodeUnknownSync(UbuntuInitialization)({ ...config, runtime: { ...download, bytes: 1024 ** 3 + 1 } })).toThrow()
  expect(() => Schema.decodeUnknownSync(UbuntuInitialization)({ ...config, adminUsername: "root; command" })).toThrow()
})

test("the generated guest launcher preserves package directory permissions", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-launcher-permissions-" })
  // Execute only the actual launcher expression from provisioning. Stub the display boundary,
  // so this checks the inherited modes without installing tools or starting a desktop locally.
  const script = readFileSync(new URL("../infra/ubuntu-worker.sh", import.meta.url), "utf8")
  const python = script.split("python3 - <<'PY'\n")[1]!.split("\nPY\n")[0]!
  const rendered = yield* checkedCommand("python3", ["-c", `import ast,pathlib,shlex,sys
tree=ast.parse(sys.argv[1])
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='launcher' for t in n.targets))
root=home=workspace=pathlib.Path(sys.argv[2])
path=str(root)+':/usr/bin:/bin'
exec(compile(ast.Module(body=[node],type_ignores=[]),'<launcher>','exec'))
print(launcher)`, python, root])
  yield* fs.writeFileString(`${root}/xvfb-run`, "#!/bin/sh\nmkdir package-control\numask\n", { mode: 0o755 })
  const observed = yield* checkedCommand("/bin/bash", ["-c", `umask 077\n${rendered.stdout}`])
  expect(observed.stdout.trim()).toBe("0022")
  expect(statSync(`${root}/package-control`).mode & 0o777).toBe(0o755)
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
