import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option, Redacted, Schema } from "effect"
import { parse } from "yaml"
import { expect, test } from "vitest"
import { prepareAzureInitialization, LinuxAzureInitialization, WindowsAzureInitialization } from "../src/providers/azure-initialization"
import { WindowsToolDownloads, windowsToolDownloads } from "../src/providers/windows-tools"
import { azureInitializationWait, azureInitializationDiagnosticsScript } from "../src/providers/azure-readiness"
import { LinuxInitialization } from "../src/providers/linux-initialization"
import { checkedCommand, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import driverPins from "../tools/nvidia-drivers.json"
import { sha256 } from "../src/snapshot"

for (const version of ["2022", "2025"] as const) for (const gpu of ["windows-a10", "windows-rtx-pro-6000"] as const) {
  test(`pinned Windows GPU recipe accepts Server ${version} with ${gpu} and rejects ARM64`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-server-gpu-" })
    const script = "Write-Output 'prepared'", file = `${root}/prepare.ps1`, digest = "a".repeat(64)
    yield* fs.writeFileString(file, script)
    const pin = { file, sha256: sha256(script) }
    const recipe = yield* Schema.decodeUnknown(WindowsAzureInitialization)({ kind: "windows", toolsSetup: pin, runtimeSetup: pin, desktopSetup: pin,
      downloads: yield* Schema.encode(WindowsToolDownloads)(yield* windowsToolDownloads),
      distribution: { os: "windows-server", version }, architecture: "x64", adminUsername: "labworker", gpu: driverPins[gpu],
      runtime: { account: "labaccount", container: "artifacts", blob: `worker-runtime/${digest}.tar.gz`, sha256: digest, bytes: 100 } })
    const scope = { executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", adminUsername: "labworker", architecture: "x64" as const, os: "windows-server", version }
    expect((yield* prepareAzureInitialization(recipe, scope)).kind).toBe("windows")
    expect((yield* prepareAzureInitialization(recipe, { ...scope, architecture: "arm64" }).pipe(Effect.either))._tag).toBe("Left")
  })).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
}

for (const mode of ["gpu", "unsupported-gpu", "renew", "wrong-blob", "write-permission", "expired", "foreign-account", "changed-script", "signing-failure", "wrong-distribution"] as const) {
  test(`runtime capability preparation: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-runtime-capability-" })
    const setup = "#!/bin/sh\nprintf ready\n", digest = "a".repeat(64)
    yield* fs.writeFileString(`${root}/setup.sh`, mode === "changed-script" ? "modified" : setup)
    const download = { url: "https://example.com/pinned-tool", bytes: 1024, sha256: digest }
    const recipe = yield* Schema.decodeUnknown(LinuxAzureInitialization)({ kind: "linux", distribution: { os: "ubuntu", version: "24.04" }, adminUsername: "labworker", architecture: "x64",
      ...(mode === "gpu" || mode === "unsupported-gpu" ? { gpu: driverPins["linux-a10"] } : {}),
      setup: { file: `${root}/setup.sh`, sha256: sha256(setup) }, node: download, rustup: download,
      runtime: { account: "labaccount", container: "artifacts", blob: `worker-runtime/${mode === "wrong-blob" ? "b".repeat(64) : digest}.tar.gz`, sha256: digest, bytes: 1024 },
    })
    let calls = 0
    const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.sync(() => {
      calls++
      const arg = (name: string) => spec.args[spec.args.indexOf(name) + 1]!
      expect(spec.args).toContain("--as-user")
      expect(arg("--subscription")).toBe("5304c4b3-d605-4193-b0cb-766c065acfa6")
      expect(arg("--permissions")).toBe("r")
      const query = new URLSearchParams({ sp: mode === "write-permission" ? "rw" : "r", sr: "b", spr: "https", skoid: "delegated-object", sktid: "delegated-tenant",
        sig: `private-signature-${calls}`, se: mode === "expired" ? "2000-01-01T00:00:00Z" : arg("--expiry") })
      return { exitCode: mode === "signing-failure" ? 1 : 0, stderr: "private-signature-error",
        stdout: `https://${mode === "foreign-account" ? "foreign" : "labaccount"}.blob.core.windows.net/artifacts/worker-runtime/${digest}.tar.gz?${query}` }
    }) })
    const prepare = prepareAzureInitialization(recipe, { executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", adminUsername: "labworker", architecture: "x64", os: (mode === "wrong-distribution" || mode === "unsupported-gpu") ? "debian" : "ubuntu", version: "24.04" }).pipe(Effect.provide(executor))
    const first = yield* prepare.pipe(Effect.either)
    if (mode !== "renew" && mode !== "gpu") {
      expect(first._tag).toBe("Left")
      if (first._tag === "Left") expect(first.left.message).not.toContain("private-signature")
      if (["unsupported-gpu", "wrong-blob", "changed-script", "wrong-distribution"].includes(mode)) expect(calls).toBe(0)
      return
    }
    if (first._tag === "Left") return yield* Effect.die(first.left)
    const second = yield* prepare
    expect(calls).toBe(2)
    if (first.right.kind !== "linux" || second.kind !== "linux") throw new Error("Expected Linux preparation")
    expect(first.right.identity).toBe(second.identity)
    expect(first.right.customData).not.toBe(second.customData)
    const rendered = parse(Buffer.from(second.customData, "base64").toString())
    const configuration = yield* Schema.decodeUnknown(Schema.parseJson(LinuxInitialization))(rendered.write_files[0].content)
    expect(configuration.runtime.sha256).toBe(digest)
    expect(Redacted.value(configuration.runtime.url)).toContain("private-signature-2")
    if (mode === "gpu") {
      expect(rendered.write_files[1].content).toContain("GPU driver integrity mismatch")
      expect(rendered.write_files[1].content).toContain("--query-gpu=name,driver_version")
    } else expect(rendered.write_files[1].content).toBe(setup)
  })).pipe(Effect.provide(BunContext.layer))))
}

for (const [exit, ready, passed] of [[0, true, true], [2, true, true], [1, true, false], [0, false, false], [2, false, false]] as const) {
  test(`cloud-init ${exit}, lab completion ${ready}: readiness ${passed}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-initialization-status-" })
    yield* fs.writeFileString(`${root}/cloud-init`, `#!/bin/sh\nprintf '%s\\n' '{"status":"done","errors":[],"recoverable_errors":{}}'\nexit ${exit}\n`, { mode: 0o755 })
    yield* fs.writeFileString(`${root}/runtime.json`, "{}")
    if (ready) yield* fs.writeFileString(`${root}/ready`, "ready\n")
    const observed = yield* checkedCommand("/bin/sh", ["-c", azureInitializationWait(root)], {
      cwd: Option.some(root), env: { PATH: `${root}:/usr/bin:/bin` }, inheritEnv: false,
    }).pipe(Effect.either)
    expect(observed._tag).toBe(passed ? "Right" : "Left")
    const retained = yield* fs.readFileString(`${root}/cloud-init-status.json`).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ status: Schema.String })))))
    expect(retained.status).toBe("done")
  })).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
}


test("initialization diagnostics preserve the native failure before a large Python command traceback", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-init-diagnostics-" })
  yield* fs.writeFileString(`${root}/cloud-init`, `#!/bin/sh\nprintf '%s\n' '{"status":"error","errors":["native preparation failed"]}'\nexit 1\n`, { mode: 0o755 })
  const log = `${root}/output.log`
  yield* fs.writeFileString(log, "earlier output\nNative dependency error: missing compiler\nTraceback (most recent call last):\n" + "inline script\n".repeat(2000))
  const result = yield* checkedCommand("/bin/sh", ["-c", azureInitializationDiagnosticsScript(log)], { env: { PATH: `${root}:/usr/bin:/bin` }, inheritEnv: false })
  expect(result.stdout).toContain("Native dependency error: missing compiler")
  expect(result.stdout).toContain('"status": "error"')
  expect(result.stdout).not.toContain("inline script")
  expect(result.stdout.length).toBeLessThan(3500)
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
