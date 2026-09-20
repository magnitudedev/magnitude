import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Schema } from "effect"
import { expect, test } from "vitest"
import { fileArtifactStore } from "../src/artifact-store"
import { LeaseId, RunId } from "../src/domain"
import { NamespaceMachine } from "../src/machines"
import { ProcessExecutor, type CommandSpec } from "../src/process"
import { NamespacePreparation, prepareNamespaceMachine } from "../src/providers/namespace-preparation"
import { sha256 } from "../src/snapshot"

for (const workKind of ["build", "test"] as const) for (const mode of ["ready", "failed", "changed-script", "expired"] as const) {
  test(`Mac ${workKind} preparation ${mode} preserves identity, private delivery and failure evidence`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-mac-preparation-test-" })
    const setup = "#!/bin/bash\nexit 0\n"
    yield* fs.writeFileString(`${directory}/setup.sh`, mode === "changed-script" ? "changed" : setup)
    const digest = "a".repeat(64)
    const download = { url: "https://example.com/tool", bytes: 100, sha256: digest }
    const recipe = yield* Schema.decodeUnknown(NamespacePreparation)({ setup: { file: `${directory}/setup.sh`, sha256: sha256(setup) },
      runtime: { account: "labaccount", container: "artifacts", blob: `worker-runtime/${digest}.tar.gz`, sha256: digest, bytes: 100 },
      node: download, rustup: download, tirith: download, azureExecutable: "az", subscription: "subscription" })
    const machine = NamespaceMachine.make({ provider: "namespace", id: "box", name: "owned-mac", tags: { schemaVersion: 1,
      leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), expiresAt: DateTime.unsafeMake(Date.now() + (mode === "expired" ? -1 : 600_000)) } })
    const image = { version: "15", selector: "sequoia", catalogCreatedAt: "locked", productVersion: "15.7.5", buildVersion: "24G624" }
    const commands: CommandSpec[] = []
    let delivered = ""
    const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.gen(function* () {
      commands.push(spec)
      expect(spec.args.join(" ")).not.toContain("secret-signature")
      if (spec.executable === "az") return { exitCode: 0, stderr: "", stdout: `https://labaccount.blob.core.windows.net/artifacts/worker-runtime/${digest}.tar.gz?sp=r&sr=b&spr=https&skoid=owner&sktid=tenant&se=${encodeURIComponent(spec.args[spec.args.indexOf("--expiry")+1]!)}&sig=secret-signature` }
      expect(spec.args[1]).toBe("owned-mac")
      if (spec.args[0] === "upload" && spec.args[2]!.endsWith("config.json")) {
        delivered = yield* fs.readFileString(spec.args[2]!)
        expect(Number((yield* fs.stat(spec.args[2]!)).mode) & 0o777).toBe(0o600)
      }
      if (spec.args.includes("/usr/bin/tail")) return { exitCode: 0, stdout: "Native compiler failure https://example.com/a?sig=secret-signature", stderr: "" }
      if (mode === "failed" && spec.args.includes("lab-prepare")) return { exitCode: 1, stdout: "", stderr: "Preparation failed" }
      return { exitCode: 0, stdout: "ready", stderr: "" }
    }).pipe(Effect.orDie) })
    yield* Effect.gen(function* () {
      const result = yield* prepareNamespaceMachine("devbox", recipe, machine, image, workKind).pipe(Effect.provide(executor), Effect.either)
      expect(result._tag).toBe(mode === "ready" ? "Right" : "Left")
      if (mode === "changed-script" || mode === "expired") { expect(commands).toHaveLength(0); return }
      expect(delivered).toContain("secret-signature")
      const deliveredConfig = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ workKind: Schema.Literal("build", "test") })))(delivered)
      expect(deliveredConfig.workKind).toBe(workKind)
      expect(commands.some(c => c.args.includes("/bin/rm") && c.args.some(a=>a.endsWith("/config.json")))).toBe(true)
      if (result._tag === "Left") {
        expect(result.left.evidence._tag).toBe("Some")
        if (result.left.evidence._tag === "Some") {
          const evidence = result.left.evidence.value[0]!
          const text = yield* fs.readFileString(`${directory}/objects/${evidence.sha256}`)
          expect(text).toContain("Native compiler failure")
          expect(text).not.toContain("secret-signature")
          expect(sha256(text)).toBe(evidence.sha256)
        }
      }
    }).pipe(Effect.provide(fileArtifactStore(`${directory}/objects`)))
  })).pipe(Effect.provide(BunContext.layer))))
}
