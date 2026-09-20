import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Schema } from "effect"
import { expect, test } from "vitest"
import { prepareWindowsMachine, windowsPreparationDiagnostics } from "../src/providers/azure-windows-preparation"
import { WindowsAzureInitialization } from "../src/providers/azure-initialization"
import { windowsToolDownloads } from "../src/providers/windows-tools"
import { AzureMachine, MachineTags } from "../src/machines"
import { Digest, InfrastructureFailure, LeaseId, RunId } from "../src/domain"
import { ProcessExecutor } from "../src/process"

for (const mode of ["fresh", "resume", "lost-put", "lost-refresh", "failed", "missing-exit", "foreign-stage", "foreign-vm", "expired", "diagnostics"] as const) {
  test(`Windows native preparation reconciliation: ${mode}`, () => Effect.runPromise(Effect.gen(function* () {
    const digest = Digest.make("a".repeat(64)), subscription = "5304c4b3-d605-4193-b0cb-766c065acfa6"
    const machine = AzureMachine.make({ provider: "azure", name: "ml-123456789abc",
      id: `/subscriptions/${subscription}/resourceGroups/magnitude-ci/providers/Microsoft.Compute/virtualMachines/ml-123456789abc`,
      tags: MachineTags.make({ schemaVersion: 1, runId: RunId.make(`run-${crypto.randomUUID()}`), leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`),
        expiresAt: DateTime.unsafeMake(Date.now() + (mode === "expired" ? -1 : 3600_000)) }) })
    const pin = { file: "unused", sha256: digest }
    const recipe = WindowsAzureInitialization.make({ kind: "windows", toolsSetup: pin, runtimeSetup: pin, desktopSetup: pin,
      downloads: yield* windowsToolDownloads, distribution: { os: "windows", version: "11" }, architecture: "x64", adminUsername: "labworker",
      runtime: { account: "labaccount", container: "artifacts", blob: `worker-runtime/${digest}.tar.gz`, sha256: digest, bytes: 100 } })
    const prepared = { kind: "windows" as const, identity: digest, recipe, toolsScript: "tools", runtimeScript: "runtime", desktopScript: "desktop" }
    const commands = new Map<string, unknown>()
    const names = ["lab-tools", "lab-runtime", "lab-desktop", "lab-ready"]
    const vmTags: Record<string, string> = { "lab-initialization": mode === "foreign-vm" ? "b".repeat(64) : digest, "lab-owner": "preserve-me" }
    if (mode === "resume" || mode === "lost-refresh" || mode === "diagnostics") {
      names.forEach(name => commands.set(name, {})); vmTags["lab-desktop-restart"] = digest
    }
    let restarts = 0, grants = 0, puts = 0
    const process = Layer.succeed(ProcessExecutor, { run: spec => Effect.sync(() => {
      grants++
      expect(spec.args[0]).toBe("storage")
      expect(commands.has("lab-tools")).toBe(true)
      const query = new URLSearchParams({ sp: "r", sr: "b", spr: "https", sig: "private-capability", skoid: "object", sktid: "tenant", se: spec.args[spec.args.indexOf("--expiry") + 1]! })
      return { exitCode: 0, stderr: "", stdout: `https://labaccount.blob.core.windows.net/artifacts/worker-runtime/${digest}.tar.gz?${query}` }
    }) })
    const operations = {
      restart: Effect.sync(() => { expect(vmTags["lab-desktop-restart"]).toBe(digest); restarts++ }),
      waitProvisioned: Effect.void,
      rest: (method: string, id: string, _version: string, body?: unknown) => Effect.gen(function* () {
        if (id === machine.id) {
          if (method === "PATCH") {
            const request = yield* Schema.decodeUnknown(Schema.Struct({ tags: Schema.Record({ key: Schema.String, value: Schema.String }) }))(body)
            expect(request.tags["lab-owner"]).toBe("preserve-me")
            Object.assign(vmTags, request.tags)
          }
          return { stdout: JSON.stringify({ tags: vmTags }) }
        }
        if (id.endsWith("/runCommands")) return { stdout: JSON.stringify({ value: [...commands.keys()].map(name => ({ id: `${machine.id}/runCommands/${name}` })) }) }
        const name = id.split("/").at(-1)!
        if (method === "PUT") {
          puts++
          const request = yield* Schema.decodeUnknown(Schema.Struct({ properties: Schema.Struct({ source: Schema.Struct({ script: Schema.String }),
            timeoutInSeconds: Schema.Number, protectedParameters: Schema.Array(Schema.Struct({ name: Schema.String, value: Schema.String })) }) }))(body)
          expect(request.properties.timeoutInSeconds).toBeGreaterThan(0)
          expect(request.properties.source.script).not.toContain("private-capability")
          if (name === "lab-runtime") expect(Buffer.from(request.properties.protectedParameters[0]!.value, "base64").toString()).toContain("private-capability")
          if (mode === "lost-refresh") return yield* new InfrastructureFailure({ operation: "fixture", message: "readiness update was not accepted" })
          commands.set(name, body)
          if (mode === "lost-put") return yield* new InfrastructureFailure({ operation: "fixture", message: "response lost after acceptance" })
          return { stdout: "{}" }
        }
        expect(commands.has(name)).toBe(true)
        const stored = commands.get(name) as { tags?: Record<string, string> }
        return { stdout: JSON.stringify({ tags: { ...stored.tags, "lab-initialization": mode === "foreign-stage" ? "b".repeat(64) : digest }, properties: { instanceView: {
          executionState: mode === "failed" ? "Failed" : "Succeeded", ...(mode === "missing-exit" ? {} : { exitCode: mode === "failed" ? 1 : 0 }),
          output: "native output", error: "native error", protectedParameters: "not retained",
        }, source: "not retained", protectedParameters: "not retained" } }) }
      }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : new InfrastructureFailure({ operation: "fixture", message: "invalid request" }))),
    }
    if (mode === "diagnostics") {
      const retained = yield* windowsPreparationDiagnostics(machine, operations.rest)
      expect(retained).toContain("lab-runtime: Succeeded")
      expect(retained).toContain("native output")
      expect(retained).not.toContain("not retained")
      return
    }
    const result = yield* prepareWindowsMachine(machine, prepared, { executable: "az", subscription, location: "westus2" }, operations).pipe(Effect.provide(process), Effect.either)
    const success = ["fresh", "resume", "lost-put"].includes(mode)
    expect(result._tag).toBe(success ? "Right" : "Left")
    expect(puts).toBe(mode === "resume" ? 1 : mode === "foreign-vm" || mode === "expired" ? 0 : success ? 4 : 1)
    expect(restarts).toBe(success && mode !== "resume" ? 1 : 0)
    expect(grants).toBe(success && mode !== "resume" ? 1 : 0)
  }).pipe(Effect.provide(BunContext.layer))))
}
