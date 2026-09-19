import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { command, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { AppleTeamId, appleTrustPolicy, verifyAppleNotarization, verifyAppleSignature } from "../src/suites/apple-package-trust"

test("release signature policy cannot inherit a missing or malformed publisher", async () => {
  for (const team of [undefined, "", "wrong", 'ABCDEFGHIJ\" or true']) {
    expect((await Effect.runPromise(appleTrustPolicy(true, team).pipe(Effect.either)))._tag).toBe("Left")
  }
  expect(await Effect.runPromise(appleTrustPolicy(false, undefined))).toEqual({ kind: "development" })
  expect(await Effect.runPromise(appleTrustPolicy(true, "ABCDEFGHIJ"))).toEqual({ kind: "production", team: "ABCDEFGHIJ" })
})

for (const scenario of ["valid", "adhoc", "wrong-team", "no-timestamp", "empty-timestamp", "invalid-seal", "missing-identity", "wrong-identity"] as const) {
  test(`production signature verification: ${scenario}`, async () => {
    const calls: string[][] = []
    const result = await Effect.runPromise(verifyAppleSignature("/fixture.app", { kind: "production", team: AppleTeamId.make("ABCDEFGHIJ") }, "dev.magnitude.desktop").pipe(
      Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
        calls.push([...spec.args])
        if (spec.args.includes("--verify")) return { exitCode: scenario === "invalid-seal" ? 1 : 0, stdout: "", stderr: "" }
        return { exitCode: 0, stdout: "", stderr: [
          scenario === "missing-identity" ? "" : scenario === "wrong-identity" ? "Identifier=other.application" : "Identifier=dev.magnitude.desktop",
          scenario === "adhoc" ? "Signature=adhoc" : "Authority=Developer ID Application: Fixture",
          `TeamIdentifier=${scenario === "wrong-team" ? "KLMNOPQRST" : "ABCDEFGHIJ"}`,
          scenario === "no-timestamp" ? "" : scenario === "empty-timestamp" ? "Timestamp=" : "Timestamp=Sep 19, 2026 at 10:00:00 AM",
        ].join("\n") }
      }) }), Effect.either))
    expect(result._tag).toBe(scenario === "valid" ? "Right" : "Left")
    expect(calls[0]).toContain("--all-architectures")
    expect(calls[0]).toContain("--strict")
    expect(calls[0]!.join(" ")).toContain('certificate leaf[subject.OU] = "ABCDEFGHIJ"')
    if (scenario === "invalid-seal") expect(calls).toHaveLength(1)
  })
}

for (const rejection of ["none", "stapler", "gatekeeper"] as const) test(`notarization requires both OS checks: ${rejection}`, async () => {
  const calls: string[] = []
  const result = await Effect.runPromise(verifyAppleNotarization("/fixture.app").pipe(Effect.provideService(ProcessExecutor, {
    run: spec => Effect.sync(() => {
      const check = spec.executable.endsWith("xcrun") ? "stapler" : "gatekeeper"
      calls.push(check)
      return { exitCode: rejection === check ? 1 : 0, stdout: "", stderr: rejection === check ? "rejected fixture" : "" }
    }),
  }), Effect.either))
  expect(result._tag).toBe(rejection === "none" ? "Right" : "Left")
  expect(calls).toEqual(rejection === "stapler" ? ["stapler"] : ["stapler", "gatekeeper"])
})

test.skipIf(process.platform !== "darwin")("native signature verification rejects altered signed bytes without changing the original", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-signature-" })
  const binary = join(root, "fixture")
  yield* fs.copyFile("/usr/bin/true", binary)
  const signed = yield* command("/usr/bin/codesign", ["--force", "--sign", "-", "--identifier", "dev.magnitude.signature-fixture", binary])
  expect(signed.exitCode).toBe(0)
  const receipt = yield* verifyAppleSignature(binary, { kind: "development" })
  expect(receipt.kind).toBe("adhoc")
  expect(Option.isNone(receipt.team)).toBe(true)
  const bytes = yield* fs.readFile(binary)
  expect(bytes.length).toBeGreaterThan(4096)
  bytes[4096] = bytes[4096]! ^ 1
  yield* fs.writeFile(binary, bytes)
  expect((yield* verifyAppleSignature(binary, { kind: "development" }).pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
