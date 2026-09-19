import { Effect } from "effect"
import { expect, test } from "vitest"
import { ProcessExecutor } from "../src/process"
import { verifyWindowsSignature, windowsTrustPolicy, WindowsPublisher } from "../src/suites/windows-package-trust"

test("production Windows policy requires explicit publisher and verifier", async () => {
  for (const environment of [{}, { LAB_EXPECTED_WINDOWS_PUBLISHER: "Fixture" }, { LAB_EXPECTED_WINDOWS_PUBLISHER: " ", LAB_WINDOWS_SIGNTOOL: "signtool.exe" }]) {
    expect((await Effect.runPromise(windowsTrustPolicy(true, environment as Readonly<Record<string, string>>).pipe(Effect.either)))._tag).toBe("Left")
  }
  expect(await Effect.runPromise(windowsTrustPolicy(false, {}))).toEqual({ kind: "development" })
})

for (const scenario of ["valid", "unsigned", "wrong-publisher", "no-timestamp", "hash-mismatch", "untrusted", "invalid-json", "missing-publisher", "missing-signature-type", "signtool-rejects", "inspection-failed"] as const) {
  test(`Windows production signature checks: ${scenario}`, async () => {
    const commands: string[] = []
    const path = "C:\\fixture $name's [candidate].exe"
    const result = await Effect.runPromise(verifyWindowsSignature(path, { kind: "production", publisher: WindowsPublisher.make("Magnitude Fixture"), signtool: "signtool.exe" }, { SystemRoot: "C:\\Windows" }).pipe(
      Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
        commands.push(spec.executable)
        if (spec.executable === "signtool.exe") {
          expect(spec.args).toEqual(["verify", "/pa", "/all", "/tw", path])
          return { exitCode: scenario === "signtool-rejects" ? 1 : 0, stdout: "", stderr: "" }
        }
        expect(spec.env.LAB_SIGNATURE_PATH).toBe(path)
        expect(spec.args.join(" ")).not.toContain(path)
        expect(spec.inheritEnv).toBe(false)
        return { exitCode: scenario === "inspection-failed" ? 1 : 0, stderr: "", stdout: scenario === "invalid-json" ? "unparseable response" : JSON.stringify({
          status: scenario === "unsigned" ? "NotSigned" : scenario === "hash-mismatch" ? "HashMismatch" : scenario === "untrusted" ? "NotTrusted" : "Valid",
          ...(scenario === "unsigned" || scenario === "missing-publisher" ? {} : { publisher: scenario === "wrong-publisher" ? "Someone Else" : "Magnitude Fixture" }),
          signatureType: scenario === "unsigned" || scenario === "missing-signature-type" ? "None" : "Authenticode",
          timestamped: scenario !== "unsigned" && scenario !== "no-timestamp",
        }) }
      }) }), Effect.either))
    expect(result._tag).toBe(scenario === "valid" ? "Right" : "Left")
    expect(commands).toEqual(scenario === "valid" || scenario === "signtool-rejects" ? ["powershell.exe", "signtool.exe"] : ["powershell.exe"])
  })
}

for (const status of ["NotSigned", "Valid", "HashMismatch", "NotTrusted"] as const) test(`development records unsigned code but rejects invalid signatures: ${status}`, async () => {
  const result = await Effect.runPromise(verifyWindowsSignature("fixture.exe", { kind: "development" }, {}).pipe(
    Effect.provideService(ProcessExecutor, { run: () => Effect.succeed({ exitCode: 0, stderr: "", stdout: JSON.stringify({ status, signatureType: status === "NotSigned" ? "None" : "Authenticode",
      ...(status === "NotSigned" ? {} : { publisher: "Fixture" }), timestamped: status !== "NotSigned" }) }) }), Effect.either))
  expect(result._tag).toBe(status === "NotSigned" || status === "Valid" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.status).toBe(status)
})

test("only an explicitly selected Microsoft runtime exception accepts that publisher", async () => {
  for (const allowed of [false, true]) {
    const result = await Effect.runPromise(verifyWindowsSignature("vcruntime140.dll", { kind: "production", publisher: WindowsPublisher.make("Magnitude Fixture"), signtool: "signtool.exe" }, {}, allowed).pipe(
      Effect.provideService(ProcessExecutor, { run: spec => Effect.succeed({ exitCode: 0, stderr: "", stdout: spec.executable === "signtool.exe" ? ""
        : JSON.stringify({ status: "Valid", signatureType: "Authenticode", publisher: "Microsoft Windows Software Compatibility Publisher", timestamped: true }) }) }), Effect.either))
    expect(result._tag).toBe(allowed ? "Right" : "Left")
  }
})
