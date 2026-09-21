import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { X509Certificate } from "node:crypto"
import { join } from "node:path"
import { expect, test } from "vitest"
import { checkedCommand, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { createWindowsUpdatePublisher, trustWindowsUpdatePublisher, validateWindowsUpdatePublisher, WindowsUpdatePublisher } from "../src/windows-update-publisher"

test("Windows publisher transfers only its public certificate and scopes creation and consumer trust cleanup", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-publisher-" })
  yield* checkedCommand("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes",
    "-days", "1", "-subj", "/O=Magnitude Update Acceptance/CN=Magnitude Update Acceptance", "-keyout", join(root, "key.pem"), "-out", join(root, "cert.pem")])
  const cert = new X509Certificate(yield* fs.readFileString(join(root, "cert.pem")))
  const publisher = { thumbprint: cert.fingerprint.replaceAll(":", ""), certificate: cert.raw.toString("base64") }
  yield* validateWindowsUpdatePublisher(publisher)
  const altered = yield* validateWindowsUpdatePublisher({ ...publisher, thumbprint: "0".repeat(40) }).pipe(Effect.either)
  expect(altered._tag).toBe("Left")
  const calls: { script: string; stores: string | undefined }[] = []
  const executor = ProcessExecutor.of({ run: spec => Effect.gen(function* () {
    const script = spec.args.at(-1)!
    calls.push({ script, stores: spec.env.LAB_PUBLISHER_STORES })
    return { exitCode: 0, stderr: "", stdout: script.includes("New-SelfSignedCertificate")
      ? yield* Schema.encode(Schema.parseJson(WindowsUpdatePublisher))(publisher) : "" }
  }).pipe(Effect.orDie) })
  yield* Effect.scoped(createWindowsUpdatePublisher).pipe(Effect.provideService(ProcessExecutor, executor))
  expect(calls[1]?.stores).toBe("Root,My")
  expect(calls[1]?.script).toContain("-DeleteKey")
  yield* Effect.scoped(trustWindowsUpdatePublisher(publisher)).pipe(Effect.provideService(ProcessExecutor, executor))
  expect(calls[2]?.script).toContain("HasPrivateKey")
  expect(calls[3]?.stores).toBe("Root")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
