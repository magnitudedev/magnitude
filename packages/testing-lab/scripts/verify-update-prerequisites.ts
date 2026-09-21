import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect } from "effect"
import { createPrivateKey, X509Certificate } from "node:crypto"
import { InfrastructureFailure } from "../src/domain"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { updateFixture } from "../src/update-fixture"
import { createWindowsUpdatePublisher, trustWindowsUpdatePublisher } from "../src/windows-update-publisher"

/** Run before expensive compilation, in the same OS account as the real worker. */
const main = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-prerequisites-" })
  const authority = yield* Effect.scoped(Effect.gen(function* () {
    const fixture = yield* updateFixture(root)
    const certificate = new X509Certificate(fixture.authority.certificate)
    if (!certificate.ca || !certificate.checkPrivateKey(createPrivateKey(fixture.authority.tlsPrivateKey)) || certificate.checkIP("127.0.0.1") !== "127.0.0.1") {
      return yield* new InfrastructureFailure({ operation: "update-prerequisites", message: "Loopback TLS certificate or private key is invalid" })
    }
    const response = yield* Effect.tryPromise({ try: () => fetch(fixture.origin, { tls: { ca: fixture.authority.certificate }, redirect: "error", signal: AbortSignal.timeout(10_000) }),
      catch: () => new InfrastructureFailure({ operation: "update-prerequisites", message: "Private HTTPS fixture failed TLS verification" }) })
    if (response.status !== 404) return yield* new InfrastructureFailure({ operation: "update-prerequisites", message: "Unexpected private HTTPS response" })
    yield* Effect.tryPromise({ try: () => response.arrayBuffer(), catch: () => new InfrastructureFailure({ operation: "update-prerequisites", message: "Private HTTPS response failed" }) })
    return fixture.authority
  }))
  // A clean consumer restores the same origin and public trust after producer exit.
  yield* Effect.scoped(updateFixture(root, authority))
  if (process.platform === "win32") {
    const publisher = yield* Effect.scoped(createWindowsUpdatePublisher)
    yield* Effect.scoped(trustWindowsUpdatePublisher(publisher))
    const removed = yield* checkedCommand("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", String.raw`
if ((Test-Path -LiteralPath "Cert:\LocalMachine\Root\$env:LAB_PUBLISHER_THUMBPRINT") -or (Test-Path -LiteralPath "Cert:\CurrentUser\My\$env:LAB_PUBLISHER_THUMBPRINT")) { throw 'Temporary publisher remains after scope exit' }
`], { env: { LAB_PUBLISHER_THUMBPRINT: publisher.thumbprint }, timeoutMs: 30_000 })
    if (removed.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "update-prerequisites", message: "Publisher cleanup failed" })
  }
  yield* Console.log(process.platform === "win32"
    ? "Private HTTPS creation, trust, restoration and native publisher cleanup passed"
    : "Private HTTPS creation, trust and restoration passed")
}))
BunRuntime.runMain(main.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
