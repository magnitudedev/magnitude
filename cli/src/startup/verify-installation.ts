import { BunContext } from "@effect/platform-bun"
import { verifyWindowsInstallationDownload } from "@magnitudedev/daemon-management/application-update"
import { Effect } from "effect"
import publicKey from "../../../packages/release/resources/distribution/magnitude-2026-01.pub.pem" with { type: "text" }

// Acceptance builds substitute their isolated public key at compilation, never at runtime.
declare const MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY: string
const trustedKey = typeof MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY === "undefined" ? publicKey : MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY
export const verifyInstallationDownload = (offer: string, artifact: string, channel: string) => Effect.runPromise(
  verifyWindowsInstallationDownload({ offer, artifact, channel, publicKey: trustedKey }).pipe(Effect.provide(BunContext.layer),
    Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))
