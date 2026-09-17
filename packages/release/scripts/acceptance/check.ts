import { FileSystem, FetchHttpClient } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Clock, Config, Effect, Option, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { release } from "node:os"
import { checkHostedUpdate, UpdateClientMetadata } from "../../src/hosted-update/client"
import { installationId, signUpdateRequest } from "../../src/hosted-update/request-auth"
import { decodePublisherPublicKey } from "../../src/hosted-update/manifest"
import { resolve } from "node:path"

class AcceptanceCheckFailed extends Schema.TaggedError<AcceptanceCheckFailed>()("AcceptanceCheckFailed", {}) {}
const Evidence = Schema.Struct({ at: Schema.Number, installation: Schema.String, metadata: UpdateClientMetadata, updateAvailable: Schema.Boolean })
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const os = process.platform === "win32" ? "windows" : process.platform
  const distro: Record<string, string> = {}
  if (os === "linux") {
    for (const line of (yield* fs.readFileString("/etc/os-release")).split("\n")) {
      const match = /^(ID|VERSION_ID)=(.*)$/.exec(line)
      if (match) distro[match[1]!] = match[2]!.replace(/^"|"$/g, "")
    }
  }
  const metadata = yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: "0.0.14", os, os_version: release(), arch: process.arch,
    package: os === "darwin" ? "mac-zip" : os === "windows" ? "windows-exe" : (yield* fs.exists("/usr/bin/dpkg")) ? "deb" : "rpm",
    ...(os === "linux" ? { distro: distro.ID, distro_version: distro.VERSION_ID } : {}),
  })
  const identity = yield* Effect.try({ try: () => generateKeyPairSync("ed25519"), catch: () => new AcceptanceCheckFailed() })
  const publisher = yield* decodePublisherPublicKey(yield* fs.readFileString(resolve(import.meta.dir, "../../resources/distribution/acceptance.pub.pem")))
  const update = yield* checkHostedUpdate({ origin: "https://magnitude-update-acceptance.vercel.app", metadata,
    sign: url => signUpdateRequest(identity.privateKey, url), trustedPublishers: new Map([["acceptance", publisher]]),
    userAgent: `Magnitude-acceptance/0.0.14 ${process.arch} Bun/${Bun.version} ${os}/${release()}`,
  })
  const evidence = { at: yield* Clock.currentTimeMillis, installation: yield* installationId(identity.publicKey), metadata, updateAvailable: Option.isSome(update) }
  yield* fs.writeFileString(yield* Config.string("MAGNITUDE_ACCEPTANCE_OUTPUT"), yield* Schema.encode(Schema.parseJson(Evidence))(evidence))
  yield* Effect.logInfo("Native signed check accepted", { os, arch: process.arch })
})
BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
