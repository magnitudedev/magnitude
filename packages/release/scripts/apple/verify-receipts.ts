import { BunContext } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import { Config, Effect, Option, Schema } from "effect"
import { resolve, dirname } from "node:path"
import { AppleConsumerReceipt, AppleDistributionReceipt } from "./distribution"
import { AppleDistributionFailed } from "./signing"
import { ReleaseArtifactSchema, ReleaseManifestSchema } from "../../src/contracts"
import { backendPacks, releaseHosts } from "../../src/targets"
import { sha256File } from "../../src/macos-app"

export const verifyAppleReceipts = (root: string, candidate?: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const expectedCommit = yield* Config.string("MAGNITUDE_SOURCE_COMMIT")
  const team = yield* Config.string("APPLE_TEAM_ID")
  const files: string[] = []
  const visit = (directory: string): Effect.Effect<void, import("@effect/platform/Error").PlatformError> => Effect.gen(function* () {
    for (const file of yield* fs.readDirectory(directory)) {
      const path = resolve(directory, file)
      if ((yield* fs.stat(path)).type === "Directory") yield* visit(path)
      else files.push(path)
    }
  })
  yield* visit(root)
  const accepted = new Map<string, string>()
  const fail = (message: string) => new AppleDistributionFailed({ message })
  for (const file of files.filter((path) => path.endsWith("/apple-distribution.receipt.json"))) {
    const receipt = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(AppleDistributionReceipt))))
    if (receipt.sourceCommit !== expectedCommit || receipt.team !== team) return yield* fail("Apple receipt does not match the selected commit and publisher")
    const units = receipt.notarizations.map((value) => value.unit)
    const requiredUnits = receipt.artifacts.some((value) => value.id.startsWith("acn-"))
      ? ["cli", "inference", "app", "desktop"]
      : receipt.artifacts.map((value) => value.id.replace(/^icn-backend-/, ""))
    if (units.length !== requiredUnits.length || requiredUnits.some((unit) => !units.includes(unit))) return yield* fail("Apple receipt is missing a native software submission")
    for (const artifact of receipt.artifacts) {
      if (accepted.has(artifact.id)) return yield* fail(`Duplicate Apple receipt for ${artifact.id}`)
      const descriptor = resolve(dirname(file), `${artifact.id}.artifact.json`)
      const metadata = yield* fs.readFileString(descriptor).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseArtifactSchema))))
      if (metadata.id !== artifact.id || metadata.sha256 !== artifact.sha256 || (yield* sha256File(resolve(dirname(file), metadata.filename))) !== artifact.sha256) return yield* fail(`Apple accepted bytes changed for ${artifact.id}`)
      if ((metadata.kind === "acn" || metadata.kind === "desktop") && !receipt.stapledApp) return yield* fail("Magnitude.app has no validated stapled ticket")
      accepted.set(artifact.id, artifact.sha256)
    }
  }
  const expected = [
    ...releaseHosts.filter((host) => host.id.startsWith("darwin-")).flatMap((host) => [`cli-${host.id}`, `acn-${host.id}`, `icn-base-${host.id}`, `desktop-${host.id}`, `desktop-update-${host.id}`]),
    ...backendPacks.filter((pack) => pack.host.startsWith("darwin-")).map((pack) => `icn-backend-${pack.id}`),
  ]
  if (expected.some((id) => !accepted.has(id)) || accepted.size !== expected.length) return yield* fail("Publication requires the complete Developer ID and notarization receipt graph")
  const consumed = new Set<string>()
  for (const file of files.filter((path) => path.endsWith("/apple-consumer.receipt.json"))) {
    const receipt = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(AppleConsumerReceipt))))
    if (receipt.sourceCommit !== expectedCommit) return yield* fail("Apple consumer validated another source commit")
    for (const artifact of receipt.artifacts) {
      if (consumed.has(artifact.id) || accepted.get(artifact.id) !== artifact.sha256) return yield* fail(`Apple consumer bytes differ for ${artifact.id}`)
      consumed.add(artifact.id)
    }
  }
  const hostArtifacts = expected.filter((id) => !id.startsWith("icn-backend-"))
  if (hostArtifacts.some((id) => !consumed.has(id)) || consumed.size !== hostArtifacts.length) return yield* fail("Publication requires independent Apple host consumer acceptance")
  if (candidate !== undefined) {
    const manifest = yield* fs.readFileString(resolve(candidate, "magnitude-release.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseManifestSchema))))
    if (manifest.sourceCommit !== expectedCommit) return yield* fail("Candidate and Apple receipts name different commits")
    for (const artifact of manifest.artifacts.filter((value) => Option.exists(value.host, (host) => host.startsWith("darwin-")))) {
      if (accepted.get(artifact.id) !== artifact.sha256 || (yield* sha256File(resolve(candidate, artifact.filename))) !== artifact.sha256) return yield* fail(`Candidate bytes lack Apple acceptance: ${artifact.id}`)
    }
    if (manifest.artifacts.filter((value) => Option.exists(value.host, (host) => host.startsWith("darwin-"))).length !== accepted.size) return yield* fail("Candidate and Apple receipt graphs differ")
  }
})

if (import.meta.main) await Effect.runPromise(verifyAppleReceipts(resolve(process.argv[2] ?? "release-artifacts"), process.argv[3] === undefined ? undefined : resolve(process.argv[3])).pipe(Effect.provide(BunContext.layer)))
