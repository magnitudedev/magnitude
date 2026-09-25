import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { releaseChannelOf } from "../../src/client-update/release-channels"
import { decodePublisherPublicKey, type PublishedUpdate } from "../../src/hosted-update/manifest"
import { InstallationOffer, installationOfferFromPublication } from "../../src/hosted-update/installation-offer"
import { renderUnixInstallationScript, renderWindowsInstallationScript } from "./installation-scripts"

export class InstallationDistributionFailed extends Schema.TaggedError<InstallationDistributionFailed>()("InstallationDistributionFailed", {}) {}

/** Produce a fresh static hosting directory; deploying it is a separate release action. */
export const writeInstallationDistribution = (options: {
  readonly output: string
  readonly origin: string
  readonly appleTeam: string
  readonly windowsPublisher: string
  readonly publicKey: string
  readonly publications: readonly PublishedUpdate[]
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const key = yield* decodePublisherPublicKey(options.publicKey)
  const trust = new Map([["publisher", key]])
  const first = options.publications[0]
  if (!first) return yield* new InstallationDistributionFailed()
  const channel = releaseChannelOf(first.release.version)
  if (channel === "unknown") return yield* new InstallationDistributionFailed()
  const names = new Set<string>()
  const offers = yield* Effect.forEach(options.publications, publication => Effect.gen(function* () {
    const offer = yield* installationOfferFromPublication(publication, trust)
    if (offer.release.version !== first.release.version) return yield* new InstallationDistributionFailed()
    const target = publication.manifest.artifact.target
    const name = `${target.os}-${target.arch}-${target.package}.json`
    if (names.has(name)) return yield* new InstallationDistributionFailed()
    names.add(name)
    return { name, content: yield* Schema.encode(Schema.parseJson(InstallationOffer))(offer) }
  }))
  const shell = yield* renderUnixInstallationScript({ origin: options.origin, appleTeam: options.appleTeam, publicKey: options.publicKey })
  const powershell = yield* renderWindowsInstallationScript({ origin: options.origin, publisher: options.windowsPublisher })
  // Verify the entire batch before creating output; never overwrite an existing hosting tree.
  yield* fs.makeDirectory(options.output)
  yield* fs.makeDirectory(join(options.output, "install", channel), { recursive: true })
  yield* fs.writeFileString(join(options.output, "install.sh"), shell, { mode: 0o755 })
  yield* fs.writeFileString(join(options.output, "install.ps1"), powershell)
  for (const offer of offers) yield* fs.writeFileString(join(options.output, "install", channel, offer.name), offer.content)
  return { version: first.release.version, channel, offers: offers.length }
})
