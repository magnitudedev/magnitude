import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { decodePublisherPublicKey, type PublishedUpdate } from "../../src/hosted-update/manifest"
import { InstallationOffer, installationOfferFromPublication } from "../../src/hosted-update/installation-offer"

/**
 * Stands in for the landing server's `/api/installer` in a static HTTPS fixture. Each fixture serves
 * one target, and a static server ignores the query, so the verified offer is the endpoint's file.
 */
export const writeInstallerOfferFixture = (hosting: string, publication: PublishedUpdate, publicKey: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const offer = yield* installationOfferFromPublication(publication, new Map([["publisher", yield* decodePublisherPublicKey(publicKey)]]))
  yield* fs.makeDirectory(join(hosting, "api"), { recursive: true })
  yield* fs.writeFileString(join(hosting, "api/installer"), yield* Schema.encode(Schema.parseJson(InstallationOffer))(offer))
})
