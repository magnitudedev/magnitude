import { createHash } from "node:crypto"
import type { LocalModelCatalogEntry } from "./types"

export const artifactIdForCatalogEntry = (entry: LocalModelCatalogEntry): string =>
  createHash("sha256").update(`${entry.id}@${entry.revision}`).digest("hex")

export const providerModelIdForArtifact = (artifactId: string): string =>
  `local-${artifactId.slice(0, 24)}`

const opaqueChoiceId = (kind: string, identity: string): string =>
  `${kind}-${createHash("sha256").update(identity).digest("hex").slice(0, 32)}`

export const storedArtifactChoiceId = (artifactId: string): string =>
  opaqueChoiceId("stored", artifactId)

export const runningModelChoiceId = (serverId: string, providerModelId: string): string =>
  opaqueChoiceId("running", `${serverId}\0${providerModelId}`)
