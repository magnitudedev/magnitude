import { Schema } from "effect"
import { HuggingFaceArtifact, HuggingFaceArtifactFile } from "./contracts"
import { HuggingFaceFilePath } from "./identity"

export const HuggingFaceCachedFile = Schema.extend(
  HuggingFaceArtifactFile,
  Schema.Struct({ snapshotRelativePath: HuggingFaceFilePath }),
)
export type HuggingFaceCachedFile = Schema.Schema.Type<typeof HuggingFaceCachedFile>

export const HuggingFaceInstallationManifest = Schema.Struct({
  version: Schema.Literal(1),
  artifact: HuggingFaceArtifact,
  files: Schema.Array(HuggingFaceCachedFile),
  installedAt: Schema.DateFromString,
})
export type HuggingFaceInstallationManifest = Schema.Schema.Type<typeof HuggingFaceInstallationManifest>
export const HuggingFaceInstallationManifestJson = Schema.parseJson(HuggingFaceInstallationManifest, { space: 2 })
