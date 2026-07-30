import { Effect } from "effect"
import type { ArtifactDownloadProgress } from "./artifact-download"

export interface ArtifactByteProgress {
  readonly completedBytes: number
  readonly totalBytes: number
}

export type ArtifactInstallationEvent =
  | {
      readonly _tag: "Downloading"
      readonly progress: ArtifactDownloadProgress
    }
  | {
      readonly _tag: "Verifying"
      readonly progress: ArtifactByteProgress
    }
  | {
      readonly _tag: "Extracting"
      readonly progress: ArtifactByteProgress
    }

export interface ArtifactInstallationObserver {
  readonly report: (
    event: ArtifactInstallationEvent
  ) => Effect.Effect<void>
}
