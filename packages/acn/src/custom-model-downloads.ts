import { Option } from "effect"
import { bundlePackages, packageHoldsFile } from "@magnitudedev/icn"
import {
  parseModelId,
  sameModelId,
  type ModelAcquisitionFailure,
  type ModelId,
  type ModelTransferProgress,
} from "@magnitudedev/acn-protocol"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import type { ModelDownload, ModelPackage } from "@magnitudedev/icn-protocol/schemas"
import { addedModel } from "./custom-models"

/** A download belongs to a model as its target bundle (an added model) or its drafter alone. */
export interface RelatedDownload {
  readonly kind: "Target" | "Drafter"
  readonly download: ModelDownload
}

/** The package of a download that holds an added model's first shard, at whatever commit. */
export const addedTargetPackage = (modelId: ModelId, download: ModelDownload): Option.Option<ModelPackage> => {
  const identity = parseModelId(modelId)
  if (identity._tag !== "HuggingFace") return Option.none()
  return Option.fromNullable(bundlePackages(download.bundle).find((pkg) => pkg.source._tag === "HuggingFace"
    && packageHoldsFile(pkg, identity.repositoryId, pkg.source.revision, identity.artifactSelector)))
}

/**
 * Downloads in ICN's order for this model. One that carries an added model's first shard acquires
 * that target, at its pinned commit or at a newer commit an update fetched (so an update that failed
 * or was cancelled still reads as the target's); any other fetches its drafter alone.
 */
export const customModelDownloads = (
  modelId: ModelId,
  config: CustomModelsConfig,
  downloads: readonly ModelDownload[],
): readonly RelatedDownload[] => {
  const added = Option.isSome(addedModel(config, modelId))
  return downloads.flatMap((download): readonly RelatedDownload[] => sameModelId(download.forModel, modelId)
    ? [{ kind: added && Option.isSome(addedTargetPackage(modelId, download)) ? "Target" : "Drafter", download }]
    : [])
}

export const downloadIsActive = (download: ModelDownload): boolean =>
  download.state._tag === "Pending" || download.state._tag === "Downloading"

export const downloadHasUnacknowledgedFailure = (download: ModelDownload): boolean =>
  download.state._tag === "Failed" && !download.state.acknowledged

export const transferProgress = (download: ModelDownload): Option.Option<ModelTransferProgress> => {
  const { state } = download
  switch (state._tag) {
    case "Pending": return Option.some({
      stage: "queued",
      completedBytes: Math.min(state.completedBytes, state.totalBytes),
      totalBytes: state.totalBytes,
      bytesPerSecond: Option.none(),
    })
    case "Downloading": return Option.some({
      stage: state.stage,
      completedBytes: Math.min(state.completedBytes, state.totalBytes),
      totalBytes: state.totalBytes,
      bytesPerSecond: Option.flatMap(state.bytesPerSecond, Option.fromNullable),
    })
    default: return Option.none()
  }
}

export const acquisitionFailure = (download: ModelDownload): Option.Option<ModelAcquisitionFailure> =>
  download.state._tag === "Failed" && !download.state.acknowledged
    ? Option.some(download.state.failure as ModelAcquisitionFailure)
    : Option.none()

/**
 * The newest related download of a kind, when its state still matters: running, or failed and
 * unacknowledged. A newer attempt supersedes an older failure.
 */
export const currentDownload = (
  related: readonly RelatedDownload[],
  kind: RelatedDownload["kind"],
): Option.Option<ModelDownload> => Option.fromNullable(related.findLast((entry) => entry.kind === kind)?.download).pipe(
  Option.filter((download) => downloadIsActive(download) || downloadHasUnacknowledgedFailure(download)),
)
