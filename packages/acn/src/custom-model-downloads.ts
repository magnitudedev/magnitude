import { Option } from "effect"
import { bundlePackages, packageHoldsFile } from "@magnitudedev/icn"
import type {
  ModelAcquisitionFailure,
  ModelId,
  ModelTransferProgress,
} from "@magnitudedev/acn-protocol"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import type { ModelDownload, ModelPackage, ServableModelBundle } from "@magnitudedev/icn-protocol/schemas"

/** A download belongs to a model as its target bundle (an added model) or its drafter alone. */
export interface RelatedDownload {
  readonly kind: "Target" | "Drafter"
  readonly download: ModelDownload
}

/**
 * Downloads in ICN's order that ACN started for this model. One that carries an added model's
 * declared target acquires that target; any other fetches its drafter alone.
 */
export const customModelDownloads = (
  modelId: ModelId,
  config: CustomModelsConfig,
  downloads: readonly ModelDownload[],
): readonly RelatedDownload[] => {
  const declaration = config.declarations[modelId as keyof CustomModelsConfig["declarations"]]
  const holdsTarget = (download: ModelDownload) => {
    if (declaration?.origin._tag !== "Added") return false
    const { repository, commit } = declaration.origin
    const primary = declaration.target[0]
    return primary !== undefined
      && bundlePackages(download.bundle).some((pkg) => packageHoldsFile(pkg, repository, commit, primary))
  }
  return downloads.flatMap((download): readonly RelatedDownload[] => Option.contains(download.forModel, modelId)
    ? [{ kind: holdsTarget(download) ? "Target" : "Drafter", download }]
    : [])
}

/** The separate drafter package of a bundle; fetching it alone leaves the target untouched. */
export const drafterPackage = (bundle: ServableModelBundle): Option.Option<ModelPackage> => {
  if (bundle._tag === "Standalone") return Option.some(bundle.package)
  return bundle.draftSource._tag === "Separate" ? Option.some(bundle.draftSource.draft) : Option.none()
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
