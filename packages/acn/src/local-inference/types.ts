export type CatalogQuantBitsClass = "q4" | "q5" | "q6" | "q8" | "mxfp4" | "other"
export type CatalogEvidenceScope =
  | "publisher_checkpoint"
  | "checkpoint_quantization"
  | "exact_artifact"
  | "cross_model_quant_tier"

export interface CatalogBenchmarkEvidence {
  /** Scores with different methodology IDs must never be compared directly. */
  readonly benchmarkId: string
  readonly label: string
  readonly score: number
  readonly unit: "percent" | "elo"
  readonly higherIsBetter: boolean
  readonly methodologyId: string
  readonly mode: string
  readonly evidenceScope: "publisher_checkpoint" | "exact_artifact"
  readonly sourceUrl: string
  readonly notes: string
}

/** A Magnitude review layered over the license reported by the live Hub repository. */
export interface CatalogLicenseReview {
  readonly expectedId: string
  readonly name: string
  readonly url: string
  readonly acknowledgementRequired: boolean
}

export interface CatalogQuantizationEvidence {
  readonly format: string
  readonly bitsClass: CatalogQuantBitsClass
  readonly quantAwareCheckpoint: boolean
  readonly fidelityRank: number
  readonly fidelityLabel: string
  readonly evidenceScope: CatalogEvidenceScope
  readonly summary: string
  readonly sourceUrl: string
}

/**
 * A stable selector, not a file manifest. ICN resolves it against the current
 * repository snapshot and returns the immutable commit, files, sizes, and hashes.
 */
export interface CanonicalArtifactOverlay {
  readonly id: string
  readonly repository: string
  readonly filenameIncludes: string
  readonly quantization: CatalogQuantizationEvidence
}

/** Only Magnitude-owned product and evaluation metadata is checked in. */
export interface CanonicalModelOverlay {
  readonly id: string
  readonly family: string
  readonly displayName: string
  readonly developer: string
  readonly description: string
  readonly modelRepository: string
  readonly productContextTokens: readonly (100_000 | 200_000)[]
  readonly performance: {
    readonly summary: string
    readonly benchmarks: readonly CatalogBenchmarkEvidence[]
  }
  readonly licenseReview: CatalogLicenseReview
  /** Compatibility input until the Phase-3 Pareto policy replaces scalar ranking. */
  readonly legacyQualityRank: number
  readonly artifacts: readonly CanonicalArtifactOverlay[]
}

export interface CanonicalModelCatalogOverlay {
  readonly schemaVersion: 2
  readonly catalogVersion: string
  readonly reviewedAt: string
  readonly models: readonly CanonicalModelOverlay[]
}

/** Runtime join of one overlay artifact with a live immutable Hub snapshot. */
export interface LocalModelCatalogEntry {
  readonly id: string
  readonly modelId: string
  readonly family: string
  readonly displayName: string
  readonly repo: string
  readonly revision: string
  readonly primaryGguf: string
  /** Live sum of the selected GGUF weight/shard files, used only for cheap prefiltering. */
  readonly publishedWeightBytes: number
  readonly additionalComponents: readonly {
    readonly path: string
    readonly role: "shard" | "projector" | "auxiliary" | "draft" | "mtp"
  }[]
  readonly supportedContextTokens: readonly number[]
  readonly quantTag: string
  readonly quantization: {
    readonly quantAwareCheckpoint: boolean
    readonly fidelityRank: number
    readonly fidelityLabel: string
    readonly fidelityEvidence: string
    readonly fidelitySourceUrl: string
  }
  readonly license: {
    readonly id: string
    readonly url: string
    readonly acknowledgementRequired: boolean
  }
  readonly modelQualityRank: number
}
