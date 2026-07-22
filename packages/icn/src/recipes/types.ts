export type RecipeQuantBitsClass = "q4" | "q5" | "q6" | "q8" | "mxfp4" | "other"
export type CatalogEvidenceScope =
  | "publisher_checkpoint"
  | "checkpoint_quantization"
  | "exact_artifact"
  | "cross_model_quant_tier"

export interface RecipeBenchmarkEvidence {
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
export interface RecipeLicenseReview {
  readonly expectedId: string
  readonly name: string
  readonly url: string
  readonly acknowledgementRequired: boolean
}

export interface RecipeQuantizationEvidence {
  readonly format: string
  readonly bitsClass: RecipeQuantBitsClass
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
export interface ModelRecipeArtifact {
  readonly id: string
  readonly repository: string
  readonly filenameIncludes: string
  readonly quantization: RecipeQuantizationEvidence
}

/** Only Magnitude-owned product and evaluation metadata is checked in. */
export interface ModelRecipe {
  readonly id: string
  readonly family: string
  readonly displayName: string
  readonly developer: string
  readonly description: string
  readonly modelRepository: string
  readonly productContextTokens: readonly (100_000 | 200_000)[]
  readonly performance: {
    readonly summary: string
    readonly benchmarks: readonly RecipeBenchmarkEvidence[]
  }
  readonly licenseReview: RecipeLicenseReview
  readonly qualityRank: number
  readonly artifacts: readonly ModelRecipeArtifact[]
}

export interface ModelRecipeRegistry {
  readonly reviewedAt: string
  readonly models: readonly ModelRecipe[]
}

/** Runtime join of one overlay artifact with a live immutable Hub snapshot. */
export interface ResolvedModelRecipe {
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
