/** Curated product/source policy. Artifact facts are supplied by ICN preview. */
export interface LocalModelCatalogEntry {
  readonly id: string
  /** Stable model identity shared by every quantization of the same checkpoint. */
  readonly modelId: string
  readonly family: string
  readonly displayName: string
  readonly repo: string
  readonly revision: string
  readonly primaryGguf: string
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
