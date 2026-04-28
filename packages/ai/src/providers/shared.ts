import type { ModelId } from "../lib/model/canonical-model"
import type { ModelCosts, ProviderModel } from "../lib/model/provider-model"

const STATIC_MODEL_COST: ModelCosts = {
  inputPerM: 0,
  outputPerM: 0,
  cacheReadPerM: null,
  cacheWritePerM: null,
}

interface ModelInit extends Partial<ProviderModel> {
  readonly id: string
  readonly name: string
  readonly releaseDate: string
}

function toCanonicalModelId(id: string): ModelId | null {
  const stripped = id.includes("/") ? id.split("/").at(-1) ?? id : id
  return stripped as ModelId
}

export function defineModels(
  providerId: string,
  providerName: string,
  models: readonly ModelInit[],
): readonly ProviderModel[] {
  return models.map((model) => ({
    id: model.id,
    providerId,
    providerName,
    canonicalModelId: model.canonicalModelId ?? toCanonicalModelId(model.id),
    name: model.name,
    contextWindow: model.contextWindow ?? 200_000,
    maxContextTokens: model.maxContextTokens ?? null,
    maxOutputTokens: model.maxOutputTokens ?? null,
    supportsToolCalls: model.supportsToolCalls ?? false,
    supportsReasoning: model.supportsReasoning ?? false,
    supportsVision: model.supportsVision ?? false,
    supportsGrammar: model.supportsGrammar,
    paradigm: model.paradigm,
    costs: model.costs ?? STATIC_MODEL_COST,
    releaseDate: model.releaseDate,
    discovery: model.discovery,
  }))
}
