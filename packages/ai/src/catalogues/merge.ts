import type { ProviderModel } from "../lib/model/provider-model"

function overlayModel(base: ProviderModel | undefined, patch: ProviderModel): ProviderModel {
  return base ? { ...base, ...patch } : { ...patch }
}

export function mergeProviderModels(
  staticModels: readonly ProviderModel[],
  ...dynamicSources: readonly ProviderModel[][]
): ProviderModel[] {
  const merged = new Map<string, ProviderModel>()
  const sourceOrder = new Map<string, number>()

  let order = 0

  for (const model of staticModels) {
    merged.set(model.id, { ...model })
    sourceOrder.set(model.id, order++)
  }

  for (const source of dynamicSources) {
    for (const model of source) {
      const existing = merged.get(model.id)
      merged.set(model.id, overlayModel(existing, model))
      if (!sourceOrder.has(model.id)) {
        sourceOrder.set(model.id, order++)
      }
    }
  }

  return Array.from(merged.values()).sort((left, right) => {
    const byReleaseDate = (right.releaseDate ?? "").localeCompare(left.releaseDate ?? "")
    if (byReleaseDate !== 0) {
      return byReleaseDate
    }
    return (sourceOrder.get(left.id) ?? 0) - (sourceOrder.get(right.id) ?? 0)
  })
}
