import type { ProviderModel } from '../model/model'

type ProviderModelKey = keyof ProviderModel

function assignDefined(
  target: ProviderModel,
  source: Partial<ProviderModel>,
  key: ProviderModelKey,
): void {
  const value = source[key]
  if (value !== undefined) {
    (target as any)[key] = value
  }
}

function overlayModel(
  base: ProviderModel | undefined,
  patch: ProviderModel,
): ProviderModel {
  if (!base) return { ...patch }

  const next: ProviderModel = { ...base }
  for (const key of Object.keys(patch) as ProviderModelKey[]) {
    assignDefined(next, patch, key)
  }
  return next
}

export function mergeProviderModels(
  staticFallback: readonly ProviderModel[],
  ...sources: readonly ProviderModel[][]
): ProviderModel[] {
  const merged = new Map<string, ProviderModel>()
  const sourceOrder = new Map<string, number>()

  let order = 0
  for (const model of staticFallback) {
    merged.set(model.id, { ...model })
    sourceOrder.set(model.id, order++)
  }

  for (const source of sources) {
    for (const model of source) {
      const existing = merged.get(model.id)
      merged.set(model.id, overlayModel(existing, model))
      if (!sourceOrder.has(model.id)) {
        sourceOrder.set(model.id, order++)
      }
    }
  }

  return Array.from(merged.values()).sort((a, b) => {
    const byDate = (b.releaseDate ?? '').localeCompare(a.releaseDate ?? '')
    if (byDate !== 0) return byDate
    return (sourceOrder.get(a.id) ?? 0) - (sourceOrder.get(b.id) ?? 0)
  })
}
