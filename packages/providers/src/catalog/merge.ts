import type { ModelDefinition } from '../types'

function assignDefined<K extends keyof ModelDefinition>(
  target: ModelDefinition,
  source: Partial<ModelDefinition>,
  key: K,
): void {
  const value = source[key]
  if (value !== undefined) {
    target[key] = value
  }
}

function overlayModel(
  base: ModelDefinition | undefined,
  patch: ModelDefinition,
): ModelDefinition {
  if (!base) return { ...patch }

  const next: ModelDefinition = { ...base }
  for (const key of Object.keys(patch) as Array<keyof ModelDefinition>) {
    assignDefined(next, patch, key)
  }
  return next
}

export function mergeProviderModels(
  staticFallback: readonly ModelDefinition[],
  ...sources: readonly ModelDefinition[][]
): ModelDefinition[] {
  const merged = new Map<string, ModelDefinition>()
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
    const byDate = b.releaseDate.localeCompare(a.releaseDate)
    if (byDate !== 0) return byDate
    return (sourceOrder.get(a.id) ?? 0) - (sourceOrder.get(b.id) ?? 0)
  })
}