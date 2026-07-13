import { type Atom, atomizeModelId } from "./atomizer"
import { type Family, match } from "./matcher"

export interface ClassifyResult {
  readonly familyId: string
  readonly priority: number
  readonly matched: boolean
}

export function classify(
  id: string,
  families: readonly Family[],
): ClassifyResult {
  const atoms = atomizeModelId(id)
  const best = match(atoms, families)
  if (!best) {
    return { familyId: "", priority: 0, matched: false }
  }
  return { familyId: best.familyId, priority: best.priority, matched: true }
}

export { type Atom, atomizeModelId }
export { type Family, type PatternEntry } from "./matcher"
