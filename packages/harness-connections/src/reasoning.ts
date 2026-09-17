import type { HarnessModel } from "./contract"

export const enabledReasoningEfforts = (model: HarnessModel): ReadonlyArray<string> =>
  model.capabilities.reasoning.efforts.filter((effort) => effort !== "none")

export const supportsReasoningEffort = (model: HarnessModel, effort: string): boolean =>
  model.capabilities.reasoning.efforts.some((candidate) => candidate === effort)

export const hasReasoning = (model: HarnessModel): boolean => enabledReasoningEfforts(model).length > 0

export interface ReasoningControlSurface<Control extends string> {
  readonly controls: ReadonlyArray<Control>
  readonly off?: Control
  readonly soleEnabled: Control
  /** Harness control labels whose normalized model meaning has a different name. */
  readonly aliases?: Readonly<Partial<Record<Control, string>>>
}

export interface ReasoningControlProjection<Control extends string> {
  readonly map: Readonly<Record<Control, string | null>>
  /** Harness control label that resolves to the model's normalized default. */
  readonly defaultControl: Control | undefined
}

/**
 * Projects a model-owned reasoning domain onto one harness-owned control surface.
 * The returned default is a harness control label, never a normalized effort name.
 */
export const projectReasoningControls = <Control extends string>(
  model: HarnessModel,
  surface: ReasoningControlSurface<Control>,
): ReasoningControlProjection<Control> => {
  const reasoning = model.capabilities.reasoning
  const supported = new Set<string>(reasoning.efforts.map(String))
  const map = Object.fromEntries(surface.controls.map((control) => {
    if (surface.off !== undefined && control === surface.off) {
      return [control, supported.has("none") ? "none" : null]
    }
    if (supported.has(control)) return [control, control]
    const alias = surface.aliases?.[control]
    return [control, alias !== undefined && supported.has(alias) ? alias : null]
  })) as Record<Control, string | null>

  const enabled = enabledReasoningEfforts(model)
  if (enabled.length === 1 && !Object.values(map).includes(enabled[0]!)) {
    map[surface.soleEnabled] = enabled[0]!
  }

  if (!reasoning.supported) return { map, defaultControl: undefined }
  return {
    map,
    defaultControl: surface.controls.find(
      (control) => map[control] === reasoning.defaultEffort,
    ),
  }
}
