import { Option } from "effect"
import {
  type ModelFailure,
  type ModelLoadPreview,
  type ModelSlot,
  type ModelSlotConfiguredLocal,
} from "@magnitudedev/sdk"

export type CurrentModelPreview =
  | { readonly _tag: "Available"; readonly allocation: ModelLoadPreview }
  | { readonly _tag: "Unavailable" }

export interface CurrentModelAllocation {
  readonly parallelSequences: number
  readonly physicalContextTokens: number
}

interface CurrentModelDetails {
  readonly slot: ModelSlotConfiguredLocal
  readonly displayName: string
  readonly contextWindow: Option.Option<number>
}

export type CurrentLocalModel =
  | { readonly _tag: "NoSelection" }
  | (CurrentModelDetails & {
      readonly _tag: "NotLoaded"
      readonly preview: Option.Option<CurrentModelPreview>
    })
  | (CurrentModelDetails & {
      readonly _tag: "Loading"
      readonly allocation: Option.Option<CurrentModelAllocation>
      readonly percentage: number
    })
  | (CurrentModelDetails & {
      readonly _tag: "Running"
      readonly allocation: CurrentModelAllocation
    })
  | (CurrentModelDetails & {
      readonly _tag: "Stopping"
      readonly allocation: Option.Option<CurrentModelAllocation>
    })
  | (CurrentModelDetails & {
      readonly _tag: "Failed"
      readonly preview: Option.Option<CurrentModelPreview>
      readonly reason: ModelFailure
    })

const allocation = (
  value: { readonly parallelSequences: number; readonly physicalContextTokens: number },
): CurrentModelAllocation => ({
  parallelSequences: value.parallelSequences,
  physicalContextTokens: value.physicalContextTokens,
})

export const deriveCurrentLocalModel = (
  slot: Option.Option<ModelSlot>,
): CurrentLocalModel => Option.match(slot, {
  onNone: () => ({ _tag: "NoSelection" }),
  onSome: (slot) => {
    if (slot._tag !== "ConfiguredLocal") return { _tag: "NoSelection" }
    const details: CurrentModelDetails = {
      slot,
      displayName: slot.descriptor.displayName,
      contextWindow: Option.match(slot.instance, {
        onNone: () => slot.readiness._tag === "Loadable"
          ? Option.some(slot.readiness.allocation.contextWindowTokens)
          : Option.none(),
        onSome: (instance) => {
          switch (instance.lifecycle._tag) {
            case "Ready":
              return Option.some(instance.lifecycle.allocation.contextWindowTokens)
            case "Stopping":
              return instance.lifecycle.allocation._tag === "Resident"
                ? Option.some(instance.lifecycle.allocation.allocation.contextWindowTokens)
                : Option.map(
                    instance.lifecycle.allocation.allocation,
                    (planned) => planned.contextWindowTokens,
                  )
            case "Loading":
              return Option.map(
                instance.lifecycle.plannedAllocation,
                (planned) => planned.contextWindowTokens,
              )
            case "Stopped":
            case "Failed":
              return slot.readiness._tag === "Loadable"
                ? Option.some(slot.readiness.allocation.contextWindowTokens)
                : Option.none()
          }
        },
      }),
    }
    const preview = slot.readiness._tag === "Assessing"
      ? Option.none<CurrentModelPreview>()
      : Option.some<CurrentModelPreview>(
          slot.readiness._tag === "Loadable"
            ? { _tag: "Available", allocation: slot.readiness.allocation }
            : { _tag: "Unavailable" },
        )
    if (Option.isNone(slot.instance)
      || slot.instance.value.lifecycle._tag === "Stopped") {
      return { _tag: "NotLoaded", ...details, preview }
    }
    switch (slot.instance.value.lifecycle._tag) {
      case "Loading":
        return {
          _tag: "Loading",
          ...details,
          allocation: Option.map(slot.instance.value.lifecycle.plannedAllocation, allocation),
          percentage: Math.round(
            Option.getOrElse(slot.instance.value.lifecycle.progress, () => 0) * 100,
          ),
        }
      case "Ready":
        return {
          _tag: "Running",
          ...details,
          allocation: allocation(slot.instance.value.lifecycle.allocation),
        }
      case "Stopping":
        return {
          _tag: "Stopping",
          ...details,
          allocation: slot.instance.value.lifecycle.allocation._tag === "Resident"
            ? Option.some(allocation(slot.instance.value.lifecycle.allocation.allocation))
            : Option.map(slot.instance.value.lifecycle.allocation.allocation, allocation),
        }
      case "Failed":
        return {
          _tag: "Failed",
          ...details,
          preview,
          reason: slot.instance.value.lifecycle.failure,
        }
    }
  },
})
