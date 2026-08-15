import { Option } from "effect"
import {
  type ModelFailure,
  type ModelSlot,
  type ModelSlotConfiguredLocal,
} from "@magnitudedev/sdk"
import { formatModelDisplayName } from "./model-presentation"

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
      displayName: formatModelDisplayName(
        slot.descriptor.displayName,
        slot.descriptor.variantLabel,
      ),
      contextWindow: (() => {
        switch (slot.residency._tag) {
          case "Ready":
            return Option.some(slot.residency.allocation.contextWindowTokens)
          case "Stopping":
            return slot.residency.allocation._tag === "Resident"
              ? Option.some(slot.residency.allocation.allocation.contextWindowTokens)
              : Option.map(
                  slot.residency.allocation.allocation,
                  (planned) => planned.contextWindowTokens,
                )
          case "Loading":
            return Option.map(
              slot.residency.plannedAllocation,
              (planned) => planned.contextWindowTokens,
            )
          case "Requested":
          case "Unloaded":
          case "Failed":
            return Option.none()
        }
      })(),
    }
    switch (slot.residency._tag) {
      case "Unloaded":
        return { _tag: "NotLoaded", ...details }
      case "Requested":
        return {
          _tag: "Loading",
          ...details,
          allocation: Option.none(),
          percentage: 0,
        }
      case "Loading":
        return {
          _tag: "Loading",
          ...details,
          allocation: Option.map(slot.residency.plannedAllocation, allocation),
          percentage: Math.round(
            Option.getOrElse(slot.residency.progress, () => 0) * 100,
          ),
        }
      case "Ready":
        return {
          _tag: "Running",
          ...details,
          allocation: allocation(slot.residency.allocation),
        }
      case "Stopping":
        return {
          _tag: "Stopping",
          ...details,
          allocation: slot.residency.allocation._tag === "Resident"
            ? Option.some(allocation(slot.residency.allocation.allocation))
            : Option.map(slot.residency.allocation.allocation, allocation),
        }
      case "Failed":
        return {
          _tag: "Failed",
          ...details,
          reason: slot.residency.failure,
        }
    }
  },
})
