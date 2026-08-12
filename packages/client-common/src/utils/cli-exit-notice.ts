import { Option } from "effect"
import type {
  AcnClientCloseResult,
  ModelSlotConfiguredLocal,
  ModelSlotsState,
} from "@magnitudedev/sdk"
import { formatModelDisplayName } from "./model-presentation"

export const CLI_EXIT_OBSERVATION_FALLBACK =
  "Magnitude may still have background processes running.\n" +
  "Run `magnitude stop` to stop the current daemon and release any loaded model now."

const activeLocalSlots = (state: ModelSlotsState): ReadonlyArray<ModelSlotConfiguredLocal> =>
  [state.slots.primary, state.slots.secondary].filter(
    (slot): slot is ModelSlotConfiguredLocal =>
      slot._tag === "ConfiguredLocal" &&
      Option.isSome(slot.instance) &&
      (slot.instance.value.lifecycle._tag === "Ready" || slot.instance.value.lifecycle._tag === "Loading")
  )

export const deriveCliExitNotice = (observation: AcnClientCloseResult): Option.Option<string> =>
  Option.match(observation, {
    onNone: () => Option.some(CLI_EXIT_OBSERVATION_FALLBACK),
    onSome: ({ modelSlots, connectedClientCount }) => {
      const byInstance = new Map<string, ModelSlotConfiguredLocal>()
      for (const slot of activeLocalSlots(modelSlots)) {
        const instance = Option.getOrThrow(slot.instance)
        byInstance.set(instance.id, byInstance.get(instance.id) ?? slot)
      }
      if (byInstance.size === 0) return Option.none()
      if (byInstance.size > 1) {
        const descriptions = [...byInstance.values()].map((slot) => {
          const instance = Option.getOrThrow(slot.instance)
          const activity = instance.lifecycle._tag === "Loading" ? "loading" : "running"
          return `${formatModelDisplayName(
            slot.descriptor.displayName,
            slot.descriptor.variantLabel,
          )} (${activity})`
        })
        const names = descriptions.length === 2
          ? descriptions.join(" and ")
          : `${descriptions.slice(0, -1).join(", ")}, and ${descriptions.at(-1)}`
        const firstSentence = connectedClientCount === 0
          ? `${names} are still active and will stop automatically after 10 minutes when idle.`
          : `${names} are still active. ${connectedClientCount} other ${
              connectedClientCount === 1 ? "client is" : "clients are"
            } connected.`
        return Option.some(
          `${firstSentence}\nRun \`magnitude stop\` to stop the current daemon and release the models now.`
        )
      }

      const slot = byInstance.values().next().value!
      const instance = Option.getOrThrow(slot.instance)
      const activity = instance.lifecycle._tag === "Loading" ? "loading" : "running"
      const name = formatModelDisplayName(
        slot.descriptor.displayName,
        slot.descriptor.variantLabel,
      )
      const firstSentence =
        connectedClientCount === 0
          ? activity === "loading"
            ? `${name} is still loading. When idle, it will stop automatically after 10 minutes.`
            : `${name} is still running and will stop automatically after 10 minutes.`
          : `${name} is still ${activity}. ${connectedClientCount} other ${
              connectedClientCount === 1 ? "client is" : "clients are"
            } connected.`

      return Option.some(
        `${firstSentence}\nRun \`magnitude stop\` to stop the current daemon and release the model now.`
      )
    },
  })
