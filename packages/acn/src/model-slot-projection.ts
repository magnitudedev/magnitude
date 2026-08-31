import {
  type ModelCapabilities,
  type ModelSlotAvailability,
  type ProviderModelCatalogEntry,
  type SlotId,
} from "@magnitudedev/acn-protocol"

export const localModelSlotAvailability = (
  input: {
    readonly catalogIdentityPending: boolean
    readonly offeringsReady: boolean
    readonly offeringExists: boolean
  },
): ModelSlotAvailability => {
  if (input.catalogIdentityPending) {
    return { _tag: "Pending" }
  }
  if (!input.offeringExists) {
    if (!input.offeringsReady) return { _tag: "Pending" }
    return {
      _tag: "Unavailable",
      failure: {
        code: "local_offering_unavailable",
        message: "The selected local configuration is unavailable",
        retryable: true,
      },
    }
  }
  return { _tag: "Available" }
}

export interface LocalOfferingSelectionEvidence {
  readonly capabilities: ModelCapabilities
}

export const selectableModelCapabilities = (
  slotId: SlotId,
  catalogModel: ProviderModelCatalogEntry | undefined,
  localOffering: LocalOfferingSelectionEvidence | undefined,
): ModelCapabilities | undefined => {
  if (localOffering) {
    if (catalogModel && !catalogModel.supportedSlots.includes(slotId)) {
      return undefined
    }
    return localOffering.capabilities
  }
  return catalogModel?.availability._tag === "Available"
    && catalogModel.supportedSlots.includes(slotId)
    ? catalogModel.capabilities
    : undefined
}
