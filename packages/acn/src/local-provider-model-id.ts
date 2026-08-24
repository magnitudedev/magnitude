import type { CatalogIdentity } from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/sdk"

export const localCatalogProviderModelId = (
  identity: CatalogIdentity,
): ProviderModelId => ProviderModelIdSchema.make(`${identity.modelId}:${identity.variantId}`)
