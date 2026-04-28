import type { Codec } from "../codec/codec"
import type { Driver } from "../driver/driver"
import type { ProviderModel } from "../model/provider-model"
import type { ProviderDefinition } from "./provider-definition"

export interface BoundModel {
  readonly provider: ProviderDefinition
  readonly model: ProviderModel
  readonly codec: Codec
  readonly driver: Driver
  readonly authToken: string
  readonly endpoint: string
}
