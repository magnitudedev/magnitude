import type { ErrorClassifier } from "../errors/classify"
import type { AuthMethod } from "../auth/types"
import type { ProviderModel } from "../model/provider-model"

export interface ProviderDefinition {
  readonly id: string
  readonly name: string
  readonly family: "local" | "cloud"
  readonly defaultBaseUrl?: string
  readonly authMethods: readonly AuthMethod[]
  readonly codecId: string
  readonly classifyError: ErrorClassifier
  readonly models: readonly ProviderModel[]
}
