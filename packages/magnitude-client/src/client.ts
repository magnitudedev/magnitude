import { Auth } from "@magnitudedev/ai"
import type { RoleId } from "./contract"
import { createModelCatalog, type ModelCatalog } from "./catalog"
import { createRoleSpec } from "./models"

export interface MagnitudeClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
}

const DEFAULT_ENDPOINT = "https://app.magnitude.dev/api/v1"

export function createMagnitudeClient(config?: MagnitudeClientConfig) {
  const apiKey = config?.apiKey ?? process.env.MAGNITUDE_API_KEY
  if (!apiKey) throw new Error("No API key provided. Pass apiKey in config or set MAGNITUDE_API_KEY environment variable.")
  const endpoint = config?.endpoint ?? DEFAULT_ENDPOINT
  const auth = Auth.bearer(apiKey)
  const catalog = createModelCatalog({ endpoint, auth })

  return {
    /** Model catalog — fetch models, look up by role */
    catalog,

    /** Get a bound model for a role — synchronous, no network call needed */
    role: (id: RoleId) => {
      const spec = createRoleSpec(id, endpoint)
      const bound = spec.bind({ auth })
      return {
        stream: bound.stream,
        spec,
      }
    },
  }
}
