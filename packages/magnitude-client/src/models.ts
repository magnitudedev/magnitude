import {
  NativeChatCompletions,
  defaultClassifyStreamError,
} from "@magnitudedev/ai"
import { classifyMagnitudeConnectionError } from "./errors"

export function createRoleSpec(roleId: string, endpoint: string) {
  return NativeChatCompletions.model({
    id: `magnitude/role/${roleId}`,
    modelId: `role/${roleId}`,
    endpoint,
    contextWindow: 0,
    maxOutputTokens: 0,
    options: {},
    classifyConnectionError: (failure) =>
      classifyMagnitudeConnectionError(roleId, failure),
    classifyStreamError: (failure) =>
      defaultClassifyStreamError(roleId, failure),
  })
}
