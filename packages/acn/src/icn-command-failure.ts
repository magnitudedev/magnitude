import { Effect } from "effect"
import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import type { IcnClientService } from "@magnitudedev/icn"

export type IcnCommandError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["ensureModelInstance"]>
>

const externalFailureMessage = (cause: unknown): string => {
  if (cause instanceof Error && cause.message.length > 0) return cause.message
  if (typeof cause === "object" && cause !== null && "message" in cause
    && typeof cause.message === "string" && cause.message.length > 0) return cause.message
  return "The local inference service did not provide failure details"
}

export const icnCommandFailure = (
  operation: string,
  cause: IcnCommandError,
): LocalModelMutationFailed => {
  switch (cause._tag) {
    case "GeneratedClientRemoteError": return new LocalModelMutationFailed({
      code: cause.body.error.code,
      message: cause.body.error.message,
      retryable: cause.status === 409 || cause.status >= 500,
    })
    case "GeneratedClientTransportError": return new LocalModelMutationFailed({
      code: `model_${operation}_transport_failed`,
      message: externalFailureMessage(cause.cause),
      retryable: true,
    })
    case "GeneratedClientInputError": return new LocalModelMutationFailed({
      code: `model_${operation}_request_invalid`,
      message: `Invalid local inference request input at ${cause.location}`,
      retryable: false,
    })
    case "GeneratedClientInvalidResponseError": return new LocalModelMutationFailed({
      code: `model_${operation}_response_invalid`,
      message: cause.message,
      retryable: true,
    })
  }
}
