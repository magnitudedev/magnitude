import { Option } from "effect"
import {
  StreamStartProviderRejection,
  type ProviderCall,
  type RejectedHttpResponse,
} from "@magnitudedev/ai"

export const classifyMiniMaxRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection => {
  const message = response.body.slice(0, 500) || `HTTP ${response.status}`
  const rejection = response.status === 401 || response.status === 403
    ? { _tag: "AuthRejected" as const, message }
    : response.status === 404
      ? { _tag: "ModelUnavailable" as const, message }
      : response.status === 429
        ? {
            _tag: "RateLimited" as const,
            message,
            retryPolicy: {
              retry: true,
              retryAfterMs: Option.fromNullable(response.retryAfterMs),
            },
          }
        : response.status >= 500
          ? {
              _tag: "UpstreamFailure" as const,
              message,
              retryPolicy: {
                retry: true,
                retryAfterMs: Option.fromNullable(response.retryAfterMs),
              },
            }
          : { _tag: "InvalidRequest" as const, message }

  return new StreamStartProviderRejection({ call, response, rejection })
}
