import { Schema } from "effect"

export const WebSearchProviderSchema = Schema.Literal("magnitude", "exa")
export type WebSearchProvider = typeof WebSearchProviderSchema.Type

const PayloadSampleSchema = Schema.Struct({
  text: Schema.String,
  encodedBytes: Schema.Number,
  truncated: Schema.Boolean,
})

export class WebSearchNotConfigured extends Schema.TaggedError<WebSearchNotConfigured>()(
  "WebSearchNotConfigured",
  {},
) {}

export class WebSearchRequestFailed extends Schema.TaggedError<WebSearchRequestFailed>()(
  "WebSearchRequestFailed",
  {
    provider: WebSearchProviderSchema,
    reason: Schema.String,
  },
) {}

export class WebSearchRequestEncodingFailed extends Schema.TaggedError<WebSearchRequestEncodingFailed>()(
  "WebSearchRequestEncodingFailed",
  {
    provider: WebSearchProviderSchema,
    reason: Schema.String,
  },
) {}

export class WebSearchTimedOut extends Schema.TaggedError<WebSearchTimedOut>()(
  "WebSearchTimedOut",
  {
    provider: WebSearchProviderSchema,
    timeoutMs: Schema.Number,
  },
) {}

export class WebSearchRejected extends Schema.TaggedError<WebSearchRejected>()(
  "WebSearchRejected",
  {
    provider: WebSearchProviderSchema,
    status: Schema.Number,
    message: Schema.String,
    body: PayloadSampleSchema,
  },
) {}

export class WebSearchResponseReadFailed extends Schema.TaggedError<WebSearchResponseReadFailed>()(
  "WebSearchResponseReadFailed",
  {
    provider: WebSearchProviderSchema,
    reason: Schema.String,
  },
) {}

export class WebSearchInvalidResponse extends Schema.TaggedError<WebSearchInvalidResponse>()(
  "WebSearchInvalidResponse",
  {
    provider: WebSearchProviderSchema,
    body: PayloadSampleSchema,
    issue: Schema.String,
  },
) {}

export type WebSearchError =
  | WebSearchNotConfigured
  | WebSearchRequestEncodingFailed
  | WebSearchRequestFailed
  | WebSearchTimedOut
  | WebSearchRejected
  | WebSearchResponseReadFailed
  | WebSearchInvalidResponse

const providerName = (provider: WebSearchProvider): string =>
  provider === "magnitude" ? "Magnitude Cloud" : "Exa"

export function formatWebSearchError(error: WebSearchError): string {
  switch (error._tag) {
    case "WebSearchNotConfigured":
      return "Web search is not configured"
    case "WebSearchRequestEncodingFailed":
      return `${providerName(error.provider)} web search request could not be encoded: ${error.reason}`
    case "WebSearchRequestFailed":
      return `${providerName(error.provider)} web search request failed: ${error.reason}`
    case "WebSearchTimedOut":
      return `${providerName(error.provider)} web search timed out after ${error.timeoutMs}ms`
    case "WebSearchRejected":
      return `${providerName(error.provider)} web search was rejected with HTTP ${error.status}: ${error.message}`
    case "WebSearchResponseReadFailed":
      return `${providerName(error.provider)} web search response could not be read: ${error.reason}`
    case "WebSearchInvalidResponse":
      return `${providerName(error.provider)} web search returned an invalid response: ${error.issue}`
  }
}
