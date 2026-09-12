import { Effect, Schema } from "effect"
import { isValidVersion } from "../client-update/release-channels"
import { RequestNonce } from "./request-auth"

const Version = Schema.String.pipe(Schema.maxLength(96), Schema.filter(isValidVersion))
const PlatformVersion = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(96), Schema.pattern(/^[a-zA-Z0-9 ._+()-]+$/))
export const UpdateRequest = Schema.Struct({
  protocol: Schema.Literal("1"),
  product: Schema.Literal("desktop"),
  version: Version,
  os: Schema.Literal("darwin", "windows", "linux"),
  os_version: PlatformVersion,
  arch: Schema.Literal("arm64", "x64"),
  package: Schema.Literal("mac-zip", "windows-exe", "deb", "rpm"),
  channel: Schema.Literal("stable", "beta", "alpha"),
  ts: Schema.NumberFromString.pipe(Schema.int(), Schema.nonNegative()),
  nonce: RequestNonce,
  distro: Schema.optionalWith(PlatformVersion, { as: "Option", exact: true }),
  distro_version: Schema.optionalWith(PlatformVersion, { as: "Option", exact: true }),
})
export type UpdateRequest = typeof UpdateRequest.Type
export class InvalidUpdateRequest extends Schema.TaggedError<InvalidUpdateRequest>()("InvalidUpdateRequest", {}) {}
export class ExpiredUpdateRequest extends Schema.TaggedError<ExpiredUpdateRequest>()("ExpiredUpdateRequest", {}) {}

export const decodeUpdateRequest = (url: URL, nowSeconds: number) => Effect.gen(function* () {
  if (url.href.length > 4096) return yield* new InvalidUpdateRequest()
  const fields: Record<string, string> = Object.create(null)
  for (const [key, value] of url.searchParams) {
    if (Object.hasOwn(fields, key)) return yield* new InvalidUpdateRequest()
    fields[key] = value
  }
  const request = yield* Schema.decodeUnknown(UpdateRequest)(fields, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new InvalidUpdateRequest()))
  if (Math.abs(nowSeconds - request.ts) > 300) return yield* new ExpiredUpdateRequest()
  const compatible = request.os === "darwin" ? request.package === "mac-zip"
    : request.os === "windows" ? request.package === "windows-exe" && request.arch === "x64"
    : request.package === "deb" || request.package === "rpm"
  if (!compatible) return yield* new InvalidUpdateRequest()
  return request
})
