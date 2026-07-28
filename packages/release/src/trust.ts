import { Schema } from "effect"
import { TrustedReleaseKeySchema, type TrustedReleaseKey } from "./contracts"

declare const MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON: string | undefined

export const embeddedTrustedReleaseKeys = (): readonly TrustedReleaseKey[] => {
  const encoded = typeof MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON === "undefined"
    ? process.env.MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON
    : MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON
  if (!encoded) throw new Error("Magnitude release trust keys are not embedded")
  return Schema.decodeUnknownSync(Schema.Array(TrustedReleaseKeySchema))(JSON.parse(encoded))
}
