import { Schema } from "effect"
import { UnsupportedReason as NativeUnsupportedReason } from "@magnitudedev/icn-protocol/schemas"

/**
 * ICN's decoded reasons in the product's encoded form, ready for product decoding: optional fields
 * are encoded back to present-or-absent and kernel statics become `[name, value]` pairs. An empty
 * list still states that the model cannot run.
 */
export const productUnsupportedReasons = (
  reasons: readonly NativeUnsupportedReason[],
  detail: string,
): readonly unknown[] => reasons.length === 0
  ? [{ _tag: "Unknown", kind: "Unknown", detail }]
  : reasons.map((reason) => {
      const encoded = Schema.encodeSync(NativeUnsupportedReason)(reason)
      return encoded._tag === "KernelDomain"
        ? { ...encoded, statics: encoded.statics.map(({ name, value }) => [name, value]) }
        : encoded
    })
