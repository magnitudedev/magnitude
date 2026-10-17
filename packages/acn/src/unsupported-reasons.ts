import { Schema } from "effect"
import {
  KnownUnsupportedReason,
  UnrecognizedUnsupportedReason,
  type UnsupportedReason as NativeUnsupportedReason,
} from "@magnitudedev/icn-protocol/schemas"

const isKnownReason = Schema.is(KnownUnsupportedReason)

const productUnsupportedReason = (reason: NativeUnsupportedReason): unknown => {
  if (!isKnownReason(reason)) return Schema.encodeSync(UnrecognizedUnsupportedReason)(reason)
  const encoded = Schema.encodeSync(KnownUnsupportedReason)(reason)
  return encoded._tag === "KernelDomain"
    ? { ...encoded, statics: encoded.statics.map(({ name, value }) => [name, value]) }
    : encoded
}

/**
 * ICN's decoded reasons in the product's encoded form, ready for product decoding: optional fields
 * are encoded back to present-or-absent and kernel statics become `[name, value]` pairs. A reason a
 * newer engine raised keeps its tag and detail, which the product decodes as `Unknown`. An empty
 * list still states that the model cannot run.
 */
export const productUnsupportedReasons = (
  reasons: readonly NativeUnsupportedReason[],
  detail: string,
): readonly unknown[] => reasons.length === 0
  ? [{ _tag: "Unknown", kind: "Unknown", detail }]
  : reasons.map(productUnsupportedReason)
