import { Option } from "effect"
import type { LocalModelFitsAssessment } from "@magnitudedev/sdk"

/**
 * A quant's weight bit width, read from its quant name: `UD-Q4_K_XL`, `IQ4_XS` and `MXFP4` are 4-bit, `Q8_0` is
 * 8-bit, `BF16` is 16-bit. None when the name doesn't state one.
 */
export const quantBitWidth = (name: string): Option.Option<number> => {
  const upper = name.toUpperCase()
  const integer = upper.match(/(?:^|[^A-Z])(?:IQ|TQ|Q)(\d+)/)
  if (integer !== null) return Option.some(Number(integer[1]))
  const float = upper.match(/(?:^|[^A-Z])(?:MX|NV)?B?FP?(\d+)(?![\d.])/)
  return float === null ? Option.none() : Option.some(Number(float[1]))
}

/** A quant as people compare them: its bit width, publisher and quant name ("4-bit unsloth UD-Q4_K_XL"). */
export const formatQuantLabel = (name: string, repository: Option.Option<string>): string => [
  ...Option.toArray(Option.map(quantBitWidth(name), (bits) => `${bits}-bit`)),
  ...Option.toArray(Option.flatMap(repository, (value) => Option.fromNullable(value.split("/")[0]))),
  name,
].join(" ")

/** How a fitting quant runs on this machine, from its placement: held whole in memory, or streamed from RAM or disk. */
export const fitPlacementPhrase = (assessment: LocalModelFitsAssessment, unifiedMemory: boolean): string =>
  assessment.residency._tag === "Resident"
    ? unifiedMemory ? "Fits in memory" : "Fits in GPU"
    : assessment.placement.diskBytes > 0 && assessment.residency.bank.coldExperts === "disk"
      ? "Streams from SSD"
      : "Streams from RAM"
