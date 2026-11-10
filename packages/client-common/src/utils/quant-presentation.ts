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

/**
 * Bit widths for a ladder's rungs, given smallest first: each rung's own width, and half a bit more on a rung that
 * shares its width with a smaller one, so `UD-IQ3_XXS` and `UD-Q3_K_XL` on one ladder read 3-bit and 3.5-bit.
 */
export const ladderBitWidths = (names: readonly string[]): ReadonlyArray<Option.Option<number>> =>
  names.map((name, index) => Option.map(quantBitWidth(name), (bits) =>
    names.slice(0, index).some((smaller) => Option.contains(quantBitWidth(smaller), bits)) ? bits + 0.5 : bits))

/**
 * A quant as people compare them: its bit width, publisher and quant name ("4-bit unsloth UD-Q4_K_XL"). A ladder
 * passes the width `ladderBitWidths` gave the rung.
 */
export const formatQuantLabel = (name: string, repository: Option.Option<string>, bits = quantBitWidth(name)): string => [
  ...Option.toArray(Option.map(bits, (width) => `${width}-bit`)),
  ...Option.toArray(Option.flatMap(repository, (value) => Option.fromNullable(value.split("/")[0]))),
  name,
].join(" ")

/**
 * How a fitting quant runs on this machine, from its placement: held whole in memory (on a discrete GPU, in its memory
 * alone or split with host RAM), or its experts streamed from RAM or disk.
 */
export const fitPlacementPhrase = (assessment: LocalModelFitsAssessment, unifiedMemory: boolean): string =>
  assessment.residency._tag === "Resident"
    ? unifiedMemory ? "Fits in memory" : assessment.placement.hostBytes > 0 ? "GPU and RAM" : "Fits in GPU"
    : assessment.placement.diskBytes > 0 && assessment.residency.bank.coldExperts === "disk"
      ? "Streams from SSD"
      : "Streams from RAM"
