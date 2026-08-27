const DECIMAL_MEGABYTE = 1_000_000
const SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"] as const

export type SizeUnit = typeof SIZE_UNITS[number]
type SizeRounding = "nearest" | "up"

export interface MinimumFractionDigits {
  readonly digits: 0 | 1 | 2
  readonly fromUnit: SizeUnit
}

interface FormatScaledSizeOptions {
  readonly base: 1000 | 1024
  readonly minimumUnit: SizeUnit
  readonly fractionDigits: (amount: number, unit: SizeUnit) => number
  readonly minimumFractionDigits?: MinimumFractionDigits
  readonly rounding?: SizeRounding
}

export interface FormatMemorySizeOptions {
  readonly rounding?: SizeRounding
}

export interface FormatStorageSizeOptions {
  readonly minimumFractionDigits?: MinimumFractionDigits
}

const roundAmount = (
  amount: number,
  fractionDigits: number,
  rounding: SizeRounding,
): number => {
  const precision = 10 ** fractionDigits
  return rounding === "up"
    ? Math.ceil(amount * precision) / precision
    : Number(amount.toFixed(fractionDigits))
}

const formatAmount = (
  amount: number,
  maximumFractionDigits: number,
  minimumFractionDigits: number,
): string => {
  const fixed = amount.toFixed(maximumFractionDigits)
  if (minimumFractionDigits === maximumFractionDigits) return fixed
  const [integer, fraction = ""] = fixed.split(".")
  const retainedFraction = fraction
    .replace(/0+$/, "")
    .padEnd(minimumFractionDigits, "0")
  return retainedFraction.length === 0 ? (integer ?? fixed) : `${integer}.${retainedFraction}`
}

const formatScaledSize = (
  bytes: number,
  options: FormatScaledSizeOptions,
): string => {
  const minimumUnitIndex = SIZE_UNITS.indexOf(options.minimumUnit)
  let unitIndex = 0
  let amount = bytes
  while (amount >= options.base && unitIndex < SIZE_UNITS.length - 1) {
    amount /= options.base
    unitIndex += 1
  }
  while (unitIndex < minimumUnitIndex) {
    amount /= options.base
    unitIndex += 1
  }

  let unit = SIZE_UNITS[unitIndex]!
  const fractionDigitsFor = (value: number, valueUnit: SizeUnit): number => {
    const adaptiveFractionDigits = options.fractionDigits(value, valueUnit)
    return options.minimumFractionDigits !== undefined
      && SIZE_UNITS.indexOf(valueUnit) >= SIZE_UNITS.indexOf(options.minimumFractionDigits.fromUnit)
      ? Math.max(adaptiveFractionDigits, options.minimumFractionDigits.digits)
      : adaptiveFractionDigits
  }
  let fractionDigits = fractionDigitsFor(amount, unit)
  let rounded = roundAmount(amount, fractionDigits, options.rounding ?? "nearest")
  if (rounded >= options.base && unitIndex < SIZE_UNITS.length - 1) {
    amount = rounded / options.base
    unitIndex += 1
    unit = SIZE_UNITS[unitIndex]!
    fractionDigits = fractionDigitsFor(amount, unit)
    rounded = roundAmount(amount, fractionDigits, options.rounding ?? "nearest")
  }
  const minimumFractionDigits = options.minimumFractionDigits !== undefined
    && unitIndex >= SIZE_UNITS.indexOf(options.minimumFractionDigits.fromUnit)
    ? options.minimumFractionDigits.digits
    : 0
  return `${formatAmount(rounded, fractionDigits, minimumFractionDigits)} ${unit}`
}

export const formatMemorySize = (
  bytes: number,
  options: FormatMemorySizeOptions = {},
): string =>
  formatScaledSize(bytes, {
    base: 1024,
    minimumUnit: "MB",
    fractionDigits: () => 1,
    rounding: options.rounding,
  })

export const formatStorageSize = (
  bytes: number,
  options: FormatStorageSizeOptions = {},
): string =>
  formatScaledSize(bytes, {
    base: 1000,
    minimumUnit: "B",
    fractionDigits: (amount, unit) => unit === "B" || amount >= 100
      ? 0
      : amount >= 10
        ? 1
        : 2,
    minimumFractionDigits: options.minimumFractionDigits,
  })

export const formatTransferRate = (bytesPerSecond: number): string => {
  const megabytes = bytesPerSecond / DECIMAL_MEGABYTE
  return `${megabytes >= 10 ? Math.round(megabytes) : megabytes.toFixed(1)} MB/s`
}
