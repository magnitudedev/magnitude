const DECIMAL_MEGABYTE = 1_000_000
const SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"] as const

type SizeUnit = typeof SIZE_UNITS[number]
type SizeRounding = "nearest" | "up"

interface FormatScaledSizeOptions {
  readonly base: 1000 | 1024
  readonly minimumUnit: SizeUnit
  readonly fractionDigits: (amount: number, unit: SizeUnit) => number
  readonly rounding?: SizeRounding
}

export interface FormatMemorySizeOptions {
  readonly rounding?: SizeRounding
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
  let rounded = roundAmount(
    amount,
    options.fractionDigits(amount, unit),
    options.rounding ?? "nearest",
  )
  if (rounded >= options.base && unitIndex < SIZE_UNITS.length - 1) {
    amount = rounded / options.base
    unitIndex += 1
    unit = SIZE_UNITS[unitIndex]!
    rounded = roundAmount(
      amount,
      options.fractionDigits(amount, unit),
      options.rounding ?? "nearest",
    )
  }
  return `${rounded} ${unit}`
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

export const formatStorageSize = (bytes: number): string =>
  formatScaledSize(bytes, {
    base: 1000,
    minimumUnit: "B",
    fractionDigits: (amount, unit) => unit === "B" || amount >= 100
      ? 0
      : amount >= 10
        ? 1
        : 2,
  })

export const formatTransferRate = (bytesPerSecond: number): string => {
  const megabytes = bytesPerSecond / DECIMAL_MEGABYTE
  return `${megabytes >= 10 ? Math.round(megabytes) : megabytes.toFixed(1)} MB/s`
}
