const DECIMAL_GIGABYTE = 1_000_000_000
const DECIMAL_MEGABYTE = 1_000_000

export interface FormatDecimalGigabytesOptions {
  readonly rounding?: "nearest" | "up"
}

export const formatDecimalGigabytes = (
  bytes: number,
  options: FormatDecimalGigabytesOptions = {},
): string => {
  const gigabytes = bytes / DECIMAL_GIGABYTE
  const rounded = options.rounding === "up"
    ? Math.ceil(gigabytes * 10) / 10
    : gigabytes
  return `${rounded.toFixed(1)} GB`
}

export const formatDecimalByteSize = (bytes: number): string => {
  const gigabytes = bytes / DECIMAL_GIGABYTE
  return gigabytes >= 1
    ? `${gigabytes.toFixed(gigabytes >= 10 ? 1 : 2)} GB`
    : `${(bytes / DECIMAL_MEGABYTE).toFixed(0)} MB`
}

export const formatDecimalMegabytesPerSecond = (bytesPerSecond: number): string => {
  const megabytes = bytesPerSecond / DECIMAL_MEGABYTE
  return `${megabytes >= 10 ? Math.round(megabytes) : megabytes.toFixed(1)} MB/s`
}
