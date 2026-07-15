import { Schema } from "effect"

/** How the `llama-server` binary was located during tiered resolution. */
export const BinarySource = Schema.Literal(
  "env",
  "config",
  "cache",
  "path",
  "common-location",
  "download",
)
export type BinarySource = Schema.Schema.Type<typeof BinarySource>

/** A successfully resolved `llama-server` binary with its metadata. */
export const ResolvedBinary = Schema.Struct({
  /** Absolute path to the `llama-server` executable. */
  path: Schema.String,
  /** Directory containing the executable and shared libraries. */
  directory: Schema.String,
  /** Parsed build number from `--version` output. */
  buildNumber: Schema.Number,
  /** How the binary was found. */
  source: BinarySource,
})
export type ResolvedBinary = Schema.Schema.Type<typeof ResolvedBinary>

/** Current binary availability status (does not trigger download). */
export const BinaryStatus = Schema.Struct({
  /** Whether a valid binary is installed. */
  installed: Schema.Boolean,
  /** Parsed build number, or `null` if no binary found. */
  buildNumber: Schema.NullOr(Schema.Number),
  /** Absolute path to the binary, or `null` if not found. */
  path: Schema.NullOr(Schema.String),
  /** How the binary was found, or `null` if not found. */
  source: Schema.NullOr(BinarySource),
  /** Whether the installed build meets the minimum required version. */
  meetsMinimum: Schema.Boolean,
  /** The minimum accepted build number. */
  minimumRequired: Schema.Number,
  /** The recommended build number for new installs. */
  recommended: Schema.Number,
})
export type BinaryStatus = Schema.Schema.Type<typeof BinaryStatus>

/** Raw download result before validation. */
export const DownloadResult = Schema.Struct({
  /** Path to the extracted binary. */
  path: Schema.String,
  /** Directory containing the extracted binary. */
  directory: Schema.String,
  /** Build number from the downloaded release. */
  buildNumber: Schema.Number,
})
export type DownloadResult = Schema.Schema.Type<typeof DownloadResult>
