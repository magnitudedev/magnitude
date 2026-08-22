import { Schema } from "effect"

const hasNul = (value: string): boolean => value.includes("\0")

/**
 * A lexically clean absolute host path: no NUL, no "." or ".." segments, no
 * doubled separators, no trailing separator except a bare root. Purely
 * lexical — host existence and kind are observations owned by the ACN
 * filesystem service.
 */
const isCleanAbsolutePath = (value: string): boolean => {
  if (hasNul(value)) return false
  if (value.startsWith("/")) {
    if (value === "/") return true
    return value
      .split("/")
      .slice(1)
      .every((segment) => segment !== "" && segment !== "." && segment !== "..")
  }
  if (!/^[A-Za-z]:[\\/]/.test(value)) return false
  const rest = value.slice(3)
  if (rest === "") return true
  return rest.split(/[\\/]/).every((segment) => segment !== "" && segment !== "." && segment !== "..")
}

/** An absolute path to an arbitrary host file or directory. */
export const AbsolutePathSchema = Schema.NonEmptyString.pipe(
  Schema.filter(isCleanAbsolutePath, { message: () => "Expected a normalized absolute path" }),
  Schema.brand("AbsolutePath"),
)
export type AbsolutePath = typeof AbsolutePathSchema.Type

/** An absolute path denoting a directory root (Project cwd, session cwd). */
export const DirectoryPathSchema = Schema.NonEmptyString.pipe(
  Schema.filter(isCleanAbsolutePath, {
    message: () => "Expected a normalized absolute directory path",
  }),
  Schema.brand("DirectoryPath"),
)
export type DirectoryPath = typeof DirectoryPathSchema.Type

/**
 * A path relative to an opened directory root: "" is the root itself;
 * otherwise "/"-joined non-empty segments with no ".", "..", NUL, backslash,
 * or drive prefix.
 */
const isRelativePath = (value: string): boolean => {
  if (value === "") return true
  if (hasNul(value) || value.includes("\\") || value.startsWith("/")) return false
  if (/^[A-Za-z]:/.test(value)) return false
  return value.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..")
}

export const RelativePathSchema = Schema.String.pipe(
  Schema.filter(isRelativePath, { message: () => "Expected a normalized relative path" }),
  Schema.brand("RelativePath"),
)
export type RelativePath = typeof RelativePathSchema.Type

/** The directory containing `path`; the root ("") for a top-level entry. */
export const parentDirectory = (path: RelativePath): RelativePath => {
  const index = path.lastIndexOf("/")
  return RelativePathSchema.make(index === -1 ? "" : path.slice(0, index))
}
