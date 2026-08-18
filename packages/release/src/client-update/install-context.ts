import { Option, Schema } from "effect"

/**
 * The launcher→CLI install-provenance contract. The launcher sets these two
 * variables when spawning the native CLI; the native CLI reads them to learn
 * which package manager owns the installation and where it lives. These
 * definitions are the single source for both sides: the launcher-side
 * detection and encoding live in packages/launcher, the reader lives here
 * beside the contract it depends on.
 */
export const MANAGED_BY_VARIABLE = "MAGNITUDE_MANAGED_BY"
export const MANAGED_PACKAGE_ROOT_VARIABLE = "MAGNITUDE_MANAGED_PACKAGE_ROOT"

export const PackageManagerSchema = Schema.Literal("npm", "bun", "pnpm")
export type PackageManager = typeof PackageManagerSchema.Type

export type InstallMethod = PackageManager | "other"

const decodePackageManager = Schema.decodeUnknownOption(PackageManagerSchema)

export const installMethodFromEnvironment = (
  environment: Readonly<Record<string, string | undefined>>,
): InstallMethod => Option.getOrElse(
  decodePackageManager(environment[MANAGED_BY_VARIABLE]),
  (): InstallMethod => "other",
)
