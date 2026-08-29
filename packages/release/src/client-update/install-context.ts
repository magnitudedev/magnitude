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

/**
 * The launcher↔CLI relaunch handshake. The launcher declares the protocol
 * version it speaks; after a successful update the CLI exits with the relaunch
 * code only on an exact version match, and the launcher then re-runs its
 * pipeline once. Mismatch or absence degrades to the manual-restart message by
 * definition.
 */
export const LAUNCH_PROTOCOL_VERSION_VARIABLE = "MAGNITUDE_LAUNCH_PROTOCOL_VERSION"
export const LAUNCH_PROTOCOL_VERSION = 1
/** BSD EX_TEMPFAIL: the installation changed under this launcher; run again. */
export const RELAUNCH_EXIT_CODE = 75
/** The installation changed; run the new CLI once as `magnitude service start`. */
export const POST_UPDATE_SERVICE_START_EXIT_CODE = 76

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
