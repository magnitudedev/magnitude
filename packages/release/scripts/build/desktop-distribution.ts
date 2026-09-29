import { Effect, Option, Schema } from "effect"
import {
  DESKTOP_DISTRIBUTION_VARIABLE,
  DesktopDistributionJson,
  type AppleTeamId,
  type DesktopDistribution,
  type WindowsPublisher,
} from "../../src/desktop-distribution"
import type { HostId } from "../../src/targets"
import { appleSigning } from "../apple/signing"
import { windowsSigning } from "./windows-signing"

/**
 * Desktop's publisher identities for one host. Each identity is resolved only on the platform that
 * compiles it into Desktop, so another platform's signing configuration is never consulted.
 */
export const resolveDesktopDistribution = (host: HostId) => Effect.gen(function* () {
  const appleTeam = host.startsWith("darwin-")
    ? yield* appleSigning.pipe(Effect.map((signing) =>
      signing.mode === "developer-id" ? Option.some(signing.team) : Option.none<AppleTeamId>()))
    : Option.none<AppleTeamId>()
  const windowsPublisher = host === "windows-x64-msvc"
    ? yield* windowsSigning.pipe(Effect.map((signing) =>
      signing.mode === "artifact-signing" ? Option.some(signing.publisher) : Option.none<WindowsPublisher>()))
    : Option.none<WindowsPublisher>()
  const distribution: DesktopDistribution = { appleTeam, windowsPublisher }
  return distribution
})

/** The environment of the Desktop build (`bun run build` in `desktop/`) for one host. */
export const desktopBuildEnvironment = (host: HostId) => resolveDesktopDistribution(host).pipe(
  Effect.flatMap(Schema.encode(DesktopDistributionJson)),
  Effect.map((encoded): Readonly<Record<string, string | undefined>> => ({
    ...process.env,
    [DESKTOP_DISTRIBUTION_VARIABLE]: encoded,
  })),
)
