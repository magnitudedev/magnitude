import { Option, Schema } from "effect"

export const AppleTeamId = Schema.String.pipe(Schema.pattern(/^[A-Z0-9]{10}$/), Schema.brand("AppleTeamId"))
export type AppleTeamId = typeof AppleTeamId.Type

export const WindowsPublisher = Schema.NonEmptyString.pipe(Schema.brand("WindowsPublisher"))
export type WindowsPublisher = typeof WindowsPublisher.Type

/**
 * The publisher identities compiled into Desktop. The release build resolves them once for its
 * host and hands them to the Desktop build; the Desktop build never reads signing configuration
 * from its environment. An absent identity means that host's build carries no publisher trust.
 */
export const DesktopDistribution = Schema.Struct({
  appleTeam: Schema.optionalWith(AppleTeamId, { as: "Option", exact: true }),
  windowsPublisher: Schema.optionalWith(WindowsPublisher, { as: "Option", exact: true }),
})
export type DesktopDistribution = typeof DesktopDistribution.Type

/** The Desktop build input. Unset is a development build: no publisher identities. */
export const DESKTOP_DISTRIBUTION_VARIABLE = "MAGNITUDE_DESKTOP_DISTRIBUTION"

export const developmentDesktopDistribution: DesktopDistribution = {
  appleTeam: Option.none(),
  windowsPublisher: Option.none(),
}

export const DesktopDistributionJson = Schema.parseJson(DesktopDistribution)
