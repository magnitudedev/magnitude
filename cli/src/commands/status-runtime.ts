import { desktopApplication, desktopServiceOrigin, readDesktopLoginStartup } from "../server/application"
import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import type { ApplicationOwner } from "@magnitudedev/sdk/desktop-host"
import { Effect, Option } from "effect"
import { existingAcnConnection } from "../server/acn-connection"
import { runCommand } from "./output"

interface ActiveModel {
  readonly displayName: string
  readonly status: "Loading" | "Ready" | "Stopping"
}

interface ServiceStatusPresentation {
  readonly status: "Stopped" | "Starting" | "Ready" | "Stopping" | "Failed" | "CleanupFailed"
  readonly address: string
  readonly version: Option.Option<string>
  readonly startsAutomaticallyOnLogin: Option.Option<boolean>
  readonly activeModel: { readonly _tag: "Unavailable" } | { readonly _tag: "Observed"; readonly model: Option.Option<ActiveModel> }
  readonly owner: Option.Option<ApplicationOwner>
}

const serviceAddress = new URL(desktopServiceOrigin).host

const readActiveModel = Effect.scoped(Effect.gen(function* () {
  const connection = yield* existingAcnConnection
  const catalog = yield* connection.client.models.getCatalog({})
  if (catalog._tag === "Initializing") return { _tag: "Unavailable" } as const
  for (const entry of catalog.models) {
    if (entry._tag !== "Local") continue
    const model = entry.product
    const residency = model._tag === "Discovered"
      ? model.state._tag === "Ready" ? model.state.residencyState : undefined
      : "residencyState" in model.acquisitionState
        ? model.acquisitionState.residencyState
        : undefined
    if (residency === undefined) continue
    if (residency._tag !== "Requested"
      && residency._tag !== "Loading"
      && residency._tag !== "Ready"
      && residency._tag !== "Stopping") continue
    return { _tag: "Observed", model: Option.some({
      displayName: formatLocalModelDisplayName(model),
      status: residency._tag === "Requested" ? "Loading" as const : residency._tag,
    }) } as const
  }
  return { _tag: "Observed", model: Option.none<ActiveModel>() } as const
}))

const publicServiceStatus = desktopApplication.observe.pipe(
  Effect.map(Option.some),
  Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())),
  Effect.flatMap((snapshot) => {
    const state = Option.isSome(snapshot) ? snapshot.value.service : undefined
    const activeModel = state?._tag === "Ready" ? readActiveModel.pipe(
      Effect.timeout("2 seconds"),
      Effect.orElseSucceed(() => ({ _tag: "Unavailable" } as const)),
    ) : Effect.succeed({ _tag: "Unavailable" } as const)
    return Effect.all({ model: activeModel, login: Option.isSome(snapshot) && snapshot.value.owner._tag === "Desktop" ? readDesktopLoginStartup.pipe(Effect.map(state => state._tag === "Enabled" ? Option.some(true) : state._tag === "Disabled" ? Option.some(false) : Option.none<boolean>()), Effect.orElseSucceed(() => Option.none<boolean>())) : Effect.succeed(Option.none<boolean>()) }, { concurrency: "unbounded" }).pipe(Effect.map(({ model, login }): ServiceStatusPresentation => ({
      status: state?._tag ?? "Stopped", address: serviceAddress,
      version: state?._tag === "Ready" ? Option.some(String(state.health.version)) : Option.none(),
      startsAutomaticallyOnLogin: login, activeModel: model,
      owner: Option.map(snapshot, value => value.owner),
    })))
  }),
)

export const renderStatus = (status: ServiceStatusPresentation): string => [
  "Magnitude service",
  `  Runtime         ${status.status}`,
  `  Owner           ${Option.match(status.owner, { onNone: () => "None", onSome: owner => owner._tag })}`,
  ...(Option.isSome(status.owner) && status.owner.value._tag === "Desktop" ? [
    `  Tray            ${status.owner.value.tray._tag === "Unavailable" ? `Unavailable · ${status.owner.value.tray.message}` : status.owner.value.tray._tag}`,
    `  Starts at login ${Option.match(status.startsAutomaticallyOnLogin, { onNone: () => "Unavailable", onSome: value => value ? "Yes" : "No" })}`,
  ] : []),
  ...(Option.isSome(status.version) ? [
    `  Version         ${status.version.value}`,
    `  Address         ${status.address}`,
  ] : []),
  ...(status.status === "Ready" ? [`  Active model    ${status.activeModel._tag === "Unavailable" ? "Unavailable" : Option.match(status.activeModel.model, {
    onNone: () => "None",
    onSome: (model) => model.status === "Ready"
      ? model.displayName
      : `${model.displayName} - ${model.status}`,
  })}`] : []),
  "",
].join("\n")

export const runStatus = () => runCommand({
  effect: publicServiceStatus,
  render: renderStatus,
})
