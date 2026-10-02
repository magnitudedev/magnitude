import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { atMostOnce } from "../transport/recovery"
import {
  ApplicationOwnerState,
  ApplicationOwnerUnavailable,
  OwnerRequestFailed,
  OwnerRequestUnsupported,
} from "../schemas/application-owner"

const OwnerError = Schema.Union(ApplicationOwnerUnavailable, OwnerRequestUnsupported, OwnerRequestFailed)
const command = <Tag extends string, Payload extends Schema.Struct.Fields>(tag: Tag, payload: Payload) =>
  Rpc.make(tag, { payload: Schema.Struct(payload), success: Schema.Struct({}), error: OwnerError }).pipe(atMostOnce)

/** The owning application's kind, supported requests, update state, and launch-at-login state. */
const WatchApplicationOwner = Rpc.make("WatchApplicationOwner", {
  payload: Schema.Struct({}),
  success: ApplicationOwnerState,
  stream: true,
})

export const Application = {
  watchApplicationOwner: WatchApplicationOwner,
  checkApplicationUpdate: command("CheckApplicationUpdate", {}),
  downloadApplicationUpdate: command("DownloadApplicationUpdate", {}),
  discardApplicationUpdate: command("DiscardApplicationUpdate", {}),
  installApplicationUpdate: command("InstallApplicationUpdate", {}),
  setApplicationAutoDownload: command("SetApplicationAutoDownload", { enabled: Schema.Boolean }),
  setLaunchAtLogin: command("SetLaunchAtLogin", { enabled: Schema.Boolean }),
  restartApplication: command("RestartApplication", {}),
  quitApplication: command("QuitApplication", {}),
}
