import { Effect, Layer, Option, Schema } from "effect"
import { dirname, join } from "node:path"
import { MacInstallerRequest } from "./mac-installer-command"
import { prepareMacInstallerHelper, nativeMacInstallerCodeVerifier } from "../desktop-native/mac-installer-helper"
import { MacUpdateAdmission, nativeMacUpdateAdmission } from "../desktop-native/mac-update-lease"
import { nativeMacUpdateFilesystem } from "../desktop-native/mac-update-filesystem"
import { nativeMacBundleVerifier } from "../desktop-native/mac-update-validation"
import { MacApplicationInstallation, NativeMacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { MacUpdateInstallationBusy } from "../desktop-native/mac-update-workspace"
import { guardedCommandLayer } from "../desktop-native/guarded-command"

/** Caller retains application admission and establishes prior-version exclusion before entering. */
export const startMacForegroundInstallation = (options: {
  readonly resources: string
  readonly stateDirectory: string
  readonly dataDirectory: string
  readonly version: string
  readonly architecture: "arm64" | "x64"
  readonly continuation: MacInstallerRequest["continuation"]
}) => {
  const addon = join(options.resources, "desktop-host.node")
  return Effect.scoped(Effect.gen(function* () {
    const bundle = dirname(dirname(options.resources))
    const installation = yield* MacApplicationInstallation
    if (yield* installation.isInstalling(bundle)) return yield* new MacUpdateInstallationBusy()
    const admission = yield* MacUpdateAdmission
    const lease = yield* admission.exclusive(bundle)
    if (Option.isNone(lease)) return yield* new MacUpdateInstallationBusy()
    const helper = yield* prepareMacInstallerHelper({ ...options, lease: lease.value })
    const request = yield* Schema.encode(Schema.parseJson(MacInstallerRequest))({ protocol: 1, bundle,
      stateDirectory: options.stateDirectory, dataDirectory: options.dataDirectory, continuation: options.continuation })
    return yield* lease.value.replaceProcess(helper.executable, ["_install-mac-application-update", request], process.env)
  })).pipe(Effect.provide([nativeMacUpdateAdmission(addon), nativeMacUpdateFilesystem(addon), nativeMacBundleVerifier(addon),
    NativeMacApplicationInstallation, nativeMacInstallerCodeVerifier.pipe(Layer.provide(guardedCommandLayer(join(options.resources, "magnitude-command"))))]))
}
