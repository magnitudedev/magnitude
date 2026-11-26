import { rpcGroup } from "../rpc-tree"
import { Connection } from "./connection"
import { Models } from "./models"
import { Changes } from "./changes"
import { Configuration } from "./configuration"
import { Connections } from "./connections"
import { Application } from "./application"
import { HuggingFace } from "./hugging-face"
import { Chat } from "./chat"

export const MagnitudeRpcs = {
  chat: Chat,
  connection: Connection,
  models: Models,
  changes: Changes,
  configuration: Configuration,
  connections: Connections,
  application: Application,
  huggingFace: HuggingFace,
}
export const AcnRpcGroup = rpcGroup(MagnitudeRpcs)

/**
 * The strict allowlist of RPCs reachable by a signed-in container web UI session.
 * Every other RPC in `AcnRpcGroup` is loopback-only by construction.
 */
export const RemoteMagnitudeRpcs = {
  chat: Chat,
  connection: Connection,
  models: Models,
  changes: Changes,
  configuration: {
    getModelStorage: Configuration.getModelStorage,
    setModelStorage: Configuration.setModelStorage,
    browseDirectories: Configuration.browseDirectories,
    getNetworkAccess: Configuration.getNetworkAccess,
    getServerMachine: Configuration.getServerMachine,
    getOnboarding: Configuration.getOnboarding,
    completeOnboarding: Configuration.completeOnboarding,
  },
  connections: {
    describeHarnessSetup: Connections.describeHarnessSetup,
  },
  application: {
    watchApplicationOwner: Application.watchApplicationOwner,
    checkApplicationUpdate: Application.checkApplicationUpdate,
    restartApplication: Application.restartApplication,
  },
  huggingFace: {
    getHuggingFaceAccount: HuggingFace.getHuggingFaceAccount,
    startHuggingFaceDeviceSignIn: HuggingFace.startHuggingFaceDeviceSignIn,
    getHuggingFaceSignInAttempt: HuggingFace.getHuggingFaceSignInAttempt,
    cancelHuggingFaceSignIn: HuggingFace.cancelHuggingFaceSignIn,
    signOutOfHuggingFace: HuggingFace.signOutOfHuggingFace,
  },
}
export const RemoteAcnRpcGroup = rpcGroup(RemoteMagnitudeRpcs)
