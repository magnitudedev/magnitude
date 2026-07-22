import { RpcGroup } from "@effect/rpc"
import * as Agent from "./agent"
import * as Session from "./session"
import * as Connection from "./connection"
import * as Config from "./config"
import * as Files from "./files"
import * as Git from "./git"
import * as Skills from "./skills"
import * as Shell from "./shell"
import * as Events from "./events"
import * as Stream from "./stream"
import * as LocalInference from "./local-inference"
import * as Onboarding from "./onboarding"
import { AcnRpcCommandActivity } from "./middleware"
import { WatchMirroredStates } from "./mirrored-state"

export const MagnitudeRpcs = RpcGroup.make(
  // Liveness — no activity tracking
  Connection.Health,

  // Display view stream — long-lived, no middleware
  Stream.StreamDisplayView,

  // Unary commands — mark command activity
  Session.PreloadSession.middleware(AcnRpcCommandActivity),
  Session.ReleaseSessionPreload.middleware(AcnRpcCommandActivity),
  Session.CreateSession.middleware(AcnRpcCommandActivity),
  Session.ListSessions.middleware(AcnRpcCommandActivity),
  Session.ListSessionCwds.middleware(AcnRpcCommandActivity),
  Session.GetSession.middleware(AcnRpcCommandActivity),
  Session.DeleteSession.middleware(AcnRpcCommandActivity),
  Agent.SendMessage.middleware(AcnRpcCommandActivity),
  Agent.StartGoal.middleware(AcnRpcCommandActivity),
  Agent.Interrupt.middleware(AcnRpcCommandActivity),
  Files.UploadAttachment.middleware(AcnRpcCommandActivity),
  Config.UpdateProviderAuth.middleware(AcnRpcCommandActivity),
  Config.GetProviderAuth.middleware(AcnRpcCommandActivity),
  Config.ListProviderAuth.middleware(AcnRpcCommandActivity),
  Config.ProviderModelCatalogMirror.getRpc.middleware(AcnRpcCommandActivity),
  Config.RefreshModelCatalog.middleware(AcnRpcCommandActivity),
  Config.ModelSlotsMirror.getRpc.middleware(AcnRpcCommandActivity),
  Config.UpdateModelSlot.middleware(AcnRpcCommandActivity),
  Config.GetCloudUsage.middleware(AcnRpcCommandActivity),
  LocalInference.LocalInferenceHardwareMirror.getRpc.middleware(AcnRpcCommandActivity),
  LocalInference.LocalModelInventoryMirror.getRpc.middleware(AcnRpcCommandActivity),
  LocalInference.DownloadLocalModel.middleware(AcnRpcCommandActivity),
  LocalInference.DeleteLocalModel.middleware(AcnRpcCommandActivity),
  LocalInference.LoadModelSlot.middleware(AcnRpcCommandActivity),
  LocalInference.ReloadModelSlot.middleware(AcnRpcCommandActivity),
  LocalInference.UnloadModelSlot.middleware(AcnRpcCommandActivity),
  Onboarding.GetOnboardingState.middleware(AcnRpcCommandActivity),
  Onboarding.CompleteOnboardingFlow.middleware(AcnRpcCommandActivity),
  Files.ListFiles.middleware(AcnRpcCommandActivity),
  Files.ReadFile.middleware(AcnRpcCommandActivity),
  Files.CheckFileExists.middleware(AcnRpcCommandActivity),
  Files.WatchFile.middleware(AcnRpcCommandActivity),
  Files.ResolvePath.middleware(AcnRpcCommandActivity),
  Files.SearchMentions.middleware(AcnRpcCommandActivity),
  Files.SearchDirectories.middleware(AcnRpcCommandActivity),
  Git.GetGitRecentFiles.middleware(AcnRpcCommandActivity),
  Skills.ListSkills.middleware(AcnRpcCommandActivity),
  Skills.GetSkill.middleware(AcnRpcCommandActivity),
  Shell.RunBash.middleware(AcnRpcCommandActivity),
  Stream.ResyncDisplayView.middleware(AcnRpcCommandActivity),
  Stream.SetDisplayViewShape.middleware(AcnRpcCommandActivity),
  Stream.CloseDisplayView.middleware(AcnRpcCommandActivity),

  // Long-running subscriptions
  WatchMirroredStates,
  Session.StreamActiveSessionStatuses,
  Events.StreamEvents,
)
