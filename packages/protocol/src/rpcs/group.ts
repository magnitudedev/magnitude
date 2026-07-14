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
import { AcnRpcCommandActivity } from "./middleware"

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
  Config.RemoveProviderAuth.middleware(AcnRpcCommandActivity),
  Config.GetProviderAuthSummary.middleware(AcnRpcCommandActivity),
  Config.ListProviderAuthSummaries.middleware(AcnRpcCommandActivity),
  Config.ListPublicSlotProfiles.middleware(AcnRpcCommandActivity),
  Config.UpdateModelConfig.middleware(AcnRpcCommandActivity),
  Config.GetCachedModelList.middleware(AcnRpcCommandActivity),
  Config.RefreshCachedModelList.middleware(AcnRpcCommandActivity),
  Config.GetBalance.middleware(AcnRpcCommandActivity),
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
  Session.StreamActiveSessionStatuses,
  Events.StreamEvents,
)
