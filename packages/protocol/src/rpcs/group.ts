import { RpcGroup } from "@effect/rpc"
import * as Agent from "./agent"
import * as Session from "./session"
import * as Connection from "./connection"
import * as Config from "./config"
import * as Files from "./files"
import * as Git from "./git"
import * as Skills from "./skills"
import * as Shell from "./shell"
import * as Stream from "./stream"
import * as LocalInference from "./local-inference"
import * as Onboarding from "./onboarding"
import { AcnRpcDemand } from "./middleware"
import { WatchMirroredStates } from "./mirrored-state"

/**
 * `AcnRpcDemand` scopes an ACN residency lease to the complete RPC effect.
 * Finite work therefore uses it: accepted work cannot lose the ACN before
 * completion.
 *
 * Subscription RPCs deliberately do not use `AcnRpcDemand`. Effect RPC keeps
 * middleware active for the complete stream lifetime, so applying it to an
 * open observer would hold the ACN resident forever and make idle shutdown
 * impossible. Their independent `AcnSubscriptionRpc` wire protocol carries
 * keepalive, suspension, and termination without turning observation into
 * demand.
 */
const AcnSubscriptions = RpcGroup.make(
  Stream.StreamDisplayView,
  WatchMirroredStates,
  Session.StreamActiveSessionStatuses,
  Files.WatchFile,
)

const AcnDemandRpcs = RpcGroup.make(
  Session.PreloadSession,
  Session.ReleaseSessionPreload,
  Session.CreateSession,
  Session.ListSessions,
  Session.ListSessionCwds,
  Session.GetSession,
  Session.DeleteSession,
  Agent.SendMessage,
  Agent.StartGoal,
  Agent.Interrupt,
  Files.UploadAttachment,
  Config.UpdateProviderAuth,
  Config.GetProviderAuth,
  Config.ListProviderAuth,
  Config.ProviderModelCatalogMirror.getRpc,
  Config.RefreshModelCatalog,
  Config.ModelSlotsMirror.getRpc,
  Config.AssignSlot,
  Config.ClearSlot,
  Config.GetCloudUsage,
  LocalInference.LocalInferenceHardwareMirror.getRpc,
  LocalInference.LocalModelsMirror.getRpc,
  LocalInference.DownloadRecommendedModel,
  LocalInference.RetryModelDownload,
  LocalInference.CancelModelDownload,
  LocalInference.DismissModelDownloadFailure,
  LocalInference.DeleteLocalModel,
  LocalInference.LoadModel,
  LocalInference.UnloadModel,
  Onboarding.GetOnboardingState,
  Onboarding.CompleteOnboardingFlow,
  Files.ListFiles,
  Files.ReadFile,
  Files.CheckFileExists,
  Files.ResolvePath,
  Files.SearchMentions,
  Files.SearchDirectories,
  Git.GetGitRecentFiles,
  Skills.ListSkills,
  Skills.GetSkill,
  Shell.RunBash,
  Stream.ResyncDisplayView,
  Stream.SetDisplayViewShape,
).middleware(AcnRpcDemand)

/**
 * The group structure is the lifecycle policy: health is neutral,
 * subscriptions use their wrapping protocol without demand, and every finite
 * RPC receives `AcnRpcDemand` together at this single boundary.
 */
export const MagnitudeRpcs = RpcGroup.make(Connection.Health).merge(
  AcnSubscriptions,
  AcnDemandRpcs,
)
