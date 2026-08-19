import { RpcGroup } from "@effect/rpc"
import { Context } from "effect"
import * as Agent from "./agent"
import * as Session from "./session"
import * as Project from "./project"
import * as ProjectFiles from "./project-files"
import * as Connection from "./connection"
import * as Config from "./config"
import * as Files from "./files"
import * as Git from "./git"
import * as Skills from "./skills"
import * as Shell from "./shell"
import * as Stream from "./stream"
import * as LocalInference from "./local-inference"
import * as Onboarding from "./onboarding"
import * as ClientLease from "./client-lease"
import { AcnRpcDemand } from "./middleware"
import { WatchMirroredStates } from "./mirrored-state"
import { AcnRpcRecoveryPolicyTag } from "./recovery-policy"

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
  Project.StreamProjectChanges,
  ProjectFiles.WatchProjectFiles,
  Files.WatchFile,
)

const ReplaySafeDemandRpcs = RpcGroup.make(
  Session.ReleaseSessionPreload,
  Session.ListSessions,
  Session.ListSessionCwds,
  Session.GetSession,
  Session.ArchiveSession,
  Session.RestoreSession,
  Session.SetSessionPinned,
  Project.ListProjects,
  Project.CreateProject,
  Project.EditProject,
  Project.RemoveProject,
  Project.RestoreProject,
  ProjectFiles.ListProjectDirectory,
  ProjectFiles.ReadProjectFile,
  Config.UpdateProviderAuth,
  Config.GetProviderAuth,
  Config.ListProviderAuth,
  Config.ProviderModelCatalogMirror.getRpc,
  Config.ModelSlotsMirror.getRpc,
  Config.AssignSlot,
  Config.ClearSlot,
  Config.SetModelFavorite,
  Config.GetCloudUsage,
  LocalInference.LocalInferenceHardwareMirror.getRpc,
  LocalInference.LocalModelsMirror.getRpc,
  LocalInference.PreviewModelLoad,
  Onboarding.OnboardingMirror.getRpc,
  Onboarding.CompleteOnboarding,
  Files.ListFiles,
  Files.ReadFile,
  Files.CheckFileExists,
  Files.ResolvePath,
  Files.SearchMentions,
  Files.SearchDirectories,
  Git.GetGitRecentFiles,
  Skills.ListSkills,
  Skills.GetSkill,
  Stream.ResyncDisplayView,
  Stream.SetDisplayViewShape,
).annotateRpcs(AcnRpcRecoveryPolicyTag, "ReplaySafe")

const AtMostOnceDemandRpcs = RpcGroup.make(
  Session.PreloadSession,
  Session.CreateSession,
  Session.DeleteArchivedSession,
  Project.RevealProjectSource,
  ProjectFiles.WriteProjectFile,
  ProjectFiles.DeleteProjectFile,
  ProjectFiles.MoveProjectEntry,
  Agent.SendMessage,
  Agent.StartGoal,
  Agent.Interrupt,
  Files.UploadAttachment,
  Config.RefreshModelCatalog,
  LocalInference.ReconcileCatalogModel,
  LocalInference.CancelModelDownload,
  LocalInference.DismissModelDownloadFailure,
  LocalInference.DeleteLocalModel,
  LocalInference.LoadModel,
  LocalInference.StopModel,
  Shell.RunBash,
).annotateRpcs(AcnRpcRecoveryPolicyTag, "AtMostOnce")

const AcnDemandRpcs = ReplaySafeDemandRpcs.merge(AtMostOnceDemandRpcs).middleware(AcnRpcDemand)

/**
 * The group structure is the lifecycle policy: health is neutral,
 * subscriptions use their wrapping protocol without demand, and every finite
 * RPC receives `AcnRpcDemand` together at this single boundary.
 */
const AcnNeutralRpcs = RpcGroup.make(
  Connection.Health,
  ClientLease.RenewClientLease,
  ClientLease.ReleaseClientLease,
).annotateRpcs(AcnRpcRecoveryPolicyTag, "ReplaySafe")

export const MagnitudeRpcs = AcnNeutralRpcs.merge(
  AcnSubscriptions,
  AcnDemandRpcs,
)

export const acnRpcRecoveryPolicy = (tag: string) => {
  const rpc = MagnitudeRpcs.requests.get(tag)
  if (rpc === undefined) throw new TypeError(`Unknown ACN RPC ${tag}`)
  const policy = Context.getOption(rpc.annotations, AcnRpcRecoveryPolicyTag)
  if (policy._tag === "None") throw new TypeError(`Finite ACN RPC ${tag} has no recovery policy`)
  return policy.value
}
