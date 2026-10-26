import { ChatService } from "../chat/service";
import { ChatInference } from "../chat/inference";
import { ServingUsage } from "../serving-usage";
import {
  AcnRpcGroup, Configuration,
  Models,
} from "@magnitudedev/acn-protocol";
import { Cause, Chunk, Effect, Option, Stream } from "effect";
import { ProviderCredentials } from "../provider-credentials";
import { ProviderModelCatalog } from "../provider-model-catalog";
import { ModelSlotController } from "../model-slot-controller";
import { MagnitudeCloudUsage } from "../magnitude-cloud-usage";
import { InferenceActivity } from "../inference-activity";
import { ACN_VERSION } from "../version";
import { makeHealthResponse } from "../identity";
import { AcnServiceLifecycle } from "../service-lifecycle";
import { getSkill, listSkills } from "../skill-ops";
import { AcnChanges } from "../changes";
import { ModelCatalog } from "../model-catalog";
import { ModelCommands } from "../model-commands";
import { CustomModelCommands } from "../custom-model-commands";
import { LocalInferenceHardware } from "../local-inference-hardware";
import { FileMentionSearcher } from "../file-mention-searcher";
import { FileSystemManager } from "../file-system-manager";
import { GitInspector } from "../git-inspector";
import { ProjectFileManager } from "../project-file-manager";
import { ProjectInspector } from "../project-inspector";
import { ProjectManager } from "../project-manager";
import { ProjectStore } from "../project-store";
import { ServerSettings } from "../server-settings";
import { AcnHarnessConnections } from "../harness-connections";
import { AcnOwner, requestOwner, watchApplicationOwner } from "../application-owner";
import { HuggingFaceAccount } from "../hugging-face-account";

/** Exhaustive server implementation of the composed ACN boundary. */
export const AcnBoundaryLive = AcnRpcGroup.toLayer(Effect.gen(function* () {
    const lifecycle = yield* AcnServiceLifecycle;
    const chats = yield* ChatService;
    const inference = yield* ChatInference;
    const projectStore = yield* ProjectStore;
    const projectManager = yield* ProjectManager;
    const projectInspector = yield* ProjectInspector;
    const projectFiles = yield* ProjectFileManager;
    const fileSystemManager = yield* FileSystemManager;
    const fileMentionSearcher = yield* FileMentionSearcher;
    const gitInspector = yield* GitInspector;
    const providerCredentials = yield* ProviderCredentials;
    const providerModelCatalog = yield* ProviderModelCatalog;
    const modelSlots = yield* ModelSlotController;
    const cloudUsage = yield* MagnitudeCloudUsage;
    const servingUsage = yield* ServingUsage;
    const inferenceActivity = yield* InferenceActivity;
    const changes = yield* AcnChanges;
    const modelCatalog = yield* ModelCatalog;
    const modelCommands = yield* ModelCommands;
    const customModelCommands = yield* CustomModelCommands;
    const localInferenceHardware = yield* LocalInferenceHardware;
    const serverSettings = yield* ServerSettings;
    const harnessConnections = yield* AcnHarnessConnections;
    const owner = yield* AcnOwner;
    const huggingFace = yield* HuggingFaceAccount;
    const ownerRequest = (name: string, request: Parameters<typeof requestOwner>[0]) => observeRpcDefects(name, requestOwner(request).pipe(Effect.provideService(AcnOwner, owner)));
    // Observe programming defects without changing the Cause. Expected domain
    // failures stay typed, defects stay defects, and interruption is preserved.
    const observeRpcDefects = <A, E, R>(
      label: string,
      eff: Effect.Effect<A, E, R>
    ): Effect.Effect<A, E, R> =>
      eff.pipe(
        Effect.tapErrorCause((cause) =>
          Chunk.isEmpty(Cause.defects(cause))
            ? Effect.void
            : Effect.logFatal(`RPC defect in ${label}`).pipe(
                Effect.annotateLogs({ defect: Cause.pretty(cause) })
              )
        )
      );

    const observeRpcStreamDefects = <A, E, R>(
      label: string,
      stream: Stream.Stream<A, E, R>
    ): Stream.Stream<A, E, R> =>
      stream.pipe(
        Stream.tapErrorCause((cause) =>
          Chunk.isEmpty(Cause.defects(cause))
            ? Effect.void
            : Effect.logFatal(`RPC stream defect in ${label}`).pipe(
                Effect.annotateLogs({ defect: Cause.pretty(cause) })
              )
        )
      );

    return {
      // Chats
      ListChats: ({ offset, limit }) => observeRpcDefects("ListChats", chats.list(offset, limit)),
      StartChat: (payload) => observeRpcDefects("StartChat", chats.start(payload)),
      SendChatMessage: (payload) => observeRpcDefects("SendChatMessage", chats.send(payload)),
      RetryChatResponse: (payload) => observeRpcDefects("RetryChatResponse", chats.retry(payload)),
      RenameChat: (payload) => observeRpcDefects("RenameChat", chats.rename(payload)),
      StopChatResponse: (payload) => observeRpcDefects("StopChatResponse", chats.stop(payload)),
      RemoveChat: ({ chatId }) => observeRpcDefects("RemoveChat", chats.remove(chatId)),
      WatchChat: ({ chatId }) => observeRpcStreamDefects("WatchChat", chats.watch(chatId)),
      // Connection
      Health: () => lifecycle.state.pipe(
        Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
      ),
      ListProjects: (payload) =>
        observeRpcDefects("ListProjects", projectStore.page(payload)),

      CreateProject: (payload) =>
        observeRpcDefects("CreateProject", projectManager.create(payload)),

      EditProject: (payload) =>
        observeRpcDefects("EditProject", projectManager.edit(payload)),

      RemoveProject: ({ projectId }) =>
        observeRpcDefects("RemoveProject", projectManager.remove(projectId)),

      RestoreProject: ({ projectId }) =>
        observeRpcDefects("RestoreProject", projectManager.restore(projectId)),

      RevealProjectSource: ({ projectId }) =>
        observeRpcDefects(
          "RevealProjectSource",
          projectManager.reveal(projectId).pipe(Effect.as({})),
        ),

      InspectProject: ({ projectId }) =>
        observeRpcDefects("InspectProject", projectInspector.inspect(projectId)),

      ListProjectDirectory: ({ projectId, directory }) =>
        observeRpcDefects("ListProjectDirectory", projectFiles.listDirectory(projectId, directory)),

      WatchProjectFiles: ({ projectId }) =>
        observeRpcStreamDefects("WatchProjectFiles", projectFiles.watchChanges(projectId)),

      ReadProjectFile: ({ projectId, path }) =>
        observeRpcDefects("ReadProjectFile", projectFiles.readFile(projectId, path)),

      WriteProjectFile: (payload) =>
        observeRpcDefects("WriteProjectFile", projectFiles.writeFile(payload)),

      DeleteProjectFile: (payload) =>
        observeRpcDefects("DeleteProjectFile", projectFiles.deleteFile(payload).pipe(Effect.as({}))),

      MoveProjectEntry: (payload) =>
        observeRpcDefects("MoveProjectEntry", projectFiles.moveEntry(payload)),

      // Config
      UpdateProviderAuth: ({ providerId, auth }) =>
        observeRpcDefects(
          "UpdateProviderAuth",
          providerCredentials.update(providerId, auth).pipe(Effect.as({}))
        ),

      GetProviderAuth: ({ providerId }) =>
        observeRpcDefects(
          "GetProviderAuth",
          providerCredentials.get(providerId).pipe(Effect.map((auth) => ({ auth })))
        ),

      ListProviderAuth: () =>
        observeRpcDefects(
          "ListProviderAuth",
          providerCredentials.list.pipe(Effect.map((auths) => ({ auths: Object.fromEntries(auths) })))
        ),

      RefreshModelCatalog: ({ providerId }) =>
        observeRpcDefects(
          "RefreshModelCatalog",
          modelCatalog.refresh(providerId).pipe(Effect.as({})),
        ),

      GetModelSlots: () =>
        observeRpcDefects("GetModelSlots", modelSlots.state),

      GetModelCatalog: () =>
        observeRpcDefects("GetModelCatalog", modelCatalog.state),

      GetServingUsage: request => servingUsage.read(request),
      ListServableModels: () => observeRpcDefects("ListServableModels", inference.models),

      GetModelStorage: () => observeRpcDefects("GetModelStorage", serverSettings.modelStorage),
      SetModelStorage: ({ path }) => observeRpcDefects("SetModelStorage", serverSettings.setModelStorage(path).pipe(Effect.as({}))),
      BrowseDirectories: ({ path }) => observeRpcDefects("BrowseDirectories", serverSettings.browseDirectories(path)),
      GetNetworkAccess: () => observeRpcDefects("GetNetworkAccess", serverSettings.networkAccess),
      SetNetworkAccess: change => observeRpcDefects("SetNetworkAccess", serverSettings.setNetworkAccess(change).pipe(Effect.as({}))),
      RegenerateNetworkApiKey: () => observeRpcDefects("RegenerateNetworkApiKey", serverSettings.regenerateNetworkApiKey.pipe(Effect.as({}))),
      GetServerMachine: () => observeRpcDefects("GetServerMachine", serverSettings.machine),

      GetHuggingFaceAccount: () => observeRpcDefects("GetHuggingFaceAccount", huggingFace.account),
      StartHuggingFaceSignIn: () => observeRpcDefects("StartHuggingFaceSignIn", huggingFace.startBrowserSignIn),
      StartHuggingFaceDeviceSignIn: () => observeRpcDefects("StartHuggingFaceDeviceSignIn", huggingFace.startDeviceSignIn),
      GetHuggingFaceSignInAttempt: ({ attemptId }) => observeRpcDefects("GetHuggingFaceSignInAttempt", huggingFace.attemptOutcome(attemptId)),
      CancelHuggingFaceSignIn: ({ attemptId }) => observeRpcDefects("CancelHuggingFaceSignIn", huggingFace.cancelSignIn(attemptId).pipe(Effect.as({}))),
      SignOutOfHuggingFace: () => observeRpcDefects("SignOutOfHuggingFace", huggingFace.signOut.pipe(Effect.as({}))),

      WatchHarnessConnections: () => observeRpcStreamDefects("WatchHarnessConnections", harnessConnections.watch),
      ConnectHarness: request => observeRpcDefects("ConnectHarness", harnessConnections.connect(request)),
      SyncHarnessConnections: ({ harness }) => observeRpcDefects("SyncHarnessConnections", harnessConnections.sync(harness).pipe(Effect.as({}))),
      DisconnectHarness: ({ harness }) => observeRpcDefects("DisconnectHarness", harnessConnections.disconnect(harness).pipe(Effect.as({}))),
      DescribeHarnessSetup: request => observeRpcDefects("DescribeHarnessSetup", harnessConnections.describe(request)),

      GetServiceActivity: () => observeRpcDefects("GetServiceActivity", inferenceActivity.count.pipe(
        Effect.map((inferenceRequests) => ({ inferenceRequests })),
      )),
      WatchApplicationOwner: () => observeRpcStreamDefects("WatchApplicationOwner", watchApplicationOwner.pipe(Stream.provideService(AcnOwner, owner))),
      CheckApplicationUpdate: () => ownerRequest("CheckApplicationUpdate", { _tag: "CheckUpdate" }),
      DownloadApplicationUpdate: () => ownerRequest("DownloadApplicationUpdate", { _tag: "DownloadUpdate" }),
      DiscardApplicationUpdate: () => ownerRequest("DiscardApplicationUpdate", { _tag: "DiscardUpdate" }),
      InstallApplicationUpdate: () => ownerRequest("InstallApplicationUpdate", { _tag: "InstallUpdate" }),
      SetApplicationAutoDownload: ({ enabled }) => ownerRequest("SetApplicationAutoDownload", { _tag: "SetAutoDownload", enabled }),
      SetLaunchAtLogin: ({ enabled }) => ownerRequest("SetLaunchAtLogin", { _tag: "SetLoginStartup", enabled }),
      RestartApplication: () => ownerRequest("RestartApplication", { _tag: "RestartService" }),
      QuitApplication: () => ownerRequest("QuitApplication", { _tag: "Quit" }),
      GetLocalInferenceEnvironment: () =>
        observeRpcDefects("GetLocalInferenceEnvironment", localInferenceHardware.state),

      SyncLocalModel: ({ modelId }) =>
        observeRpcDefects("SyncLocalModel", modelCommands.sync(modelId)),

      DownloadCatalogComponent: ({ modelId, component }) =>
        observeRpcDefects("DownloadCatalogComponent", modelCommands.downloadCatalogComponent(modelId, component)),

      CancelLocalModelSync: ({ modelId }) =>
        observeRpcDefects("CancelLocalModelSync", modelCommands.cancelSync(modelId)),

      AcknowledgeLocalModelSyncFailure: ({ modelId }) =>
        observeRpcDefects("AcknowledgeLocalModelSyncFailure", modelCommands.acknowledgeSyncFailure(modelId)),

      RemoveLocalModel: ({ modelId }) =>
        observeRpcDefects("RemoveLocalModel", modelCommands.remove(modelId)),

      LoadLocalModel: ({ modelId }) =>
        observeRpcDefects("LoadLocalModel", modelCommands.loadModel(modelId)),

      StopActiveLocalModel: () =>
        observeRpcDefects("StopActiveLocalModel", modelCommands.stopActiveModel),

      StopLocalModel: ({ modelId }) =>
        observeRpcDefects("StopLocalModel", modelCommands.stopModel(modelId)),

      LoadModelSlot: ({ slotId }) =>
        observeRpcDefects("LoadModelSlot", modelCommands.loadSlot(slotId)),

      PreviewModelSlotLoad: ({ slotId }) =>
        observeRpcDefects("PreviewModelSlotLoad", modelCommands.previewSlotLoad(slotId)),

      StopModelSlot: ({ slotId }) =>
        observeRpcDefects("StopModelSlot", modelCommands.stopSlot(slotId)),

      SearchHuggingFaceModels: (request) =>
        observeRpcDefects("SearchHuggingFaceModels", customModelCommands.searchHuggingFace(request)),

      InspectModelCandidate: ({ reference }) =>
        observeRpcDefects("InspectModelCandidate", customModelCommands.inspectCandidate(reference)),

      // Like every ACN subscription, it stays open after its last update until the subscriber leaves.
      StreamModelCandidateInspection: ({ reference }) =>
        observeRpcStreamDefects(
          "StreamModelCandidateInspection",
          customModelCommands.inspectCandidateWithProgress(reference).pipe(Stream.concat(Stream.never)),
        ),

      ListDrafterCandidates: ({ target, other }) =>
        observeRpcDefects(
          "ListDrafterCandidates",
          customModelCommands.listDrafterCandidates(target, other),
        ),

      ListModelLocations: () =>
        observeRpcDefects("ListModelLocations", customModelCommands.listLocations.pipe(Effect.map((locations) => ({ locations })))),

      AddHuggingFaceModel: (request) =>
        observeRpcDefects("AddHuggingFaceModel", customModelCommands.addHuggingFaceModel(request)),


      SetModelDrafter: ({ modelId, drafter }) =>
        observeRpcDefects(
          "SetModelDrafter",
          customModelCommands.setModelDrafter(modelId, drafter),
        ),

      HideModel: ({ modelId }) =>
        observeRpcDefects("HideModel", customModelCommands.hide(modelId).pipe(Effect.as({}))),

      UnhideModel: ({ modelId }) =>
        observeRpcDefects("UnhideModel", customModelCommands.unhide(modelId).pipe(Effect.as({}))),

      AddModelFolder: ({ path }) =>
        observeRpcDefects("AddModelFolder", customModelCommands.addFolder(path).pipe(Effect.as({}))),

      RemoveModelFolder: ({ path }) =>
        observeRpcDefects("RemoveModelFolder", customModelCommands.removeFolder(path).pipe(Effect.as({}))),

      SetModelLocationEnabled: ({ kind, enabled }) =>
        observeRpcDefects("SetModelLocationEnabled", customModelCommands.setLocationEnabled(kind, enabled).pipe(Effect.as({}))),

      AssignModelSlot: ({ slotId, selection }) =>
        observeRpcDefects(
          "AssignSlot",
          modelSlots.updateModelSlot(slotId, Option.some(selection)).pipe(Effect.as({})),
        ),

      ClearModelSlot: ({ slotId }) =>
        observeRpcDefects(
          "ClearSlot",
          modelSlots.updateModelSlot(slotId, Option.none()).pipe(Effect.as({})),
        ),

      SetModelFavorite: ({ model, favorite }) =>
        observeRpcDefects(
          "SetModelFavorite",
          modelSlots.setModelFavorite(model, favorite).pipe(Effect.as({})),
        ),

      GetCloudUsage: (payload) =>
        observeRpcDefects(
          "GetCloudUsage",
          cloudUsage.get({
            ...(payload.period !== undefined ? { period: payload.period } : {}),
            ...(payload.days !== undefined ? { days: payload.days } : {}),
            ...(payload.tz !== undefined ? { tz: payload.tz } : {}),
          })
        ),

      // Server-side operations
      ListFiles: ({ cwd, glob, limit }) =>
        observeRpcDefects(
          "ListFiles",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) =>
              fileMentionSearcher.listFiles(directory, { glob, limit })),
          )
        ),

      ReadFile: ({ cwd, path, format, offset }) =>
        observeRpcDefects(
          "ReadFile",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) =>
              fileMentionSearcher.readFile(directory, { path, format, offset })),
          )
        ),

      CheckFileExists: ({ cwd, path }) =>
        observeRpcDefects(
          "CheckFileExists",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) => fileMentionSearcher.checkFileExists(directory, path)),
          )
        ),

      WatchFile: ({ cwd, path }) =>
        observeRpcStreamDefects(
          "WatchFile",
          Stream.unwrap(fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.map((directory) => fileMentionSearcher.watchFile(directory, path)),
          ))
        ),

      ResolvePath: ({ cwd, path, checkExists }) =>
        observeRpcDefects(
          "ResolvePath",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) =>
              fileMentionSearcher.resolvePath(directory, { path, checkExists })),
          )
        ),

      SearchMentions: ({ cwd, query, limit, visibleLimit, includeRecent }) =>
        observeRpcDefects(
          "SearchMentions",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) =>
              fileMentionSearcher.searchMentions(directory, {
                query,
                limit,
                visibleLimit,
                includeRecent,
              })),
          )
        ),

      SearchDirectories: ({ query, limit }) =>
        observeRpcDefects("SearchDirectories", fileMentionSearcher.searchDirectories({ query, limit })),

      GetGitRecentFiles: ({ cwd, limit }) =>
        observeRpcDefects(
          "GetGitRecentFiles",
          fileSystemManager.normalizeDirectory(cwd).pipe(
            Effect.flatMap((directory) => gitInspector.recentFiles(directory, limit)),
          )
        ),

      ListSkills: ({ cwd }) => observeRpcDefects("ListSkills", listSkills(cwd)),

      GetSkill: ({ cwd, name }) =>
        observeRpcDefects("GetSkill", getSkill(cwd, name)),

      // Streams
      StreamChanges: () => observeRpcStreamDefects("StreamChanges", changes.stream),
    };
  })
);
