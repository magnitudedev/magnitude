import { ChatService } from "../chat/service";
import { ChatModelDefaultsStore } from "../chat/model-defaults";
import { ChatInference } from "../chat/inference";
import { ServingUsage } from "../serving-usage";
import { AcnRpcGroup } from "@magnitudedev/acn-protocol";
import { Cause, Chunk, Effect, Option, Stream } from "effect";
import { ModelSlotController } from "../model-slot-controller";
import { InferenceActivity } from "../inference-activity";
import { ACN_VERSION } from "../version";
import { makeHealthResponse } from "../identity";
import { AcnServiceLifecycle } from "../service-lifecycle";
import { AcnChanges } from "../changes";
import { ModelCatalog } from "../model-catalog";
import { Recommendations } from "../recommendations";
import { ModelCommands } from "../model-commands";
import { CustomModelCommands } from "../custom-model-commands";
import { LocalInferenceHardware } from "../local-inference-hardware";
import { ServerSettings } from "../server-settings";
import { AcnHarnessConnections } from "../harness-connections";
import { AcnOwner, requestOwner, watchApplicationOwner } from "../application-owner";
import { HuggingFaceAccount } from "../hugging-face-account";

/** Exhaustive server implementation of the composed ACN boundary. */
export const AcnBoundaryLive = AcnRpcGroup.toLayer(Effect.gen(function* () {
    const lifecycle = yield* AcnServiceLifecycle;
    const chats = yield* ChatService;
    const modelDefaults = yield* ChatModelDefaultsStore;
    const inference = yield* ChatInference;
    const modelSlots = yield* ModelSlotController;
    const servingUsage = yield* ServingUsage;
    const inferenceActivity = yield* InferenceActivity;
    const changes = yield* AcnChanges;
    const modelCatalog = yield* ModelCatalog;
    const recommendations = yield* Recommendations;
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
      ListChatModelDefaults: () => observeRpcDefects("ListChatModelDefaults", modelDefaults.list),
      SaveChatModelDefaults: ({ model, defaults }) =>
        observeRpcDefects("SaveChatModelDefaults", modelDefaults.save(model, defaults)),
      ClearChatModelDefaults: ({ model }) =>
        observeRpcDefects("ClearChatModelDefaults", modelDefaults.clear(model)),

      // Connection
      Health: () => lifecycle.state.pipe(
        Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
      ),

      // Models & Usage
      RefreshModelCatalog: ({ providerId }) =>
        observeRpcDefects(
          "RefreshModelCatalog",
          modelCatalog.refresh(providerId).pipe(Effect.as({})),
        ),

      GetModelSlots: () =>
        observeRpcDefects("GetModelSlots", modelSlots.state),

      GetModelCatalog: () =>
        observeRpcDefects("GetModelCatalog", modelCatalog.state),

      GetRecommendations: ({ developerCountry }) =>
        observeRpcDefects("GetRecommendations", recommendations.get(developerCountry)),

      GetServingUsage: request => servingUsage.read(request),
      ListServableModels: () => observeRpcDefects("ListServableModels", inference.models),

      // Configuration
      GetModelStorage: () => observeRpcDefects("GetModelStorage", serverSettings.modelStorage),
      SetModelStorage: ({ path }) => observeRpcDefects("SetModelStorage", serverSettings.setModelStorage(path).pipe(Effect.as({}))),
      BrowseDirectories: ({ path }) => observeRpcDefects("BrowseDirectories", serverSettings.browseDirectories(path)),
      GetNetworkAccess: () => observeRpcDefects("GetNetworkAccess", serverSettings.networkAccess),
      SetNetworkAccess: change => observeRpcDefects("SetNetworkAccess", serverSettings.setNetworkAccess(change).pipe(Effect.as({}))),
      RegenerateNetworkApiKey: () => observeRpcDefects("RegenerateNetworkApiKey", serverSettings.regenerateNetworkApiKey.pipe(Effect.as({}))),
      GetServerMachine: () => observeRpcDefects("GetServerMachine", serverSettings.machine),
      GetOnboarding: () => observeRpcDefects("GetOnboarding", serverSettings.onboarding),
      CompleteOnboarding: () => observeRpcDefects("CompleteOnboarding", serverSettings.completeOnboarding.pipe(Effect.as({}))),

      // Hugging Face
      GetHuggingFaceAccount: () => observeRpcDefects("GetHuggingFaceAccount", huggingFace.account),
      StartHuggingFaceSignIn: () => observeRpcDefects("StartHuggingFaceSignIn", huggingFace.startBrowserSignIn),
      StartHuggingFaceDeviceSignIn: () => observeRpcDefects("StartHuggingFaceDeviceSignIn", huggingFace.startDeviceSignIn),
      GetHuggingFaceSignInAttempt: ({ attemptId }) => observeRpcDefects("GetHuggingFaceSignInAttempt", huggingFace.attemptOutcome(attemptId)),
      CancelHuggingFaceSignIn: ({ attemptId }) => observeRpcDefects("CancelHuggingFaceSignIn", huggingFace.cancelSignIn(attemptId).pipe(Effect.as({}))),
      SignOutOfHuggingFace: () => observeRpcDefects("SignOutOfHuggingFace", huggingFace.signOut.pipe(Effect.as({}))),

      // Harness connections
      WatchHarnessConnections: () => observeRpcStreamDefects("WatchHarnessConnections", harnessConnections.watch),
      ConnectHarness: request => observeRpcDefects("ConnectHarness", harnessConnections.connect(request)),
      SyncHarnessConnections: ({ harness }) => observeRpcDefects("SyncHarnessConnections", harnessConnections.sync(harness).pipe(Effect.as({}))),
      DisconnectHarness: ({ harness }) => observeRpcDefects("DisconnectHarness", harnessConnections.disconnect(harness).pipe(Effect.as({}))),
      DescribeHarnessSetup: request => observeRpcDefects("DescribeHarnessSetup", harnessConnections.describe(request)),

      // Application owner
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

      CheckHuggingFaceRepository: (request) =>
        observeRpcDefects("CheckHuggingFaceRepository", customModelCommands.checkHuggingFace(request)),

      // Like every ACN subscription, it stays open after its last update until the subscriber leaves.
      InspectModelCandidate: ({ reference }) =>
        observeRpcDefects("InspectModelCandidate", customModelCommands.inspectCandidateResult(reference)),

      StreamModelCandidateInspection: ({ reference }) =>
        observeRpcStreamDefects(
          "StreamModelCandidateInspection",
          customModelCommands.inspectCandidate(reference).pipe(Stream.concat(Stream.never)),
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

      // Streams
      StreamChanges: () => observeRpcStreamDefects("StreamChanges", changes.stream),
    };
  })
);
