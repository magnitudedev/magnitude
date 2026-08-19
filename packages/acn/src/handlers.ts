import {
  MagnitudeRpcs,
  LocalModelMutationFailed,
  SessionOperationFailed,
  ModelSlotMutationRejected,
  type DisplayViewShape,
  type ModelServingConfigurationId,
  ModelDownloadIdSchema,
  type SessionError,
} from "@magnitudedev/acn-protocol";
import { Cause, Chunk, Effect, Option, Stream } from "effect";
import { SessionCommands } from "./session-commands";
import { SessionLifecycle } from "./session-lifecycle";
import { ProviderCredentials } from "./provider-credentials";
import { ProviderModelCatalog } from "./provider-model-catalog";
import { ModelSlotController } from "./model-slot-controller";
import { MagnitudeCloudUsage } from "./magnitude-cloud-usage";
import { ActiveSessionStatusesService } from "./active-session-statuses";
import { DisplayViewStreams } from "./display-view-streams";
import { ACN_VERSION } from "./version";
import { makeHealthResponse } from "./identity";
import { AcnServiceLifecycle } from "./service-lifecycle";
import { AcnDisplayViewIntrospector } from "./introspection";
import { uploadAttachment } from "./attachment-upload";
import {
  checkFileExists,
  getGitRecentFiles,
  getSkill,
  listFiles,
  listSkills,
  readFileOp,
  resolvePath,
  runBash,
  searchDirectories,
  searchMentions,
  watchFile,
} from "./ops";
import { UserBashCommandId, type AppEvent } from "@magnitudedev/agent";
import { createId } from "@magnitudedev/generate-id";
import { Onboarding } from "./onboarding";
import { MirroredStateChanges } from "./mirrored-state";
import { LocalInferenceHardware } from "./local-inference-hardware";
import { LocalModelPackages } from "./local-model-packages";
import { LocalModels } from "./local-models";
import { servableModelBundlePackageIds } from "@magnitudedev/acn-protocol";
import { ClientLeaseManager } from "./client-lease-manager";
import { LocalModelConfigurationCoordinator } from "./local-model-configuration-coordinator";
import { LocalModelConfigurationResolver } from "./local-model-configuration-resolver";
import { localCatalogProviderModelId } from "./local-provider-model-id";
import { IcnModels } from "@magnitudedev/icn";
import { Projects } from "./projects";
import { ProjectFiles } from "./project-files";

const MAX_BASH_OUTPUT_LENGTH = 50_000;

const normalizeBashOutput = (output: string): string =>
  output.length > MAX_BASH_OUTPUT_LENGTH
    ? `${output.slice(0, MAX_BASH_OUTPUT_LENGTH)}\n[truncated]`
    : output;

export const HandlersLive = MagnitudeRpcs.toLayer(
  Effect.gen(function* () {
    const lifecycle = yield* AcnServiceLifecycle;
    const sessionCommands = yield* SessionCommands;
    const sessionLifecycle = yield* SessionLifecycle;
    const projects = yield* Projects;
    const projectFiles = yield* ProjectFiles;
    const providerCredentials = yield* ProviderCredentials;
    const providerModelCatalog = yield* ProviderModelCatalog;
    const modelSlots = yield* ModelSlotController;
    const cloudUsage = yield* MagnitudeCloudUsage;
    const activeSessionStatuses = yield* ActiveSessionStatusesService;
    const displayStreams = yield* DisplayViewStreams;
    const onboarding = yield* Onboarding;
    const mirroredStateChanges = yield* MirroredStateChanges;
    const localHardware = yield* LocalInferenceHardware;
    const localModelPackages = yield* LocalModelPackages;
    const localModels = yield* LocalModels;
    const icnModels = yield* IcnModels;
    const localModelConfigurations = yield* LocalModelConfigurationResolver;
    const modelConfigurationCoordinator = yield* LocalModelConfigurationCoordinator;
    const clientLeases = yield* ClientLeaseManager;
    const displayViewIntrospector = yield* Effect.serviceOption(
      AcnDisplayViewIntrospector
    );
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

    const withSessionContext = <A, E, R>(
      sessionId: string,
      run: (context: {
        cwd: string;
        projectRoot: string;
        scratchpadPath: string;
      }) => Effect.Effect<A, E, R>
    ) =>
      sessionLifecycle
        .getSessionExecutionContext(sessionId)
        .pipe(Effect.flatMap((context) => run(context)));

    const observeDisplayViewStream = <A, E, R>(
      sessionId: string,
      viewId: string,
      stream: Stream.Stream<A, E, R>
    ): Stream.Stream<A, E, R> =>
      Option.match(displayViewIntrospector, {
        onNone: () => stream,
        onSome: (introspector) =>
          Stream.fromEffect(introspector.openStream(sessionId, viewId)).pipe(
            Stream.flatMap(() => stream),
            Stream.ensuring(introspector.closeStream(sessionId, viewId))
          ),
      });

    const recordDisplayViewShape = (
      sessionId: string,
      viewId: string,
      shape: DisplayViewShape
    ) =>
      Option.match(displayViewIntrospector, {
        onNone: () => Effect.void,
        onSome: (introspector) =>
          introspector.setShape(sessionId, viewId, shape),
      });

    const recordDisplayViewResync = (sessionId: string, viewId: string) =>
      Option.match(displayViewIntrospector, {
        onNone: () => Effect.void,
        onSome: (introspector) => introspector.resync(sessionId, viewId),
      });

    const deleteLocalModel = (configurationId: ModelServingConfigurationId) =>
      modelConfigurationCoordinator.exclusive(Effect.gen(function* () {
      const configuration = yield* localModelConfigurations.resolve(configurationId);
      if (Option.isNone(configuration)) {
        return yield* new LocalModelMutationFailed({
          code: "local_model_not_found",
          message: `Local model configuration ${configurationId} was not found`,
          retryable: false,
        });
      }
      const catalogIdentity = Option.map(
        configuration.value.catalogModel,
        ({ modelId, variantId }) => ({ modelId, variantId }),
      );
      const providerModelId = Option.match(catalogIdentity, {
        onNone: () => configuration.value.servingConfiguration.id,
        onSome: localCatalogProviderModelId,
      });
      const slots = (yield* modelSlots.snapshot).state.slots;
      for (const slot of [slots.primary, slots.secondary]) {
        if (slot._tag === "ConfiguredLocal"
          && String(slot.selection.providerModelId) === String(providerModelId)
          && (slot.residency._tag === "Loading"
            || slot.residency._tag === "Stopping")) {
          return yield* new ModelSlotMutationRejected({
            slotId: slot.slotId,
            message: "The local model cannot be deleted while loading or unloading",
          });
        }
        if (slot._tag === "ConfiguredLocal"
          && String(slot.selection.providerModelId) === String(providerModelId)
          && slot.residency._tag === "Ready") {
          yield* modelSlots.stopModel(slot.slotId);
        }
      }
      const installedPackageIds = yield* localModelPackages.installedPackageIds;
      const referencedPackageIds = new Set(
        [...(yield* localModelConfigurations.get).values()]
          .map(({ servingConfiguration }) => servingConfiguration)
          .filter((candidate) => candidate.id !== configurationId)
          .filter((candidate) => servableModelBundlePackageIds(candidate.bundle)
            .every((packageId) => installedPackageIds.has(packageId)))
          .flatMap((candidate) => servableModelBundlePackageIds(candidate.bundle)),
      );
      yield* localModelPackages.removeBundlePackages(
        configuration.value.servingConfiguration.bundle,
        referencedPackageIds,
      );
      if (Option.isSome(catalogIdentity)) {
        const packageState = (yield* localModelPackages.snapshot).state;
        const activePackageIds = new Set(
          servableModelBundlePackageIds(configuration.value.servingConfiguration.bundle),
        );
        for (const entry of packageState.entries) {
          if (entry.localState._tag !== "Installed"
            || entry.catalogAttribution._tag !== "Attributed"
            || entry.catalogAttribution.modelId !== catalogIdentity.value.modelId
            || entry.catalogAttribution.variantId !== catalogIdentity.value.variantId
            || activePackageIds.has(entry.package.id)) {
            continue;
          }
          yield* localModelPackages.removeBundlePackages(
            { _tag: "Standalone", package: entry.package },
            referencedPackageIds,
          );
        }
      }
      return {};
      }));

    return {
      // Connection
      Health: () => lifecycle.state.pipe(
        Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
      ),
      RenewClientLease: ({ clientId }) => clientLeases.renew(clientId),
      ReleaseClientLease: ({ clientId }) => clientLeases.release(clientId),

      // Session lifecycle
      PreloadSession: ({ cwd, projectId, options, draftOwnerId }) =>
        observeRpcDefects(
          "PreloadSession",
          sessionLifecycle.preloadSession(
            cwd,
            Option.getOrUndefined(options),
            Option.getOrNull(draftOwnerId),
            Option.getOrUndefined(projectId)
          )
        ),

      ReleaseSessionPreload: ({ cwd, projectId, sessionId, options, draftOwnerId }) =>
        observeRpcDefects(
          "ReleaseSessionPreload",
          sessionLifecycle
            .releaseSessionPreload(
              cwd,
              sessionId,
              Option.getOrUndefined(options),
              Option.getOrNull(draftOwnerId),
              Option.getOrUndefined(projectId)
            )
            .pipe(Effect.as({}))
        ),

      CreateSession: ({ cwd, projectId, sessionId, initial, options, draftOwnerId }) =>
        observeRpcDefects(
          "CreateSession",
          sessionLifecycle.createSession(
            cwd,
            Option.getOrUndefined(sessionId),
            Option.getOrUndefined(initial),
            Option.getOrUndefined(options),
            Option.getOrNull(draftOwnerId),
            Option.getOrUndefined(projectId)
          )
        ),

      ListSessions: (payload) =>
        observeRpcDefects(
          "ListSessions",
          sessionLifecycle.listSessions({
            ...Option.match(payload.cwd, {
              onNone: () => ({}),
              onSome: (cwd) => ({ cwd }),
            }),
            ...Option.match(payload.projectId, {
              onNone: () => ({}),
              onSome: (projectId) => ({ projectId }),
            }),
            archiveFilter: payload.archiveFilter,
            prioritizePinned: payload.prioritizePinned,
            ...Option.match(payload.query, {
              onNone: () => ({}),
              onSome: (query) => ({ query }),
            }),
            ...Option.match(payload.cursor, {
              onNone: () => ({}),
              onSome: (cursor) => ({ cursor }),
            }),
            limit: payload.limit,
          })
        ),

      ListSessionCwds: () =>
        observeRpcDefects(
          "ListSessionCwds",
          sessionLifecycle.listSessionCwds()
        ),

      StreamActiveSessionStatuses: () => activeSessionStatuses.stream,

      GetSession: ({ sessionId }: { sessionId: string }) =>
        observeRpcDefects(
          "GetSession",
          sessionLifecycle.getSessionInfo(sessionId)
        ),

      DeleteArchivedSession: ({ sessionId }: { sessionId: string }) =>
        observeRpcDefects(
          "DeleteArchivedSession",
          sessionLifecycle.deleteArchivedSession(sessionId).pipe(Effect.as({}))
        ),

      ArchiveSession: ({ sessionId }) =>
        observeRpcDefects("ArchiveSession", sessionLifecycle.archiveSession(sessionId)),

      RestoreSession: ({ sessionId }) =>
        observeRpcDefects("RestoreSession", sessionLifecycle.restoreSession(sessionId)),

      SetSessionPinned: ({ sessionId, pinned }) =>
        observeRpcDefects("SetSessionPinned", sessionLifecycle.setSessionPinned(sessionId, pinned)),

      ListProjects: ({ includeRemoved }) =>
        observeRpcDefects("ListProjects", projects.list(includeRemoved)),

      CreateProject: (payload) =>
        observeRpcDefects("CreateProject", projects.create(payload)),

      EditProject: (payload) =>
        observeRpcDefects("EditProject", projects.edit(payload)),

      RemoveProject: ({ projectId }) =>
        observeRpcDefects("RemoveProject", projects.remove(projectId)),

      RestoreProject: ({ projectId }) =>
        observeRpcDefects("RestoreProject", projects.restore(projectId)),

      RevealProjectSource: ({ projectId }) =>
        observeRpcDefects(
          "RevealProjectSource",
          projects.revealSource(projectId).pipe(Effect.as({})),
        ),

      StreamProjectChanges: () =>
        observeRpcStreamDefects("StreamProjectChanges", projects.changes),

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

      // Agent control
      SendMessage: (payload) =>
        observeRpcDefects(
          "SendMessage",
          sessionCommands
            .sendUserMessage({
              sessionId: payload.sessionId,
              messageId: Option.getOrUndefined(payload.messageId),
              content: payload.content,
              taskMode: payload.taskMode,
              uploads: payload.uploads,
              mentions: payload.mentions,
            })
            .pipe(Effect.as({}))
        ),

      StartGoal: (payload) =>
        observeRpcDefects(
          "StartGoal",
          sessionCommands
            .startGoal({
              sessionId: payload.sessionId,
              objective: payload.objective,
            })
            .pipe(Effect.as({}))
        ),

      Interrupt: ({ sessionId, target }) =>
        observeRpcDefects(
          "Interrupt",
          sessionCommands.interrupt(sessionId, target).pipe(Effect.as({}))
        ),

      UploadAttachment: ({ sessionId, filename, data }) =>
        observeRpcDefects(
          "UploadAttachment",
          withSessionContext(sessionId, (context) =>
            uploadAttachment(context.scratchpadPath, filename, data)
          )
        ),

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

      GetProviderModelCatalog: () =>
        observeRpcDefects("GetProviderModelCatalog", providerModelCatalog.snapshot),

      RefreshModelCatalog: ({ providerId }) =>
        observeRpcDefects(
          "RefreshModelCatalog",
          providerModelCatalog.refresh(providerId).pipe(Effect.as({})),
        ),

      GetModelSlots: () =>
        observeRpcDefects("GetModelSlots", modelSlots.snapshot),

      AssignSlot: ({ slotId, selection }) =>
        observeRpcDefects(
          "AssignSlot",
          modelSlots.updateModelSlot(slotId, Option.some(selection)).pipe(Effect.as({})),
        ),

      ClearSlot: ({ slotId }) =>
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

      GetLocalInferenceHardware: () =>
        observeRpcDefects("GetLocalInferenceHardware", localHardware.snapshot),

      GetLocalModels: () =>
        observeRpcDefects("GetLocalModels", localModels.snapshot),

      WatchMirroredStates: () =>
        observeRpcStreamDefects(
          "WatchMirroredStates",
          mirroredStateChanges.stream,
        ),

      ReconcileCatalogModel: ({ modelId, variantId }) =>
        observeRpcDefects(
          "ReconcileCatalogModel",
          icnModels.reconcileCatalogModel(modelId, variantId).pipe(
            Effect.mapError((error) => new LocalModelMutationFailed({
              code: "catalog_model_reconciliation_failed",
              message: String(error),
              retryable: true,
            })),
            Effect.map((admission) => {
              const providerModelId = localCatalogProviderModelId({ modelId, variantId });
              return admission._tag === "Current"
                ? { _tag: "Current" as const, providerModelId }
                : {
                    _tag: "DownloadAdmitted" as const,
                    providerModelId,
                    downloadId: ModelDownloadIdSchema.make(admission.downloadId),
                  };
            }),
            Effect.tap(() => localModels.refresh),
          ),
        ),

      CancelModelDownload: ({ downloadId }) =>
        observeRpcDefects(
          "CancelModelDownload",
          localModelPackages.cancelDownload(downloadId).pipe(
            Effect.zipRight(localModels.refresh),
            Effect.as({}),
          ),
        ),

      DismissModelDownloadFailure: ({ downloadId }) =>
        observeRpcDefects(
          "DismissModelDownloadFailure",
          localModelPackages.acknowledgeFailure(downloadId).pipe(
            Effect.zipRight(localModels.refresh),
            Effect.as({}),
          ),
        ),

      DeleteLocalModel: ({ configurationId }) =>
        observeRpcDefects(
          "DeleteLocalModel",
          deleteLocalModel(configurationId).pipe(
            Effect.tap(() => localModels.refresh),
          ),
        ),

      LoadModel: ({ slotId }) =>
        observeRpcDefects(
          "LoadModel",
          modelSlots.loadModel(slotId).pipe(Effect.as({})),
        ),

      PreviewModelLoad: ({ slotId }) =>
        observeRpcDefects(
          "PreviewModelLoad",
          modelSlots.previewModelLoad(slotId),
        ),

      StopModel: ({ slotId }) =>
        observeRpcDefects(
          "StopModel",
          modelSlots.stopModel(slotId).pipe(Effect.as({})),
        ),

      GetOnboardingState: () =>
        observeRpcDefects(
          "GetOnboardingState",
          onboarding.snapshot,
        ),

      CompleteOnboarding: () =>
        observeRpcDefects(
          "CompleteOnboarding",
          onboarding.complete.pipe(Effect.as({})),
        ),

      // Server-side operations
      ListFiles: ({ cwd, glob, limit }) =>
        observeRpcDefects("ListFiles", listFiles(cwd, glob, limit)),

      ReadFile: ({ cwd, path, format, offset }) =>
        observeRpcDefects("ReadFile", readFileOp(cwd, path, format, offset)),

      CheckFileExists: ({ cwd, path }) =>
        observeRpcDefects("CheckFileExists", checkFileExists(cwd, path)),

      WatchFile: ({ cwd, path }) =>
        observeRpcStreamDefects(
          "WatchFile",
          watchFile(cwd, path)
        ),

      ResolvePath: ({ cwd, path, checkExists }) =>
        observeRpcDefects("ResolvePath", resolvePath(cwd, path, checkExists)),

      SearchMentions: ({ cwd, query, limit, visibleLimit, includeRecent }) =>
        observeRpcDefects(
          "SearchMentions",
          searchMentions(cwd, query, limit, visibleLimit, includeRecent)
        ),

      SearchDirectories: ({ query, limit, includeRecent }) =>
        observeRpcDefects(
          "SearchDirectories",
          Effect.gen(function* () {
            const cwdSummaries = includeRecent
              ? yield* sessionLifecycle.listSessionCwds()
              : [];
            const recentDirectories = cwdSummaries.map((summary) => ({
              path: summary.cwd,
              lastActivity: summary.updatedAt,
            }));
            return yield* searchDirectories(
              query,
              recentDirectories,
              limit,
              includeRecent
            );
          })
        ),

      GetGitRecentFiles: ({ cwd, limit }) =>
        observeRpcDefects("GetGitRecentFiles", getGitRecentFiles(cwd, limit)),

      ListSkills: ({ cwd }) => observeRpcDefects("ListSkills", listSkills(cwd)),

      GetSkill: ({ cwd, name }) =>
        observeRpcDefects("GetSkill", getSkill(cwd, name)),

      RunBash: ({ sessionId, command, stdin }) =>
        observeRpcDefects(
          "RunBash",
          sessionCommands.getRuntimeExecutionContext(sessionId).pipe(
            Effect.flatMap((context) =>
              runBash(context, command, stdin).pipe(
                Effect.flatMap((result) => {
                  const stdout = normalizeBashOutput(result.stdout)
                  const stderr = normalizeBashOutput(result.stderr)
                  const event: Extract<AppEvent, { type: "user_bash_command" }> = {
                    type: "user_bash_command",
                    commandId: UserBashCommandId(createId()),
                    forkId: null,
                    timestamp: Date.now(),
                    command,
                    cwd: context.cwd,
                    exitCode: result.exitCode,
                    stdout,
                    stderr,
                  }
                  return sessionCommands.sendUserEvent(sessionId, event).pipe(
                    Effect.as({ ...result, stdout, stderr }),
                  )
                })
              )
            )
          )
        ),

      // Streams
      StreamDisplayView: ({ sessionId, viewId, shape }) =>
        observeRpcStreamDefects(
          "StreamDisplayView",
          observeDisplayViewStream(
            sessionId,
            viewId,
            displayStreams.getDisplayViewStream(sessionId, viewId, shape)
          )
        ),

      ResyncDisplayView: ({ sessionId, viewId }) =>
        observeRpcDefects(
          "ResyncDisplayView",
          displayStreams.requestDisplayViewSnapshot(sessionId, viewId).pipe(
            Effect.tap(() => recordDisplayViewResync(sessionId, viewId)),
          )
        ),

      SetDisplayViewShape: ({ sessionId, viewId, shape }) =>
        observeRpcDefects(
          "SetDisplayViewShape",
          displayStreams
            .setDisplayViewShape(sessionId, viewId, shape)
            .pipe(
              Effect.tap(() => recordDisplayViewShape(sessionId, viewId, shape))
            )
        ),

    };
  })
);
