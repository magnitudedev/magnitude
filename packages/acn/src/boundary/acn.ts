import {
  AcnBoundary,
  AcnRpc,
  Configuration,
  Models,
  SessionOperationFailed,
  type DisplayViewShape,
  type SessionError,
} from "@magnitudedev/acn-protocol";
import { Cause, Chunk, Effect, Option, Stream } from "effect";
import { SessionCommands } from "../session-commands";
import { SessionLifecycle } from "../session-lifecycle";
import { ProviderCredentials } from "../provider-credentials";
import { ProviderModelCatalog } from "../provider-model-catalog";
import { ModelSlotController } from "../model-slot-controller";
import { MagnitudeCloudUsage } from "../magnitude-cloud-usage";
import { ActiveSessionStatusesService } from "../active-session-statuses";
import { DisplayViewStreams, displayViewId } from "../display-view-streams";
import { ACN_VERSION } from "../version";
import { makeHealthResponse } from "../identity";
import { AcnServiceLifecycle } from "../service-lifecycle";
import { AcnDisplayViewIntrospector } from "../introspection";
import { uploadAttachment } from "../attachment-upload";
import { getSkill, listSkills, runBash } from "../skill-shell-ops";
import { UserBashCommandId, type AppEvent } from "@magnitudedev/agent";
import { createId } from "@magnitudedev/generate-id";
import { Onboarding } from "../onboarding";
import { AcnChanges } from "../changes";
import { ModelCatalog } from "../model-catalog";
import { ModelCommands } from "../model-commands";
import { LocalInferenceHardware } from "../local-inference-hardware";
import { FileMentionSearcher } from "../file-mention-searcher";
import { FileSystemManager } from "../file-system-manager";
import { GitInspector } from "../git-inspector";
import { ProjectFileManager } from "../project-file-manager";
import { ProjectInspector } from "../project-inspector";
import { ProjectManager } from "../project-manager";
import { ProjectStore } from "../project-store";
import { SessionInspector } from "../session-inspector";

const MAX_BASH_OUTPUT_LENGTH = 50_000;

const normalizeBashOutput = (output: string): string =>
  output.length > MAX_BASH_OUTPUT_LENGTH
    ? `${output.slice(0, MAX_BASH_OUTPUT_LENGTH)}\n[truncated]`
    : output;

/** Exhaustive server implementation of the composed ACN boundary. */
export const AcnBoundaryLive = AcnRpc.toLayer(AcnBoundary,
  Effect.gen(function* () {
    const lifecycle = yield* AcnServiceLifecycle;
    const sessionCommands = yield* SessionCommands;
    const sessionLifecycle = yield* SessionLifecycle;
    const sessionInspector = yield* SessionInspector;
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
    const activeSessionStatuses = yield* ActiveSessionStatusesService;
    const displayStreams = yield* DisplayViewStreams;
    const onboarding = yield* Onboarding;
    const changes = yield* AcnChanges;
    const modelCatalog = yield* ModelCatalog;
    const modelCommands = yield* ModelCommands;
    const localInferenceHardware = yield* LocalInferenceHardware;
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
      shape: DisplayViewShape,
      stream: Stream.Stream<A, E, R>
    ): Stream.Stream<A, E, R> =>
      Option.match(displayViewIntrospector, {
        onNone: () => stream,
        onSome: (introspector) => {
          const viewId = displayViewId(shape);
          return Stream.fromEffect(introspector.openStream(sessionId, viewId, shape)).pipe(
            Stream.flatMap(() => stream),
            Stream.ensuring(introspector.closeStream(sessionId, viewId))
          );
        },
      });

    return {
      // Connection
      Health: () => lifecycle.state.pipe(
        Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
      ),
      // Session lifecycle
      PreloadSession: ({ cwd, options, draftOwnerId }) =>
        observeRpcDefects(
          "PreloadSession",
          sessionLifecycle.preloadSession(
            cwd,
            Option.getOrUndefined(options),
            Option.getOrNull(draftOwnerId)
          )
        ),

      ReleaseSessionPreload: ({ cwd, sessionId, options, draftOwnerId }) =>
        observeRpcDefects(
          "ReleaseSessionPreload",
          sessionLifecycle
            .releaseSessionPreload(
              cwd,
              sessionId,
              Option.getOrUndefined(options),
              Option.getOrNull(draftOwnerId)
            )
            .pipe(Effect.as({}))
        ),

      CreateSession: ({ cwd, sessionId, initial, options, draftOwnerId }) =>
        observeRpcDefects(
          "CreateSession",
          sessionLifecycle.createSession(
            cwd,
            Option.getOrUndefined(sessionId),
            Option.getOrUndefined(initial),
            Option.getOrUndefined(options),
            Option.getOrNull(draftOwnerId)
          )
        ),

      ListSessions: (payload) =>
        observeRpcDefects("ListSessions", sessionInspector.page(payload)),

      ListRecentSessionDirectories: (payload) =>
        observeRpcDefects(
          "ListRecentSessionDirectories",
          sessionInspector.recentDirectories(payload)
        ),

      StreamActiveSessionStatuses: () => activeSessionStatuses.stream,

      GetSession: ({ sessionId }: { sessionId: string }) =>
        observeRpcDefects("GetSession", sessionInspector.get(sessionId)),

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

      RefreshModelCatalog: ({ providerId }) =>
        observeRpcDefects(
          "RefreshModelCatalog",
          modelCatalog.refresh(providerId).pipe(Effect.as({})),
        ),

      GetModelSlots: () =>
        observeRpcDefects("GetModelSlots", modelSlots.state),

      GetModelCatalog: () =>
        observeRpcDefects("GetModelCatalog", modelCatalog.state),

      GetLocalInferenceEnvironment: () =>
        observeRpcDefects("GetLocalInferenceEnvironment", localInferenceHardware.state),

      SyncLocalModel: ({ modelId }) =>
        observeRpcDefects("SyncLocalModel", modelCommands.sync(modelId)),

      CancelLocalModelSync: ({ modelId }) =>
        observeRpcDefects("CancelLocalModelSync", modelCommands.cancelSync(modelId)),

      AcknowledgeLocalModelSyncFailure: ({ modelId }) =>
        observeRpcDefects("AcknowledgeLocalModelSyncFailure", modelCommands.acknowledgeSyncFailure(modelId)),

      RemoveLocalModel: ({ modelId }) =>
        observeRpcDefects("RemoveLocalModel", modelCommands.remove(modelId)),

      LoadModelSlot: ({ slotId }) =>
        observeRpcDefects("LoadModelSlot", modelCommands.loadSlot(slotId)),

      PreviewModelSlotLoad: ({ slotId }) =>
        observeRpcDefects("PreviewModelSlotLoad", modelCommands.previewSlotLoad(slotId)),

      StopModelSlot: ({ slotId }) =>
        observeRpcDefects("StopModelSlot", modelCommands.stopSlot(slotId)),

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

      GetOnboardingState: () =>
        observeRpcDefects(
          "GetOnboardingState",
          onboarding.state,
        ),

      CompleteOnboarding: () =>
        observeRpcDefects(
          "CompleteOnboarding",
          onboarding.complete.pipe(Effect.as({})),
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

      SearchDirectories: ({ query, limit, includeRecent }) =>
        observeRpcDefects(
          "SearchDirectories",
          Effect.gen(function* () {
            const recents = includeRecent
              ? (yield* sessionInspector.recentDirectories({
                  cursor: Option.none(),
                  limit: 20,
                }).pipe(
                  Effect.map((page) => page.items.map((item) => ({
                    path: item.cwd,
                    lastActivity: item.lastActiveAt,
                  }))),
                  // Recents are advisory: directory suggestions degrade to
                  // plain path completion when session inspection fails.
                  Effect.orElseSucceed(() => []),
                ))
              : [];
            return yield* fileMentionSearcher.searchDirectories({
              query,
              limit,
              includeRecent,
              recentDirectories: recents,
            });
          })
        ),

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
      StreamChanges: () => observeRpcStreamDefects("StreamChanges", changes.stream),

      StreamDisplayView: ({ sessionId, shape }) =>
        observeRpcStreamDefects(
          "StreamDisplayView",
          observeDisplayViewStream(sessionId, shape, displayStreams.stream(sessionId, shape))
        ),

    };
  })
);
