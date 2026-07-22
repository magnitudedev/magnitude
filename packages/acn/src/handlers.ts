import {
  MagnitudeRpcs,
  SessionOperationFailed,
  ModelSlotMutationRejected,
  type DisplayViewShape,
  type SessionError,
} from "@magnitudedev/protocol";
import { Cause, Chunk, Effect, Option, Stream } from "effect";
import { SessionCommands } from "./session-commands";
import { SessionLifecycle } from "./session-lifecycle";
import { ProviderCredentials } from "./provider-credentials";
import { ProviderModelCatalog } from "./provider-model-catalog";
import { ModelSlotCoordinator } from "./model-slot-coordinator";
import { MagnitudeCloudUsage } from "./magnitude-cloud-usage";
import { ActiveSessionStatusesService } from "./active-session-statuses";
import { DisplayViewStreams } from "./display-view-streams";
import { ACN_VERSION } from "./version";
import { makeHealthResponse } from "./identity";
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
import type { AppEvent } from "@magnitudedev/agent";
import { Onboarding } from "./onboarding";
import { MirroredStateChanges } from "./mirrored-state";
import { LocalModelInventory } from "./local-model-inventory";
import { LocalInferenceHardware } from "./local-inference-hardware";

export const HandlersLive = MagnitudeRpcs.toLayer(
  Effect.gen(function* () {
    const sessionCommands = yield* SessionCommands;
    const sessionLifecycle = yield* SessionLifecycle;
    const providerCredentials = yield* ProviderCredentials;
    const providerModelCatalog = yield* ProviderModelCatalog;
    const modelSlots = yield* ModelSlotCoordinator;
    const cloudUsage = yield* MagnitudeCloudUsage;
    const activeSessionStatuses = yield* ActiveSessionStatusesService;
    const displayStreams = yield* DisplayViewStreams;
    const onboarding = yield* Onboarding;
    const mirroredStateChanges = yield* MirroredStateChanges;
    const localModels = yield* LocalModelInventory;
    const localHardware = yield* LocalInferenceHardware;
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

    return {
      // Connection
      Health: () => Effect.succeed(makeHealthResponse(ACN_VERSION)),

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

      ReleaseSessionPreload: ({ cwd, options, draftOwnerId }) =>
        observeRpcDefects(
          "ReleaseSessionPreload",
          sessionLifecycle
            .releaseSessionPreload(
              cwd,
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
        observeRpcDefects(
          "ListSessions",
          sessionLifecycle.listSessions({
            ...Option.match(payload.cwd, {
              onNone: () => ({}),
              onSome: (cwd) => ({ cwd }),
            }),
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

      DeleteSession: ({ sessionId }: { sessionId: string }) =>
        observeRpcDefects(
          "DeleteSession",
          sessionLifecycle.deleteSession(sessionId).pipe(Effect.as({}))
        ),

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
              imageAttachments: payload.imageAttachments,
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

      UpdateModelSlot: ({ slotId, selection }) =>
        observeRpcDefects(
          "UpdateModelSlot",
          modelSlots.updateModelSlot(slotId, selection).pipe(Effect.as({})),
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

      GetLocalModelInventory: () =>
        observeRpcDefects("GetLocalModelInventory", localModels.snapshot),

      WatchMirroredStates: () =>
        observeRpcStreamDefects(
          "WatchMirroredStates",
          mirroredStateChanges.stream,
        ),

      DownloadLocalModel: ({ localModelId }) =>
        observeRpcDefects(
          "DownloadLocalModel",
          localModels.download(localModelId).pipe(Effect.as({})),
        ),

      LoadModelSlot: ({ slotId, selection }) =>
        observeRpcDefects(
          "LoadModelSlot",
          modelSlots.loadModelSlot(slotId, selection).pipe(Effect.as({})),
        ),

      ReloadModelSlot: ({ slotId }) =>
        observeRpcDefects(
          "ReloadModelSlot",
          modelSlots.reloadModelSlot(slotId).pipe(Effect.as({})),
        ),

      DeleteLocalModel: ({ localModelId }) =>
        observeRpcDefects(
          "DeleteLocalModel",
          modelSlots.deleteLocalModel(localModelId).pipe(
            Effect.as({}),
          ),
        ),

      UnloadModelSlot: ({ slotId }) =>
        observeRpcDefects(
          "UnloadModelSlot",
          modelSlots.unloadModelSlot(slotId).pipe(Effect.as({})),
        ),

      // Generic onboarding completion, independent of local inference
      GetOnboardingState: () =>
        observeRpcDefects(
          "GetOnboardingState",
          onboarding.state,
        ),

      CompleteOnboardingFlow: ({ flowId }) =>
        observeRpcDefects(
          "CompleteOnboardingFlow",
          onboarding.complete(flowId).pipe(Effect.as({})),
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
                Effect.tap((result) => {
                  const event: Extract<AppEvent, { type: "user_bash_command" }> = {
                    type: "user_bash_command",
                    forkId: null,
                    timestamp: Date.now(),
                    command,
                    cwd: context.cwd,
                    exitCode: result.exitCode,
                    stdout:
                      result.stdout.length > 50_000
                        ? result.stdout.slice(0, 50_000) + "\n[truncated]"
                        : result.stdout,
                    stderr:
                      result.stderr.length > 50_000
                        ? result.stderr.slice(0, 50_000) + "\n[truncated]"
                        : result.stderr,
                  }
                  return sessionCommands.sendUserEvent(sessionId, event)
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
