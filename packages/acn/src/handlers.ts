import {
  MagnitudeRpcs,
  SessionOperationFailed,
  STREAM_HEARTBEAT_INTERVAL_MS,
  type DisplayViewShape,
  type SessionError,
  type StreamHeartbeat,
} from "@magnitudedev/protocol";
import { Cause, Chunk, Effect, Option, Schedule, Stream } from "effect";
import { SessionCommands } from "./session-commands";
import { SessionLifecycle } from "./session-lifecycle";
import { Account } from "./account";
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
import { LocalInference } from "./local-inference";
import { LocalModelProviderSource } from "./local-inference/provider-source";
import { Onboarding } from "./onboarding";

export const HandlersLive = MagnitudeRpcs.toLayer(
  Effect.gen(function* () {
    const sessionCommands = yield* SessionCommands;
    const sessionLifecycle = yield* SessionLifecycle;
    const account = yield* Account;
    const activeSessionStatuses = yield* ActiveSessionStatusesService;
    const displayStreams = yield* DisplayViewStreams;
    const localInference = yield* LocalInference;
    const localModelSource = yield* LocalModelProviderSource;
    const onboarding = yield* Onboarding;
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
    // Liveness heartbeats: every long-lived stream emits one at a fixed
    // cadence so clients can distinguish "daemon dead" from "no events".
    // Consumers filter them when they need only domain events. Halts when the
    // source stream halts; source errors propagate untouched.
    const heartbeatEvent: StreamHeartbeat = { _tag: "heartbeat" };
    const withHeartbeat = <A, E, R>(
      stream: Stream.Stream<A, E, R>
    ): Stream.Stream<A | StreamHeartbeat, E, R> =>
      Stream.merge(
        stream,
        Stream.repeatEffectWithSchedule(
          Effect.succeed(heartbeatEvent),
          Schedule.spaced(`${STREAM_HEARTBEAT_INTERVAL_MS} millis`)
        ),
        { haltStrategy: "left" }
      );

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

    const recordDisplayViewClose = (sessionId: string, viewId: string) =>
      Option.match(displayViewIntrospector, {
        onNone: () => Effect.void,
        onSome: (introspector) => introspector.closeView(sessionId, viewId),
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

      StreamActiveSessionStatuses: () =>
        withHeartbeat(activeSessionStatuses.stream),

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
          account.updateProviderAuth(providerId, auth).pipe(Effect.as({}))
        ),

      GetProviderAuth: ({ providerId }) =>
        observeRpcDefects(
          "GetProviderAuth",
          account
            .getProviderAuth(providerId)
            .pipe(Effect.map((auth) => ({ auth: Option.fromNullable(auth) })))
        ),

      ListProviderAuth: () =>
        observeRpcDefects(
          "ListProviderAuth",
          account.listProviderAuth.pipe(Effect.map((auths) => ({ auths })))
        ),

      ListPublicSlotProfiles: () =>
        observeRpcDefects(
          "ListPublicSlotProfiles",
          account.listPublicSlotProfiles
        ),

      GetModelCatalog: () =>
        observeRpcDefects("GetModelCatalog", account.modelCatalog),

      WatchModelCatalog: () =>
        observeRpcStreamDefects("WatchModelCatalog", withHeartbeat(account.watchModelCatalog)),

      RefreshModelCatalog: ({ providerId }) =>
        observeRpcDefects(
          "RefreshModelCatalog",
          account.refreshModelCatalog(providerId).pipe(Effect.as({})),
        ),

      GetModelSlots: () =>
        observeRpcDefects("GetModelSlots", account.modelSlots),

      WatchModelSlots: () =>
        observeRpcStreamDefects("WatchModelSlots", withHeartbeat(account.watchModelSlots)),

      UpdateModelSlots: ({ slots }) =>
        observeRpcDefects(
          "UpdateModelSlots",
          account.updateModelSlots(slots ?? {}).pipe(
            Effect.mapError((error) => new SessionOperationFailed({
              operation: "operation" in error ? String(error.operation) : "select local model",
              reason: "reason" in error ? String(error.reason) : String(error),
            })),
            Effect.as({}),
          ),
        ),

      GetCloudUsage: (payload) =>
        observeRpcDefects(
          "GetCloudUsage",
          account.getCloudUsage({
            ...(payload.period !== undefined ? { period: payload.period } : {}),
            ...(payload.days !== undefined ? { days: payload.days } : {}),
            ...(payload.tz !== undefined ? { tz: payload.tz } : {}),
          })
        ),

      // Local inference capability
      GetLocalInferenceState: () =>
        observeRpcDefects(
          "GetLocalInferenceState",
          localInference.state,
        ),

      WatchLocalInferenceState: () =>
        observeRpcStreamDefects(
          "WatchLocalInferenceState",
          withHeartbeat(localInference.watchState),
        ),

      ConfigureLocalInferenceUsage: (selection) =>
        observeRpcDefects(
          "ConfigureLocalInferenceUsage",
          localInference.configureUsage(selection).pipe(Effect.as({})),
        ),

      InstallManagedLlamaCpp: () =>
        observeRpcDefects(
          "InstallManagedLlamaCpp",
          localInference.installLlamaCpp.pipe(Effect.map((operationId) => ({ operationId }))),
        ),

      RefreshLocalInferenceInstallations: () =>
        observeRpcDefects(
          "RefreshLocalInferenceInstallations",
          localInference.refreshInstallations.pipe(Effect.as({})),
        ),

      DownloadLocalModel: ({ configurationId }) =>
        observeRpcDefects(
          "DownloadLocalModel",
          localInference.downloadModel(configurationId).pipe(Effect.as({})),
        ),

      ActivateLocalModel: ({ selectionId }) =>
        observeRpcDefects(
          "ActivateLocalModel",
          localInference.activateModel(selectionId).pipe(Effect.as({})),
        ),

      DeleteLocalModel: ({ selectionId }) =>
        observeRpcDefects(
          "DeleteLocalModel",
          localInference.deleteModel(selectionId).pipe(Effect.as({})),
        ),

      RestartLocalInference: () =>
        observeRpcDefects(
          "RestartLocalInference",
          localInference.restart.pipe(Effect.as({})),
        ),

      DisableLocalInference: () =>
        observeRpcDefects(
          "DisableLocalInference",
          localInference.disable.pipe(Effect.as({})),
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
          withHeartbeat(watchFile(cwd, path))
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
          withHeartbeat(
            observeDisplayViewStream(
              sessionId,
              viewId,
              displayStreams.getDisplayViewStream(sessionId, viewId, shape)
            )
          )
        ),

      ResyncDisplayView: ({ sessionId, viewId }) =>
        observeRpcDefects(
          "ResyncDisplayView",
          displayStreams.requestDisplayViewSnapshot(sessionId, viewId).pipe(
            Effect.tap(() => recordDisplayViewResync(sessionId, viewId)),
            Effect.map((): "ok" => "ok")
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

      CloseDisplayView: ({ sessionId, viewId }) =>
        observeRpcDefects(
          "CloseDisplayView",
          displayStreams
            .closeDisplayView(sessionId, viewId)
            .pipe(Effect.tap(() => recordDisplayViewClose(sessionId, viewId)))
        ),

      StreamEvents: () => Stream.empty,
    };
  })
);
