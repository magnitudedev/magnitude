import {
  MagnitudeRpcs,
  LocalModelMutationFailed,
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
import { UserBashCommandId, type AppEvent } from "@magnitudedev/agent";
import { createId } from "@magnitudedev/generate-id";
import { Onboarding } from "./onboarding";
import { MirroredStateChanges } from "./mirrored-state";
import { LocalInferenceHardware } from "./local-inference-hardware";
import { LocalModelPackages } from "./local-model-packages";
import { LocalModelRecommendations } from "./local-model-recommendations";
import { LocalModels } from "./local-models";
import { LocalProviderOfferings } from "./local-provider-offerings";
import { modelOfferingTargetPackageIds } from "@magnitudedev/protocol";

const MAX_BASH_OUTPUT_LENGTH = 50_000;

const normalizeBashOutput = (output: string): string =>
  output.length > MAX_BASH_OUTPUT_LENGTH
    ? `${output.slice(0, MAX_BASH_OUTPUT_LENGTH)}\n[truncated]`
    : output;

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
    const localHardware = yield* LocalInferenceHardware;
    const localModelPackages = yield* LocalModelPackages;
    const localModelRecommendations = yield* LocalModelRecommendations;
    const localModels = yield* LocalModels;
    const localProviderOfferings = yield* LocalProviderOfferings;
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

      DownloadRecommendedModel: ({ recommendationId }) =>
        observeRpcDefects(
          "DownloadRecommendedModel",
          Effect.gen(function* () {
            const recommendation = yield* localModelRecommendations.get(recommendationId);
            if (!recommendation) {
              return yield* new LocalModelMutationFailed({
                code: "model_recommendation_not_found",
                message: `Model recommendation ${recommendationId} is no longer available`,
                retryable: false,
              });
            }
            yield* localProviderOfferings.save(
              recommendation.modelId,
              recommendation.configuration,
              { _tag: "Recommendation", recommendationId },
            );
            yield* localModelPackages.downloadTarget(recommendation.configuration.target);
            return {};
          }),
        ),

      RetryModelDownload: ({ modelId }) =>
        observeRpcDefects(
          "RetryModelDownload",
          Effect.gen(function* () {
            const target = yield* localModels.target(modelId);
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${modelId} was not found`,
                retryable: false,
              });
            }
            yield* localModelPackages.downloadTarget(target);
            return {};
          }),
        ),

      CancelModelDownload: ({ modelId }) =>
        observeRpcDefects(
          "CancelModelDownload",
          Effect.gen(function* () {
            const target = yield* localModels.target(modelId);
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${modelId} was not found`,
                retryable: false,
              });
            }
            yield* localModelPackages.cancelTargetDownload(target);
            return {};
          }),
        ),

      DismissModelDownloadFailure: ({ modelId }) =>
        observeRpcDefects(
          "DismissModelDownloadFailure",
          Effect.gen(function* () {
            const target = yield* localModels.target(modelId);
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${modelId} was not found`,
                retryable: false,
              });
            }
            yield* localModelPackages.dismissTargetFailure(target);
            return {};
          }),
        ),

      DeleteLocalModel: ({ modelId }) =>
        observeRpcDefects(
          "DeleteLocalModel",
          Effect.gen(function* () {
            const target = yield* localModels.target(modelId);
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${modelId} was not found`,
                retryable: false,
              });
            }
            const targetOfferings = (yield* localProviderOfferings.list)
              .filter((offering) => offering.modelId === modelId);
            const targetProviderModelIds = new Set(
              targetOfferings.map((offering) => offering.providerModelId),
            );
            const slots = (yield* modelSlots.snapshot).state.slots;
            for (const slot of [slots.primary, slots.secondary]) {
              if ((slot._tag === "LoadingLocalModel" || slot._tag === "UnloadingLocalModel")
                && targetProviderModelIds.has(slot.selection.providerModelId)) {
                return yield* new ModelSlotMutationRejected({
                  slotId: slot.slotId,
                  message: "The local model cannot be deleted while loading or unloading",
                });
              }
              if (slot._tag === "Ready"
                && targetProviderModelIds.has(slot.selection.providerModelId)) {
                yield* modelSlots.unloadModel(slot.slotId);
              }
            }
            const retainedOfferings = (yield* localProviderOfferings.list)
              .filter((offering) => offering.modelId !== modelId);
            const retainedPackageIds = new Set(retainedOfferings.flatMap((offering) =>
              modelOfferingTargetPackageIds(offering.configuration.target)));
            yield* localModelPackages.removeTargetPackages(target, retainedPackageIds);
            return {};
          }),
        ),

      LoadModel: ({ slotId }) =>
        observeRpcDefects(
          "LoadModel",
          modelSlots.loadModel(slotId).pipe(Effect.as({})),
        ),

      UnloadModel: ({ slotId }) =>
        observeRpcDefects(
          "UnloadModel",
          modelSlots.unloadModel(slotId).pipe(Effect.as({})),
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
