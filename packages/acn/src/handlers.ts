import {
  MagnitudeRpcs,
  LocalModelMutationFailed,
  SessionOperationFailed,
  ModelSlotMutationRejected,
  type DisplayViewShape,
  type ModelOfferingTargetId,
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
import { LocalProviderOfferings } from "./local-provider-offerings";
import { modelOfferingTargetPackageIds } from "@magnitudedev/acn-protocol";

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
    const modelSlots = yield* ModelSlotController;
    const cloudUsage = yield* MagnitudeCloudUsage;
    const activeSessionStatuses = yield* ActiveSessionStatusesService;
    const displayStreams = yield* DisplayViewStreams;
    const onboarding = yield* Onboarding;
    const mirroredStateChanges = yield* MirroredStateChanges;
    const localHardware = yield* LocalInferenceHardware;
    const localModelPackages = yield* LocalModelPackages;
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

    const deleteLocalModel = (targetId: ModelOfferingTargetId) => Effect.gen(function* () {
      const target = yield* localModels.target(targetId);
      if (!target) {
        return yield* new LocalModelMutationFailed({
          code: "local_model_not_found",
          message: `Local model ${targetId} was not found`,
          retryable: false,
        });
      }
      const targetOfferings = (yield* localProviderOfferings.list)
        .filter((offering) => offering.targetId === targetId);
      const targetProviderModelIds = new Set(
        targetOfferings.map((offering) => offering.providerModelId),
      );
      const slots = (yield* modelSlots.snapshot).state.slots;
      for (const slot of [slots.primary, slots.secondary]) {
        if (slot._tag === "ConfiguredLocal"
          && targetProviderModelIds.has(slot.selection.providerModelId)
          && Option.isSome(slot.instance)
          && (slot.instance.value.lifecycle._tag === "Loading"
            || slot.instance.value.lifecycle._tag === "Stopping")) {
          return yield* new ModelSlotMutationRejected({
            slotId: slot.slotId,
            message: "The local model cannot be deleted while loading or unloading",
          });
        }
        if (slot._tag === "ConfiguredLocal"
          && targetProviderModelIds.has(slot.selection.providerModelId)
          && Option.isSome(slot.instance)
          && slot.instance.value.lifecycle._tag === "Ready") {
          yield* modelSlots.stopModel(slot.instance.value.id);
        }
      }
      const retainedOfferings = (yield* localProviderOfferings.list)
        .filter((offering) => offering.targetId !== targetId);
      const retainedPackageIds = new Set(retainedOfferings.flatMap((offering) =>
        modelOfferingTargetPackageIds(offering.configuration.target)));
      yield* localModelPackages.removeTargetPackages(target, retainedPackageIds);
      return {};
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

      DownloadModel: ({ targetId }) =>
        observeRpcDefects(
          "DownloadModel",
          Effect.gen(function* () {
            const target = yield* localModels.target(targetId)
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${targetId} was not found`,
                retryable: false,
              })
            }
            yield* localModelPackages.acquireTarget(target)
            return {}
          }),
        ),

      CancelModelDownload: ({ targetId }) =>
        observeRpcDefects(
          "CancelModelDownload",
          Effect.gen(function* () {
            const target = yield* localModels.target(targetId)
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${targetId} was not found`,
                retryable: false,
              })
            }
            yield* localModelPackages.cancelTargetDownload(target)
            return {}
          }),
        ),

      DismissModelDownloadFailure: ({ targetId }) =>
        observeRpcDefects(
          "DismissModelDownloadFailure",
          Effect.gen(function* () {
            const target = yield* localModels.target(targetId)
            if (!target) {
              return yield* new LocalModelMutationFailed({
                code: "local_model_not_found",
                message: `Local model ${targetId} was not found`,
                retryable: false,
              })
            }
            yield* localModelPackages.dismissTargetFailure(target)
            return {}
          }),
        ),

      DeleteLocalModel: ({ targetId }) =>
        observeRpcDefects(
          "DeleteLocalModel",
          deleteLocalModel(targetId),
        ),

      LoadModel: ({ slotId }) =>
        observeRpcDefects(
          "LoadModel",
          modelSlots.loadModel(slotId),
        ),

      PreviewModelLoad: ({ slotId }) =>
        observeRpcDefects(
          "PreviewModelLoad",
          modelSlots.previewModelLoad(slotId),
        ),

      StopModel: ({ instanceId }) =>
        observeRpcDefects(
          "StopModel",
          modelSlots.stopModel(instanceId).pipe(Effect.as({})),
        ),

      GetOnboardingState: () =>
        observeRpcDefects(
          "GetOnboardingState",
          onboarding.snapshot,
        ),

      UpdateOnboardingState: ({ completed }) =>
        observeRpcDefects(
          "UpdateOnboardingState",
          onboarding.update(completed).pipe(Effect.as({})),
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
