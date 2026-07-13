import { Effect, Schema } from "effect";
import { DisplayViewSnapshot, type StreamEvent } from "@magnitudedev/sdk";
import { compilePatchMap, applyDecodedPatch } from "@magnitudedev/utils/patch";
import type { DisplaySyncSink } from "./display-view-store";

export type RestoreQueuedMessagesCallback = (payload: {
  forkId: string | null;
  messages: readonly { id: string; content: string; taskMode: boolean }[];
}) => void;

export type ResyncDisplayViewCallback = (
  sessionId: string,
  viewId: string
) => void;

// Compile the patch map once at module level for the DisplayViewSnapshot schema.
const patchMap = compilePatchMap(DisplayViewSnapshot);

export function applyStreamEvent(
  store: DisplaySyncSink,
  event: StreamEvent,
  resync: ResyncDisplayViewCallback | null,
  sessionId: string,
  viewId: string,
  onRestoreQueuedMessages?: RestoreQueuedMessagesCallback
): Effect.Effect<void> {
  switch (event._tag) {
    case "state":
      return Effect.sync(() => {
        const snapshot = { shape: event.shape, state: event.state };
        store.accept(snapshot);
      });

    case "patch":
      return Effect.gen(function* () {
        const prev = store.acceptedSnapshot();
        const result = yield* applyDecodedPatch(prev, event.ops, patchMap);
        const revalidated = yield* Schema.validate(DisplayViewSnapshot)(result);
        store.accept(revalidated);
      }).pipe(
        Effect.catchAll((error) =>
          Effect.gen(function* () {
            yield* Effect.logWarning(
              `Failed to apply state patch; requesting full state resync: ${error.message} [${sessionId}]`
            );
            if (resync) {
              // A thrown callback is a programming defect. Preserve it as a
              // defect instead of logging it and pretending resync succeeded.
              yield* Effect.sync(() => resync(sessionId, viewId));
            }
          })
        )
      );

    case "restore_queued_messages":
      return Effect.sync(() =>
        onRestoreQueuedMessages?.({
          forkId: event.forkId,
          messages: event.messages,
        })
      );
  }
}
