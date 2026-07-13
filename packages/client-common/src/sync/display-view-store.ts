import { Effect, Exit, Option, Schema } from "effect";
import {
  DisplayViewSnapshot,
  type DisplayMessage,
  type DisplayState,
  type DisplayTimeline,
  type DisplayTimelineEntry,
  type DisplayViewShape,
} from "@magnitudedev/sdk";
import {
  compilePatchMap,
  diffDecoded,
  type PatchApplyError,
  type DecodedPatchOp,
  type DecodedValue,
  type Path,
} from "@magnitudedev/utils/patch";
import { ReferencePreservingStore } from "./store";
import type { Transform } from "./share-refs";

type Mutable<T> = T extends (...args: any[]) => any
  ? T
  : T extends readonly (infer U)[]
  ? readonly Mutable<U>[]
  : T extends object
  ? { -readonly [K in keyof T]: Mutable<T[K]> }
  : T;

export type MutableDisplayViewSnapshot = Mutable<DisplayViewSnapshot>;
export type WriteKey = string;

export interface SpeculativeDisplayHandle {
  readonly id: string;
  readonly owner: string;
  readonly remove: () => void;
}

export interface SpeculativeMutationOptions {
  readonly owner: string;
  readonly label?: string;
}

interface SpeculativeDisplayTransaction {
  readonly id: string;
  readonly owner: string;
  readonly label: string | undefined;
  readonly createdAt: number;
  readonly apply: (draft: MutableDisplayViewSnapshot) => void;
  lastWriteKeys: readonly WriteKey[];
}

export interface DisplayReader {
  readonly getSnapshot: () => DisplayViewSnapshot;
  readonly subscribe: (listener: () => void) => () => void;
}

export interface DisplaySyncSink {
  readonly accept: (next: DisplayViewSnapshot) => void;
  readonly resetAccepted: (next: DisplayViewSnapshot) => void;
  /** Returns the current decoded accepted state. */
  readonly acceptedSnapshot: () => DisplayViewSnapshot;
}

export interface DisplaySpeculator {
  readonly mutate: (
    options: SpeculativeMutationOptions,
    apply: (draft: MutableDisplayViewSnapshot) => void
  ) => SpeculativeDisplayHandle;
  readonly remove: (id: string) => void;
  readonly removeOwner: (owner: string) => void;
  readonly clear: () => void;
}

export interface DisplayViewStore
  extends DisplayReader,
    DisplaySyncSink,
    DisplaySpeculator {}

let speculativeIdCounter = 0;

const nextSpeculativeId = (): string =>
  `spec-${Date.now()}-${speculativeIdCounter++}`;

// Compile the patch map once for decoded-level diffing.
const patchMap = compilePatchMap(DisplayViewSnapshot);

function hasId(v: object): v is { id: string } {
  return "id" in v && typeof v.id === "string";
}

function hasKey(v: object): v is { _key: string } {
  return "_key" in v && typeof v._key === "string";
}

import type { DecodedSome, DecodedNone } from "@magnitudedev/utils/patch";

function isDecodedOption(v: DecodedValue): v is DecodedSome | DecodedNone {
  return (
    typeof v === "object" &&
    v !== null &&
    !Array.isArray(v) &&
    "_tag" in v &&
    (v._tag === "Some" || v._tag === "None")
  );
}

function isRecord(value: DecodedValue): value is Record<string, DecodedValue> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    !isDecodedOption(value)
  );
}

const injectKeyTransform: Transform = (obj) => {
  if (hasId(obj) && !hasKey(obj)) {
    return { ...obj, _key: obj.id };
  }
  return obj;
};

// ---------------------------------------------------------------------------
// Write key derivation from decoded patch ops
// ---------------------------------------------------------------------------

/** Convert a path array to a string key for conflict detection. */
function pathKey(path: Path): string {
  return path.map(String).join("/");
}

function opValue(
  op: DecodedPatchOp,
  before: DecodedValue
): DecodedValue | null {
  if (op.op === "replace" || op.op === "add") return op.value;
  // For remove/move, the value is at the source path in the before state
  if (op.op === "remove") {
    return valueAtPath(before, op.path);
  }
  // move
  return valueAtPath(before, op.from);
}

function valueAtPath(root: DecodedValue, path: Path): DecodedValue | null {
  let current: DecodedValue | null = root;
  for (const key of path) {
    if (current === null) return null;
    // Unwrap Option
    if (isDecodedOption(current)) {
      current = current._tag === "Some" ? current.value : null;
      if (current === null) return null;
    }
    if (Array.isArray(current)) {
      current = current[Number(key)] ?? null;
    } else if (isRecord(current)) {
      current = current[String(key)] ?? null;
    } else {
      return null;
    }
  }
  // Unwrap Option for value inspection
  if (current !== null && isDecodedOption(current)) {
    return current._tag === "Some" ? current.value : null;
  }
  return current;
}

function keyedPath(path: Path): string | null {
  const byIdIndex = path.lastIndexOf("byId");
  if (byIdIndex >= 0 && path[byIdIndex + 1] !== undefined) {
    return pathKey(path.slice(0, byIdIndex + 2));
  }

  const messagesIndex = path.findIndex(
    (part, index) =>
      part === "messages" &&
      path[index + 1] === "byId" &&
      path[index + 2] !== undefined
  );
  if (messagesIndex >= 0) {
    return pathKey(path.slice(0, messagesIndex + 3));
  }

  return null;
}

function orderPath(path: Path, value: DecodedValue | null): string | null {
  const orderIndex = path.lastIndexOf("order");
  if (orderIndex < 0) return null;
  const base = pathKey(path.slice(0, orderIndex + 1));
  if (path.length === orderIndex + 1) return base;
  if (value !== null && typeof value === "string")
    return `${base}/$member/${value}`;
  return base;
}

function presentationEntryPath(
  path: Path,
  value: DecodedValue | null
): string | null {
  const entriesIndex = path.lastIndexOf("entries");
  if (entriesIndex < 0) return null;
  const base = pathKey(path.slice(0, entriesIndex + 1));
  if (path.length === entriesIndex + 1) return base;
  if (value !== null && isRecord(value) && typeof value.id === "string") {
    return `${base}/$member/${value.id}`;
  }
  return base;
}

function normalizeOp(op: DecodedPatchOp, before: DecodedValue): WriteKey[] {
  const path = op.op === "move" ? op.to : op.path;
  if (path.length === 0) return [""];

  if (path.includes("window")) return [];

  const value = opValue(op, before);
  const presentationPath = presentationEntryPath(path, value);
  if (presentationPath) return [presentationPath];

  const order = orderPath(path, value);
  if (order) return [order];

  const keyed = keyedPath(path);
  if (keyed) return [keyed];

  if (path.length <= 2) return [pathKey(path)];
  return [pathKey(path)];
}

function normalizeOps(
  ops: readonly DecodedPatchOp[],
  before: DecodedValue
): readonly WriteKey[] {
  return [...new Set(ops.flatMap((op) => normalizeOp(op, before)))];
}

function writeKeysConflict(left: WriteKey, right: WriteKey): boolean {
  if (left === right) return true;
  if (left === "" || right === "") return true;
  return left.startsWith(`${right}/`) || right.startsWith(`${left}/`);
}

function hasWriteKeyConflict(
  left: readonly WriteKey[],
  right: readonly WriteKey[]
): boolean {
  return left.some((l) => right.some((r) => writeKeysConflict(l, r)));
}

function deriveMutation(
  base: DisplayViewSnapshot,
  apply: (draft: MutableDisplayViewSnapshot) => void
): Effect.Effect<
  {
    readonly next: DisplayViewSnapshot;
    readonly ops: readonly DecodedPatchOp[];
    readonly writeKeys: readonly WriteKey[];
  },
  PatchApplyError
> {
  return Effect.gen(function* () {
    // Clone via Schema encode→decode round-trip — preserves Option identity.
    const encoded = yield* Schema.encode(DisplayViewSnapshot)(base).pipe(
      Effect.orDie
    );
    const draft: MutableDisplayViewSnapshot = yield* Schema.decode(
      DisplayViewSnapshot
    )(encoded).pipe(Effect.orDie);
    apply(draft);
    const next: DisplayViewSnapshot = draft;
    const ops = yield* diffDecoded(base, next, patchMap);
    return {
      next,
      ops,
      writeKeys: normalizeOps(ops, base),
    };
  });
}

export class SpeculativeDisplayViewStore implements DisplayViewStore {
  private readonly accepted: ReferencePreservingStore<DisplayViewSnapshot>;
  private readonly rendered: ReferencePreservingStore<DisplayViewSnapshot>;
  private transactions: SpeculativeDisplayTransaction[] = [];

  constructor(initial: DisplayViewSnapshot, transform?: Transform) {
    // The accepted store holds raw server values — NO transform.
    // This ensures diffDecoded in accept() compares raw values without
    // _key contamination from injectKeyTransform.
    this.accepted = new ReferencePreservingStore<DisplayViewSnapshot>(initial);
    // The rendered store applies the transform (e.g. _key injection) for UI.
    this.rendered = new ReferencePreservingStore<DisplayViewSnapshot>(
      initial,
      transform
    );
  }

  getSnapshot = (): DisplayViewSnapshot => this.rendered.get();

  subscribe = (listener: () => void): (() => void) =>
    this.rendered.subscribe(listener);

  acceptedSnapshot = (): DisplayViewSnapshot => this.accepted.get();

  accept = (next: DisplayViewSnapshot): void => {
    // Fast path: no speculative transactions → no conflict detection.
    if (this.transactions.length === 0) {
      this.accepted.set(next);
      this.rendered.set(next);
      return;
    }

    // Slow path: transactions active → conflict detection via decoded diff.
    const prevAccepted = this.accepted.get();
    const authoritativeOps = Effect.runSyncExit(
      diffDecoded(prevAccepted, next, patchMap)
    );
    if (Exit.isFailure(authoritativeOps)) {
      this.accepted.set(next);
      this.rendered.set(next);
      return;
    }
    const authoritativeKeys = normalizeOps(
      authoritativeOps.value,
      prevAccepted
    );
    if (authoritativeKeys.length > 0) {
      this.transactions = this.transactions.filter(
        (tx) => !hasWriteKeyConflict(tx.lastWriteKeys, authoritativeKeys)
      );
    }
    this.accepted.set(next);
    this.recompute();
  };

  resetAccepted = (next: DisplayViewSnapshot): void => {
    this.accepted.set(next);
    this.recompute();
  };

  mutate = (
    options: SpeculativeMutationOptions,
    apply: (draft: MutableDisplayViewSnapshot) => void
  ): SpeculativeDisplayHandle => {
    const tx: SpeculativeDisplayTransaction = {
      id: nextSpeculativeId(),
      owner: options.owner,
      label: options.label,
      createdAt: Date.now(),
      apply,
      lastWriteKeys: [],
    };
    this.transactions = [...this.transactions, tx];
    this.recompute();
    return {
      id: tx.id,
      owner: tx.owner,
      remove: () => this.remove(tx.id),
    };
  };

  remove = (id: string): void => {
    const next = this.transactions.filter((tx) => tx.id !== id);
    if (next.length === this.transactions.length) return;
    this.transactions = next;
    this.recompute();
  };

  removeOwner = (owner: string): void => {
    const next = this.transactions.filter((tx) => tx.owner !== owner);
    if (next.length === this.transactions.length) return;
    this.transactions = next;
    this.recompute();
  };

  clear = (): void => {
    if (this.transactions.length === 0) return;
    this.transactions = [];
    this.rendered.set(this.accepted.get());
  };

  private recompute(): void {
    let rendered = this.accepted.get();
    const kept: SpeculativeDisplayTransaction[] = [];

    for (const tx of this.transactions) {
      const result = Effect.runSyncExit(deriveMutation(rendered, tx.apply));
      if (Exit.isFailure(result)) {
        continue;
      }
      if (result.value.ops.length === 0) {
        continue;
      }
      tx.lastWriteKeys = result.value.writeKeys;
      rendered = result.value.next;
      kept.push(tx);
    }

    if (kept.length !== this.transactions.length) {
      this.transactions = kept;
    }
    this.rendered.set(rendered);
  }
}

export const emptyTimeline = (): DisplayTimeline => ({
  mode: "idle",
  messages: { byId: {}, order: [] },
  streamingMessageId: null,
  window: {
    start: 0,
    end: 0,
    totalCount: 0,
    hasMoreBefore: false,
    hasMoreAfter: false,
  },
  presentation: {
    mode: "default",
    entries: [],
    statusSlot: { kind: "none" },
  },
});

export function appendMessageToTimeline(
  timeline: DisplayTimeline,
  message: DisplayMessage
): DisplayTimeline {
  const byId = {
    ...timeline.messages.byId,
    [message.id]: message,
  };
  const order = timeline.messages.order.includes(message.id)
    ? timeline.messages.order
    : [...timeline.messages.order, message.id];
  const role: Extract<DisplayTimelineEntry, { kind: "message" }>["role"] =
    message.type === "user_message" || message.type === "queued_user_message"
      ? "user"
      : message.type === "assistant_message"
      ? "assistant"
      : message.type === "status_indicator" ||
        message.type === "error" ||
        message.type === "interrupted"
      ? "system"
      : "agent";
  const entry: Extract<DisplayTimelineEntry, { kind: "message" }> = {
    kind: "message",
    id: `entry:${message.id}`,
    messageId: message.id,
    timestamp: message.timestamp,
    role,
    streaming: false,
    interrupted: false,
    nextMessageInterrupted: false,
  };
  const entries = timeline.presentation.entries.some(
    (candidate) => candidate.id === entry.id
  )
    ? timeline.presentation.entries
    : [...timeline.presentation.entries, entry];

  return {
    ...timeline,
    messages: { byId, order },
    window: {
      ...timeline.window,
      end: Math.max(timeline.window.end, order.length),
      totalCount: Math.max(timeline.window.totalCount, order.length),
    },
    presentation: {
      ...timeline.presentation,
      entries,
    },
  };
}

export function createDisplayViewStore(
  initial: DisplayState,
  shape: DisplayViewShape
): DisplayViewStore {
  return new SpeculativeDisplayViewStore(
    { shape, state: initial },
    injectKeyTransform
  );
}
