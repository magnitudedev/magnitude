import {
  Clock,
  Context,
  Data,
  Deferred,
  Duration,
  Effect,
  Exit,
  Option,
  Ref,
  Scope,
} from "effect";

export class ResourceRetired extends Data.TaggedError("ResourceRetired")<{
  readonly resource: string;
  readonly generation: number;
}> {}

export interface ResourceUseGateSnapshot {
  readonly resource: string;
  readonly generation: number;
  readonly phase: "open" | "retirement-claimed" | "retired";
  readonly leaseCount: number;
  readonly idleSince: number | null;
  readonly revision: number;
}

export interface ResourceRetirementClaim {
  readonly resource: string;
  readonly generation: number;
  readonly reason: string;
  readonly revision: number;
}

export interface ResourceUseGate {
  readonly resource: string;
  readonly generation: number;
  readonly acquire: (
    label: string
  ) => Effect.Effect<Effect.Effect<void>, ResourceRetired>;
  readonly joinIfBusy: (
    label: string
  ) => Effect.Effect<Option.Option<Effect.Effect<void>>, ResourceRetired>;
  readonly withUse: <A, E, R>(
    label: string,
    effect: Effect.Effect<A, E, R>
  ) => Effect.Effect<A, E | ResourceRetired, R>;
  readonly withBusyUse: <A, E, R>(
    label: string,
    effect: Effect.Effect<A, E, R>
  ) => Effect.Effect<Option.Option<A>, E | ResourceRetired, R>;
  readonly retireNow: (reason: string) => Effect.Effect<boolean>;
  /** Atomically rejects new admission while allowing exact existing leases to release. */
  readonly closeAdmission: Effect.Effect<void>;
  readonly snapshot: Effect.Effect<ResourceUseGateSnapshot>;
  readonly awaitRetired: Effect.Effect<void>;
}

export interface ResourceUseGateOptions<E = never, R = never> {
  readonly resource: string;
  readonly generation: number;
  readonly idleTimeout: Duration.DurationInput;
  /**
   * Runs after admission has been closed and all leases are known to be absent.
   * Returning false rolls the claim back and opens a fresh idle interval.
   */
  readonly retire: (
    claim: ResourceRetirementClaim
  ) => Effect.Effect<boolean, E, R>;
}

interface OpenState {
  readonly phase: "open";
  readonly accepting: boolean;
  readonly leases: ReadonlyMap<number, string>;
  readonly idleSince: number | null;
  readonly revision: number;
  readonly changed: Deferred.Deferred<void>;
}

interface ClaimedState {
  readonly phase: "retirement-claimed";
  readonly claimId: number;
  readonly reason: string;
  readonly revision: number;
  readonly changed: Deferred.Deferred<void>;
  readonly resolution: Deferred.Deferred<"rolled-back" | "retired">;
}

interface RetiredState {
  readonly phase: "retired";
  readonly revision: number;
  readonly changed: Deferred.Deferred<void>;
}

type GateState = OpenState | ClaimedState | RetiredState;

type Admission =
  | {
      readonly _tag: "acquired";
      readonly token: number;
      readonly notify: Deferred.Deferred<void>;
    }
  | {
      readonly _tag: "wait";
      readonly resolution: Deferred.Deferred<"rolled-back" | "retired">;
    }
  | { readonly _tag: "retired" };

type ClaimAttempt =
  | {
      readonly _tag: "claimed";
      readonly claimId: number;
      readonly revision: number;
      readonly resolution: Deferred.Deferred<"rolled-back" | "retired">;
      readonly notify: Deferred.Deferred<void>;
    }
  | {
      readonly _tag: "wait";
      readonly changed: Deferred.Deferred<void>;
      readonly delayMs: number | null;
    }
  | { readonly _tag: "done" };

const completeChange = (changed: Deferred.Deferred<void>) =>
  Deferred.succeed(changed, undefined).pipe(Effect.asVoid);

const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)

/**
 * Creates the single use/admission state machine shared by ACN and session
 * resource generations. The returned gate and its deadline watcher are owned
 * by the current Effect scope.
 */
export const makeResourceUseGate = <E = never, R = never>(
  options: ResourceUseGateOptions<E, R>
): Effect.Effect<ResourceUseGate, never, Scope.Scope | R> =>
  Effect.gen(function* () {
    const idleTimeoutMs = Duration.toMillis(
      Duration.decode(options.idleTimeout)
    );
    if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs < 0) {
      return yield* Effect.die(
        new TypeError(
          "Resource idle timeout must be a finite, non-negative duration"
        )
      );
    }
    const retireContext = yield* Effect.context<R>();
    const initialChanged = yield* Deferred.make<void>();
    const retired = yield* Deferred.make<void>();
    const startedAt = yield* monotonicMillis;
    const state = yield* Ref.make<GateState>({
      phase: "open",
      accepting: true,
      leases: new Map(),
      idleSince: startedAt,
      revision: 0,
      changed: initialChanged,
    });
    let nextToken = 0;
    let nextClaim = 0;

    const resourceRetired = () =>
      new ResourceRetired({
        resource: options.resource,
        generation: options.generation,
      });

    const release = (token: number): Effect.Effect<void> =>
      Effect.gen(function* () {
        const now = yield* monotonicMillis;
        const nextChanged = yield* Deferred.make<void>();
        const changed = yield* Ref.modify(
          state,
          (current): readonly [Deferred.Deferred<void> | null, GateState] => {
            if (current.phase !== "open" || !current.leases.has(token))
              return [null, current];

            const leases = new Map(current.leases);
            leases.delete(token);
            return [
              current.changed,
              {
                phase: "open",
                accepting: current.accepting,
                leases,
                idleSince: leases.size === 0 ? now : null,
                revision: current.revision + 1,
                changed: nextChanged,
              },
            ];
          }
        );
        if (changed) yield* completeChange(changed);
      }).pipe(Effect.uninterruptible);

    const admit = (
      label: string,
      onlyIfBusy: boolean
    ): Effect.Effect<Option.Option<Effect.Effect<void>>, ResourceRetired> =>
      Effect.suspend(() =>
        Effect.uninterruptibleMask((restore) =>
          Effect.gen(function* () {
            const changed = yield* Deferred.make<void>();
            const admission = yield* Ref.modify(
              state,
              (current): readonly [Admission | null, GateState] => {
                if (current.phase === "retired")
                  return [{ _tag: "retired" }, current];
                if (current.phase === "retirement-claimed") {
                  return [
                    { _tag: "wait", resolution: current.resolution },
                    current,
                  ];
                }
                if (!current.accepting && !onlyIfBusy) {
                  return [{ _tag: "retired" }, current];
                }
                if (onlyIfBusy && current.leases.size === 0)
                  return [null, current];

                const token = ++nextToken;
                const leases = new Map(current.leases).set(token, label);
                return [
                  { _tag: "acquired", token, notify: current.changed },
                  {
                    phase: "open",
                    accepting: current.accepting,
                    leases,
                    idleSince: null,
                    revision: current.revision + 1,
                    changed,
                  },
                ];
              }
            );

            if (admission === null) return Option.none<Effect.Effect<void>>();
            if (admission._tag === "retired") return yield* resourceRetired();
            if (admission._tag === "wait") {
              const resolution = yield* restore(
                Deferred.await(admission.resolution)
              );
              if (resolution === "retired") return yield* resourceRetired();
              return yield* restore(admit(label, onlyIfBusy));
            }

            yield* completeChange(admission.notify);
            return Option.some(release(admission.token));
          })
        )
      );

    const acquire = (label: string) =>
      admit(label, false).pipe(
        Effect.flatMap(
          Option.match({
            onNone: () => Effect.fail(resourceRetired()),
            onSome: Effect.succeed,
          })
        )
      );

    const claim = (
      reason: string,
      force: boolean
    ): Effect.Effect<ClaimAttempt> =>
      Effect.gen(function* () {
        const now = yield* monotonicMillis;
        const changed = yield* Deferred.make<void>();
        const resolution = yield* Deferred.make<"rolled-back" | "retired">();
        const result = yield* Ref.modify(
          state,
          (current): readonly [ClaimAttempt, GateState] => {
            if (current.phase === "retired") return [{ _tag: "done" }, current];
            if (current.phase === "retirement-claimed") {
              return [
                { _tag: "wait", changed: current.changed, delayMs: null },
                current,
              ];
            }
            if (current.leases.size > 0) {
              return [
                { _tag: "wait", changed: current.changed, delayMs: null },
                current,
              ];
            }

            const idleSince = current.idleSince ?? now;
            const remaining = idleSince + idleTimeoutMs - now;
            if (!force && remaining > 0) {
              return [
                { _tag: "wait", changed: current.changed, delayMs: remaining },
                current,
              ];
            }

            const claimId = ++nextClaim;
            const revision = current.revision + 1;
            return [
              {
                _tag: "claimed",
                claimId,
                revision,
                resolution,
                notify: current.changed,
              },
              {
                phase: "retirement-claimed",
                claimId,
                reason,
                revision,
                changed,
                resolution,
              },
            ];
          }
        );
        if (result._tag === "claimed") yield* completeChange(result.notify);
        return result;
      }).pipe(Effect.uninterruptible);

    const resolveClaim = (
      attempt: Extract<ClaimAttempt, { readonly _tag: "claimed" }>,
      commit: boolean
    ): Effect.Effect<boolean> =>
      Effect.gen(function* () {
        const now = yield* monotonicMillis;
        const nextChanged = yield* Deferred.make<void>();
        const result = yield* Ref.modify(
          state,
          (
            current
          ): readonly [
            {
              readonly applied: boolean;
              readonly changed: Deferred.Deferred<void> | null;
            },
            GateState
          ] => {
            if (
              current.phase !== "retirement-claimed" ||
              current.claimId !== attempt.claimId
            ) {
              return [{ applied: false, changed: null }, current];
            }
            if (commit) {
              return [
                { applied: true, changed: current.changed },
                {
                  phase: "retired",
                  revision: current.revision + 1,
                  changed: nextChanged,
                },
              ];
            }
            return [
              { applied: true, changed: current.changed },
              {
                phase: "open",
                accepting: true,
                leases: new Map(),
                idleSince: now,
                revision: current.revision + 1,
                changed: nextChanged,
              },
            ];
          }
        );
        if (!result.applied) return false;
        if (result.changed) yield* completeChange(result.changed);
        yield* Deferred.succeed(
          attempt.resolution,
          commit ? "retired" : "rolled-back"
        );
        if (commit) yield* Deferred.succeed(retired, undefined);
        return true;
      }).pipe(Effect.uninterruptible);

    const executeClaim = (
      attempt: Extract<ClaimAttempt, { readonly _tag: "claimed" }>,
      reason: string
    ) =>
      Effect.uninterruptibleMask((restore) =>
        restore(
          options
            .retire({
              resource: options.resource,
              generation: options.generation,
              reason,
              revision: attempt.revision,
            })
            .pipe(Effect.provide(retireContext))
        ).pipe(
          Effect.exit,
          Effect.flatMap((exit) => {
            const commit = Exit.isSuccess(exit) && exit.value;
            return resolveClaim(attempt, commit).pipe(
              Effect.map((applied) => applied && commit)
            );
          })
        )
      );

    const waitForAttempt = (
      attempt: Extract<ClaimAttempt, { readonly _tag: "wait" }>
    ) => {
      const changed = Deferred.await(attempt.changed);
      return attempt.delayMs === null
        ? changed
        : Effect.raceFirst(
            changed,
            Effect.sleep(Duration.millis(Math.max(0, attempt.delayMs)))
          );
    };

    const deadlineLoop: Effect.Effect<void> = Effect.suspend(() =>
      claim("idle-timeout", false).pipe(
        Effect.flatMap((attempt) => {
          if (attempt._tag === "done") return Effect.void;
          if (attempt._tag === "wait")
            return waitForAttempt(attempt).pipe(Effect.zipRight(deadlineLoop));
          return executeClaim(attempt, "idle-timeout").pipe(
            Effect.zipRight(deadlineLoop)
          );
        })
      )
    );
    yield* deadlineLoop.pipe(Effect.forkScoped);

    const retireNow = (reason: string): Effect.Effect<boolean> =>
      claim(reason, true).pipe(
        Effect.flatMap((attempt) => {
          if (attempt._tag === "done") return Effect.succeed(false);
          if (attempt._tag === "wait") {
            return waitForAttempt(attempt).pipe(
              Effect.zipRight(retireNow(reason))
            );
          }
          return executeClaim(attempt, reason);
        })
      );

    const closeAdmission = Effect.gen(function* () {
      const changed = yield* Deferred.make<void>();
      const notify = yield* Ref.modify(
        state,
        (current): readonly [Deferred.Deferred<void> | null, GateState] => {
          if (current.phase !== "open" || !current.accepting) return [null, current];
          return [
            current.changed,
            {
              ...current,
              accepting: false,
              revision: current.revision + 1,
              changed,
            },
          ];
        },
      );
      if (notify) yield* completeChange(notify);
    }).pipe(Effect.uninterruptible);

    const snapshot = Ref.get(state).pipe(
      Effect.map(
        (current): ResourceUseGateSnapshot => ({
          resource: options.resource,
          generation: options.generation,
          phase: current.phase,
          leaseCount: current.phase === "open" ? current.leases.size : 0,
          idleSince: current.phase === "open" ? current.idleSince : null,
          revision: current.revision,
        })
      )
    );

    const withBusyUse = <A, E2, R2>(
      label: string,
      effect: Effect.Effect<A, E2, R2>
    ): Effect.Effect<Option.Option<A>, E2 | ResourceRetired, R2> =>
      Effect.acquireUseRelease(
        admit(label, true),
        (lease) =>
          Option.isSome(lease)
            ? Effect.map(effect, Option.some)
            : Effect.succeed(Option.none<A>()),
        (lease) => (Option.isSome(lease) ? lease.value : Effect.void)
      );

    return {
      resource: options.resource,
      generation: options.generation,
      acquire,
      joinIfBusy: (label) => admit(label, true),
      withUse: (label, effect) =>
        Effect.acquireUseRelease(
          acquire(label),
          () => effect,
          (release) => release
        ),
      withBusyUse,
      retireNow,
      closeAdmission,
      snapshot,
      awaitRetired: Deferred.await(retired),
    };
  });
