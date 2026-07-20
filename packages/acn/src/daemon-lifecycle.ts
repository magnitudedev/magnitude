import * as HttpServer from "@effect/platform/HttpServer";
import * as FileSystem from "@effect/platform/FileSystem";
import * as NodePath from "path";
import * as NodeOs from "os";
import { Cause, Context, Effect, Layer, Runtime, Schedule } from "effect";
import { AgentRuntime } from "./agent-runtime";
import { AcnActivityTracker } from "./activity-tracker";
import {
  type AcnRegistration,
  readRegistration,
  registrationIsOwnedBy,
  registrationPath,
  writeRegistrationAtomic,
} from "./daemon-registration";
import { ACN_OWNER_ID } from "./identity";

export interface DaemonLifecycleOptions {
  readonly version: string;
  readonly register: boolean;
  readonly debug: boolean;
  readonly idleTimeoutMinutes: number;
  readonly checkIntervalSeconds: number;
  readonly ownershipCheckIntervalMs?: number;
  readonly ownershipCheckTimeoutMs?: number;
  readonly dataDir: string;
}

export const defaultDataDir = (): string =>
  NodePath.join(NodeOs.homedir(), ".magnitude");

const getServerUrl = (address: HttpServer.Address): string => {
  if (address._tag === "UnixAddress") {
    throw new TypeError("Unix sockets are not supported for ACN registration");
  }
  return `http://${
    address.hostname === "0.0.0.0" ? "127.0.0.1" : address.hostname
  }:${address.port}`;
};

export const DaemonLifecycleLive = (
  options: DaemonLifecycleOptions
): Layer.Layer<
  never,
  never,
  | AgentRuntime
  | AcnActivityTracker
  | HttpServer.HttpServer
  | FileSystem.FileSystem
> =>
  Layer.scopedDiscard(
    Effect.gen(function* () {
      const agentRuntime = yield* AgentRuntime;
      const activity = yield* AcnActivityTracker;
      const server = yield* HttpServer.HttpServer;
      const runtime = yield* Effect.runtime<
        | AgentRuntime
        | AcnActivityTracker
        | HttpServer.HttpServer
        | FileSystem.FileSystem
      >();
      const url = getServerUrl(server.address);
      const registrationPath_ = registrationPath(
        options.dataDir,
        options.version
      );
      const ownerId = ACN_OWNER_ID;

      const registerDaemon = Effect.fn("acn.register")(function* () {
        const registration: AcnRegistration = {
          id: ownerId,
          version: options.version,
          url,
          pid: process.pid,
          timestamp: Date.now(),
        };
        yield* Effect.annotateCurrentSpan({
          url,
          pid: process.pid,
          version: options.version,
        });
        yield* writeRegistrationAtomic(registrationPath_, registration).pipe(
          Effect.catchAll((error) =>
            Effect.gen(function* () {
              yield* Effect.logError(
                "Failed to write ACN registration file; exiting"
              ).pipe(Effect.annotateLogs({ error: String(error) }));
              return yield* Effect.sync(() => process.exit(1));
            })
          )
        );
        yield* Effect.logInfo("ACN daemon registered").pipe(
          Effect.annotateLogs({
            url,
            id: ownerId,
            pid: process.pid,
            version: options.version,
          })
        );
        if (options.debug) {
          yield* Effect.logDebug("ACN registered").pipe(
            Effect.annotateLogs({
              id: ownerId,
              pid: process.pid,
              version: options.version,
              url,
            })
          );
        }
      });

      if (options.register) {
        yield* registerDaemon();
      }

      const timers: Array<ReturnType<typeof setInterval>> = [];

      const shutdown = Effect.fn("acn.shutdown")(function* () {
        yield* Effect.annotateCurrentSpan({ pid: process.pid });
        yield* Effect.logInfo("ACN daemon shutting down gracefully");
        // Leave the canonical registration in place. Its health probe will
        // fail after exit, and the next owner atomically replaces it. Deleting
        // here could race with and remove a successor's registration.
        yield* agentRuntime.disposeAll();
        return yield* Effect.sync(() => process.exit(0));
      });

      const shutdownHandler = () => {
        Runtime.runPromise(runtime, shutdown()).catch((error) => {
          console.error("Failed to shut down ACN daemon:", error);
          process.exit(1);
        });
      };

      process.on("SIGTERM", shutdownHandler);
      process.on("SIGINT", shutdownHandler);

      const onTimerError =
        (label: string): ((error: unknown) => void) =>
        (error) => {
          Runtime.runPromise(
            runtime,
            Effect.gen(function* () {
              yield* Effect.logError(`${label}; ACN is unrecoverable`).pipe(
                Effect.annotateLogs({ error: String(error) })
              );
              return yield* Effect.sync(() => process.exit(1));
            })
          ).catch(() => process.exit(1));
        };

      const idleCheck = Effect.fn("acn.idle-check")(function* () {
        const hasActiveWork = (yield* agentRuntime.hasActiveWork) || (yield* activity.hasActiveWork);
        if (hasActiveWork) {
          yield* activity.touch("active-work");
          return;
        }

        const { lastActivityAt } = yield* activity.current;
        const idleMinutes = (Date.now() - lastActivityAt) / 1000 / 60;
        yield* Effect.annotateCurrentSpan({ idleMinutes, hasActiveWork });
        if (idleMinutes >= options.idleTimeoutMinutes) {
          yield* Effect.logWarning(
            `Idle timeout reached (${options.idleTimeoutMinutes}m); shutting down`
          );
          yield* Effect.sync(() => process.kill(process.pid, "SIGTERM"));
        }
      });

      timers.push(
        setInterval(() => {
          Runtime.runPromise(runtime, idleCheck()).catch(
            onTimerError("Error during idle check")
          );
        }, options.checkIntervalSeconds * 1000)
      );

      timers.push(
        setInterval(() => {
          Runtime.runPromise(runtime, agentRuntime.evictIdleSessions()).catch(
            onTimerError("Error during session eviction")
          );
        }, options.checkIntervalSeconds * 1000)
      );

      const checkRegistration = Effect.fn("acn.check-registration")(
        function* () {
          const reg = yield* readRegistration(registrationPath_);
          if (registrationIsOwnedBy(reg, ownerId)) return;

          yield* Effect.logWarning(
            "ACN can no longer prove registry ownership; exiting"
          ).pipe(
            Effect.annotateLogs({
              ownerId,
              observedOwner: reg?.id ?? null,
              observedUrl: reg?.url ?? null,
              observedPid: reg?.pid ?? null,
            })
          );
          // Ownership loss is fail-closed. Exit immediately rather than allowing
          // graceful cleanup to extend a split-brain window.
          return yield* Effect.sync(() => process.exit(0));
        }
      );

      if (options.register) {
        yield* checkRegistration().pipe(
          Effect.timeout(`${options.ownershipCheckTimeoutMs ?? 800} millis`),
          Effect.catchAllCause((cause) =>
            Effect.logError(
              "ACN registry ownership watchdog failed; exiting"
            ).pipe(
              Effect.annotateLogs({ cause: Cause.pretty(cause) }),
              Effect.andThen(Effect.sync(() => process.exit(1)))
            )
          ),
          Effect.repeat(
            Schedule.spaced(`${options.ownershipCheckIntervalMs ?? 1000} millis`)
          ),
          Effect.forkScoped
        );
      }

      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          for (const timer of timers) clearInterval(timer);
          process.off("SIGTERM", shutdownHandler);
          process.off("SIGINT", shutdownHandler);
        })
      );
    })
  );
