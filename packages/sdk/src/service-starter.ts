import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import { Context, Effect, Layer, Stream } from "effect";
import type { ServiceStartProgress } from "@magnitudedev/acn-protocol";
import {
  ServiceCommandFailed,
  ServiceExecutableNotFound,
  ServiceStartFailed,
  type ServiceStartError,
} from "./connection-errors";

export interface MagnitudeServiceStarter {
  readonly start: Stream.Stream<ServiceStartProgress, ServiceStartError>;
}
export const MagnitudeServiceStarter = Object.assign(
  Context.GenericTag<MagnitudeServiceStarter>(
    "@magnitudedev/sdk/MagnitudeServiceStarter"
  ),
  {
    cliLayer,
  }
);

function cliLayer(
  options: { readonly executable?: string } = {}
): Layer.Layer<
  MagnitudeServiceStarter,
  never,
  CommandExecutor.CommandExecutor
> {
  return Layer.effect(
    MagnitudeServiceStarter,
    Effect.gen(function* () {
      const executor = yield* CommandExecutor.CommandExecutor;
      const executable = options.executable ?? "magnitude";
      const start = Effect.scoped(
        Effect.gen(function* () {
          const process = yield* Command.make(
            executable,
            "service",
            "start"
          ).pipe(Command.start);
          const [exitCode, , stderr] = yield* Effect.all(
            [
              process.exitCode,
              Stream.runDrain(process.stdout),
              process.stderr.pipe(
                Stream.decodeText(),
                Stream.runFold("", (tail, part) => (tail + part).slice(-16_384))
              ),
            ],
            { concurrency: "unbounded" }
          );
          if (exitCode !== 0)
            return yield* new ServiceCommandFailed({
              executable,
              exitCode,
              stderr,
            });
        })
      ).pipe(
        Effect.catchIf(
          (error) =>
            error._tag === "SystemError" && error.reason === "NotFound",
          () => Effect.fail(new ServiceExecutableNotFound({ executable }))
        ),
        Effect.mapError((error) =>
          error._tag === "ServiceCommandFailed" ||
          error._tag === "ServiceExecutableNotFound"
            ? error
            : new ServiceStartFailed({ message: String(error) })
        ),
        Effect.timeoutFail({
          duration: "10 minutes",
          onTimeout: () =>
            new ServiceStartFailed({
              message: "Magnitude service start timed out",
            }),
        }),
        Effect.provideService(CommandExecutor.CommandExecutor, executor)
      );
      return MagnitudeServiceStarter.of({ start: Stream.execute(start) });
    })
  );
}
