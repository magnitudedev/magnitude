import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import * as HttpClient from "@effect/platform/HttpClient";
import {
  Context,
  Data,
  Duration,
  Effect,
  Layer,
  Option,
  Ref,
  Schedule,
  Schema,
  Scope,
  Stream,
} from "effect";
import { type IcnClient, IcnClientError, makeIcnClient } from "./client.js";

const PositivePort = Schema.Int.pipe(
  Schema.greaterThanOrEqualTo(1),
  Schema.lessThanOrEqualTo(65_535)
);
const PositiveInt = Schema.Int.pipe(Schema.greaterThanOrEqualTo(1));
const NonNegativeInt = Schema.Int.pipe(Schema.greaterThanOrEqualTo(0));

export const IcnHost = Schema.Literal("127.0.0.1", "::1");
export type IcnHost = typeof IcnHost.Type;
export const IcnFlashAttention = Schema.Literal("auto", "off", "on");
export type IcnFlashAttention = typeof IcnFlashAttention.Type;

export class IcnProcessOptions extends Schema.Class<IcnProcessOptions>(
  "IcnProcessOptions"
)({
  executable: Schema.String.pipe(Schema.minLength(1)),
  modelPath: Schema.OptionFromSelf(Schema.String.pipe(Schema.minLength(1))),
  modelAlias: Schema.OptionFromSelf(Schema.String.pipe(Schema.minLength(1))),
  host: IcnHost,
  port: PositivePort,
  contextSize: PositiveInt,
  batchSize: PositiveInt,
  ubatchSize: PositiveInt,
  maxSequences: PositiveInt,
  prefillQuantum: PositiveInt,
  gpuLayers: NonNegativeInt,
  threads: Schema.OptionFromSelf(PositiveInt),
  threadsBatch: Schema.OptionFromSelf(PositiveInt),
  flashAttention: IcnFlashAttention,
  startupTimeout: Schema.DurationFromSelf.pipe(
    Schema.greaterThanDuration(Duration.zero)
  ),
  outputLimitBytes: PositiveInt,
}) {}

export const IcnProcessOperation = Schema.Literal(
  "start",
  "wait-ready",
  "observe-exit"
);
export type IcnProcessOperation = typeof IcnProcessOperation.Type;
export const IcnProcessFailureReason = Schema.Literal(
  "command-failed",
  "startup-timeout",
  "exited-before-ready",
  "health-failed"
);
export type IcnProcessFailureReason = typeof IcnProcessFailureReason.Type;

export class IcnProcessError extends Data.TaggedError("IcnProcessError")<{
  readonly operation: IcnProcessOperation;
  readonly reason: IcnProcessFailureReason;
  readonly message: string;
  readonly diagnostic: Option.Option<string>;
}> {}

export interface IcnProcess {
  readonly pid: number;
  readonly origin: URL;
  readonly client: IcnClient;
  readonly output: Effect.Effect<string>;
  readonly exited: Effect.Effect<number, IcnProcessError>;
}

export class Icn extends Context.Tag("@magnitudedev/icn/Icn")<
  Icn,
  IcnProcess
>() {}

const processError = (
  operation: IcnProcessOperation,
  reason: IcnProcessFailureReason,
  message: string,
  diagnostic = Option.none<string>()
) => new IcnProcessError({ operation, reason, message, diagnostic });

const optionArguments = <A>(
  value: Option.Option<A>,
  render: (value: A) => readonly string[]
): readonly string[] =>
  Option.match(value, { onNone: () => [], onSome: render });

export const renderIcnArguments = (
  options: IcnProcessOptions
): readonly string[] => [
  "serve",
  "--bind",
  `${options.host === "::1" ? "[::1]" : options.host}:${options.port}`,
  ...optionArguments(options.modelPath, (modelPath) => [
    "--model",
    modelPath,
    ...optionArguments(options.modelAlias, (alias) => ["--model-alias", alias]),
  ]),
  "--context-size",
  String(options.contextSize),
  "--batch-size",
  String(options.batchSize),
  "--ubatch-size",
  String(options.ubatchSize),
  "--max-sequences",
  String(options.maxSequences),
  "--prefill-quantum",
  String(options.prefillQuantum),
  "--gpu-layers",
  String(options.gpuLayers),
  ...optionArguments(options.threads, (threads) => [
    "--threads",
    String(threads),
  ]),
  ...optionArguments(options.threadsBatch, (threads) => [
    "--threads-batch",
    String(threads),
  ]),
  "--flash-attention",
  options.flashAttention,
];

const captureOutput = (
  process: CommandExecutor.Process,
  limit: number
): Effect.Effect<Ref.Ref<string>, never, Scope.Scope> =>
  Effect.gen(function* () {
    const output = yield* Ref.make("");
    yield* Stream.merge(process.stdout, process.stderr).pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) =>
        Ref.update(output, (current) => `${current}${chunk}`.slice(-limit))
      ),
      Effect.catchAll(() => Effect.void),
      Effect.forkScoped
    );
    return output;
  });

const withDiagnostic = (
  error: IcnProcessError,
  output: Ref.Ref<string>
): Effect.Effect<never, IcnProcessError> =>
  Ref.get(output).pipe(
    Effect.flatMap((diagnostic) =>
      Effect.fail(
        new IcnProcessError({
          ...error,
          diagnostic:
            diagnostic.trim().length === 0
              ? Option.none()
              : Option.some(diagnostic),
        })
      )
    )
  );

export const makeIcnProcess = (
  options: IcnProcessOptions
): Effect.Effect<
  IcnProcess,
  IcnProcessError,
  CommandExecutor.CommandExecutor | HttpClient.HttpClient | Scope.Scope
> =>
  Effect.gen(function* () {
    const process = yield* Command.start(
      Command.make(options.executable, ...renderIcnArguments(options))
    ).pipe(
      Effect.mapError((error) =>
        processError("start", "command-failed", String(error))
      )
    );
    const output = yield* captureOutput(process, options.outputLimitBytes);
    yield* Effect.addFinalizer(() =>
      process.isRunning.pipe(
        Effect.flatMap((running) =>
          running ? process.kill("SIGTERM") : Effect.void
        ),
        Effect.catchAll(() => Effect.void)
      )
    );

    const address = options.host === "::1" ? "[::1]" : options.host;
    const origin = new URL(`http://${address}:${options.port}`);
    const client = yield* makeIcnClient(origin);
    const healthReady = client.health.pipe(
      Effect.retry(Schedule.spaced("50 millis")),
      Effect.timeoutFail({
        duration: options.startupTimeout,
        onTimeout: () =>
          processError(
            "wait-ready",
            "startup-timeout",
            `ICN did not become healthy within ${Duration.format(
              options.startupTimeout
            )}`
          ),
      }),
      Effect.mapError((error) =>
        error instanceof IcnClientError
          ? processError("wait-ready", "health-failed", error.message)
          : error
      )
    );
    const earlyExit = process.exitCode.pipe(
      Effect.mapError((error) =>
        processError("observe-exit", "command-failed", String(error))
      ),
      Effect.flatMap((exitCode) =>
        Effect.fail(
          processError(
            "wait-ready",
            "exited-before-ready",
            `ICN exited with status ${Number(exitCode)} before becoming healthy`
          )
        )
      )
    );
    yield* Effect.raceFirst(healthReady, earlyExit).pipe(
      Effect.catchAll((error) => withDiagnostic(error, output))
    );

    return {
      pid: Number(process.pid),
      origin,
      client,
      output: Ref.get(output),
      exited: process.exitCode.pipe(
        Effect.map(Number),
        Effect.mapError((error) =>
          processError("observe-exit", "command-failed", String(error))
        )
      ),
    };
  });

export const IcnLive = (
  options: IcnProcessOptions
): Layer.Layer<
  Icn,
  IcnProcessError,
  CommandExecutor.CommandExecutor | HttpClient.HttpClient
> => Layer.scoped(Icn, makeIcnProcess(options));
