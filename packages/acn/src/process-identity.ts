import * as Command from "@effect/platform/Command";
import * as CommandExecutor from "@effect/platform/CommandExecutor";
import {
  AcnProcessStartIdentitySchema,
  type AcnProcessStartIdentity,
} from "@magnitudedev/acn-protocol";
import { Data, Effect, Option } from "effect";

export class AcnProcessInspectionFailed extends Data.TaggedError(
  "AcnProcessInspectionFailed",
)<{
  readonly pid: number;
  readonly reason: string;
}> {}

const commandFor = (pid: number): Command.Command =>
  process.platform === "win32"
    ? Command.make(
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().Ticks`,
      )
    : Command.make("/bin/ps", "-o", "lstart=", "-p", String(pid));

export const readProcessStartIdentity = (
  pid: number,
): Effect.Effect<
  Option.Option<AcnProcessStartIdentity>,
  AcnProcessInspectionFailed,
  CommandExecutor.CommandExecutor
> => {
  const exists = (): boolean => {
    try {
      process.kill(pid, 0);
      return true;
    } catch (error) {
      return !(
        error instanceof Error &&
        "code" in error &&
        error.code === "ESRCH"
      );
    }
  };

  return Effect.gen(function* () {
    if (!exists()) return Option.none();

    const inspected = yield* Command.string(commandFor(pid)).pipe(
      Effect.timeout("1 second"),
      Effect.map((output) => output.trim()),
      Effect.either,
    );
    if (inspected._tag === "Right" && inspected.right.length > 0) {
      return Option.some(AcnProcessStartIdentitySchema.make(inspected.right));
    }
    if (!exists()) return Option.none();

    return yield* new AcnProcessInspectionFailed({
      pid,
      reason:
        inspected._tag === "Left"
          ? String(inspected.left)
          : "process inspection returned an empty identity",
    });
  });
};

export const currentProcessStartIdentity: Effect.Effect<
  AcnProcessStartIdentity,
  AcnProcessInspectionFailed,
  CommandExecutor.CommandExecutor
> = readProcessStartIdentity(process.pid).pipe(
  Effect.flatMap(
    Option.match({
      onNone: () =>
        Effect.fail(
          new AcnProcessInspectionFailed({
            pid: process.pid,
            reason: "current ACN process disappeared during identity inspection",
          }),
        ),
      onSome: Effect.succeed,
    }),
  ),
);
