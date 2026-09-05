import { Command, CommandExecutor } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect, Layer, Stream } from "effect";
import { describe, expect, it } from "vitest";
import { MagnitudeServiceStarter } from "./service-starter";

const run = (
  script: string,
  inspect: (command: Command.Command) => void = () => {}
) =>
  Effect.gen(function* () {
    const executor = yield* CommandExecutor.CommandExecutor;
    const substitute = CommandExecutor.makeExecutor((command) =>
      Effect.sync(() => inspect(command)).pipe(
        Effect.zipRight(
          executor.start(Command.make(process.execPath, "-e", script))
        )
      )
    );
    return yield* Effect.gen(function* () {
      const starter = yield* MagnitudeServiceStarter;
      return yield* starter.start.pipe(Stream.runCollect);
    }).pipe(
      Effect.provide(
        MagnitudeServiceStarter.cliLayer({
          executable: "/path with spaces/magnitude",
        }).pipe(
          Layer.provide(
            Layer.succeed(CommandExecutor.CommandExecutor, substitute)
          )
        )
      )
    );
  });

describe("SDK CLI service starter", () => {
  it("uses an argv command, drains human output, and parses no JSON", async () => {
    const result = await Effect.runPromise(
      run(
        "console.log('ordinary human output'); console.error('diagnostic')",
        (command) => {
          expect(command._tag).toBe("StandardCommand");
          if (command._tag === "StandardCommand") {
            expect(command.command).toBe("/path with spaces/magnitude");
            expect(command.args).toEqual(["service", "start"]);
            expect(command.shell).toBe(false);
          }
        }
      ).pipe(Effect.provide(BunContext.layer))
    );
    expect([...result]).toEqual([]);
  });
  it("reports exit status with bounded diagnostic output", async () => {
    const result = await Effect.runPromise(
      run("console.error('x'.repeat(20000)); process.exit(7)").pipe(
        Effect.either,
        Effect.provide(BunContext.layer)
      )
    );
    expect(result._tag).toBe("Left");
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("ServiceCommandFailed");
      if (result.left._tag === "ServiceCommandFailed") {
        expect(result.left.exitCode).toBe(7);
        expect(result.left.stderr.length).toBe(16384);
      }
    }
  });
  it("exposes a missing executable as its own error", async () => {
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const starter = yield* MagnitudeServiceStarter;
        return yield* starter.start.pipe(Stream.runDrain);
      }).pipe(
        Effect.provide(
          MagnitudeServiceStarter.cliLayer({
            executable: "/missing-magnitude-sdk-test-executable",
          })
        ),
        Effect.either,
        Effect.provide(BunContext.layer)
      )
    );
    expect(result._tag === "Left" && result.left._tag).toBe(
      "ServiceExecutableNotFound"
    );
  });
});
