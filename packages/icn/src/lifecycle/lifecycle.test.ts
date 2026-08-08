import { Duration, Effect, Layer, Option } from "effect";
import * as FileSystem from "@effect/platform/FileSystem";
import { SystemError } from "@effect/platform/Error";
import * as BunContext from "@effect/platform-bun/BunContext";
import * as FetchHttpClient from "@effect/platform/FetchHttpClient";
import {
  chmod,
  mkdir,
  mkdtemp,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  IcnBinaryResolutionConfig,
  IcnBinaryResolver,
  IcnBinaryNotFound,
  IcnExitedBeforeReady,
  makeIcnBinaryResolver,
  IcnLifecycleConfig,
  IcnStorageConfig,
  renderIcnArguments,
  IcnPreparationReporter,
} from "./index.js";

const config = (host: "127.0.0.1" | "::1" = "127.0.0.1") =>
  new IcnLifecycleConfig({
    binary: new IcnBinaryResolutionConfig({
      source: {
        _tag: "Installation",
        path: "/opt/magnitude/installation.json",
      },
      supportedApiVersion: 1,
      expectedNativeBuild: Option.none(),
      expectedTarget: Option.none(),
      requiredCapabilities: ["model_load_control"],
      probeTimeout: Duration.seconds(2),
    }),
    storage: new IcnStorageConfig({
      modelStore: Option.some("/data/models"),
      cacheRoot: Option.some("/data/cache"),
      modelSources: ["/read-only/models"],
      huggingFaceCaches: ["/read-only/hf"],
    }),
    host,
    startupTimeout: Duration.seconds(30),
    gracefulShutdownTimeout: Duration.seconds(5),
    forceShutdownTimeout: Duration.seconds(2),
    outputLimitBytes: 64 * 1024,
  });

describe("ICN managed launch", () => {
  it("uses semantic tagged errors with derived messages and owned output", () => {
    const error = new IcnExitedBeforeReady({
      pid: 42,
      code: 17,
      output: "native startup failure",
    });

    expect(error._tag).toBe("IcnExitedBeforeReady");
    expect(error.message).toBe("Inference server process 42 exited with code 17 before readiness");
    expect(error.output).toBe("native startup failure");
    expect(error).not.toHaveProperty("operation");
    expect(error).not.toHaveProperty("reason");
    expect(error).not.toHaveProperty("diagnostic");
  });

  it("renders a model-free, owner-bound, port-zero command", () => {
    const args = renderIcnArguments(
      config(),
      "instance-1",
      "/opt/magnitude/installation.json",
    );
    expect(args).toEqual([
      "serve",
      "--bind",
      "127.0.0.1:0",
      "--instance-id",
      "instance-1",
      "--exit-on-stdin-eof",
      "--installation",
      "/opt/magnitude/installation.json",
      "--model-store",
      "/data/models",
      "--cache-root",
      "/data/cache",
      "--model-source",
      "/read-only/models",
      "--hf-cache",
      "/read-only/hf",
    ]);
    expect(args).not.toContain("--model");
    expect(args).not.toContain("--context-size");
  });

  it("brackets IPv6 loopback while retaining port zero", () => {
    expect(
      renderIcnArguments(
        config("::1"),
        "instance-2",
        "/opt/magnitude/installation.json",
      ).slice(0, 3)
    ).toEqual(["serve", "--bind", "[::1]:0"]);
  });

  it.runIf(process.platform !== "win32")(
    "resolves and verifies an explicit binary before publication",
    async () => {
      const directory = await mkdtemp(join(tmpdir(), "icn-resolver-test-"));
      const executable = join(directory, "bin", "magnitude-icn");
      const installation = join(directory, "installation.json");
      await mkdir(join(directory, "bin"), { recursive: true });
      await mkdir(join(directory, "runtime"), { recursive: true });
      await writeFile(installation, "{}");
      await writeFile(
        executable,
        `#!/bin/sh\nprintf '%s\\n' '{"version":"1.0.0","api_version":1,"native_build":"native-test","backend_module_abi":"abi-test","target":"test-target","profile":"test","rustc":"rustc test","capabilities":["model_load_control"],"backends":["cpu"]}'\n`
      );
      await chmod(executable, 0o755);
      try {
        const resolution = await Effect.runPromise(
          Effect.gen(function* () {
            const resolver = yield* IcnBinaryResolver;
            return yield* resolver.resolve(
              new IcnBinaryResolutionConfig({
                source: {
                  _tag: "Installation",
                  path: installation,
                },
                supportedApiVersion: 1,
                expectedNativeBuild: Option.some("native-test"),
                expectedTarget: Option.some("test-target"),
                requiredCapabilities: ["model_load_control"],
                probeTimeout: Duration.seconds(2),
              })
            );
          }).pipe(
            Effect.provide(
              makeIcnBinaryResolver().pipe(
                Layer.provide(
                  Layer.succeed(IcnPreparationReporter, {
                    report: () => Effect.void,
                  }),
                ),
                Layer.provideMerge(
                  Layer.merge(BunContext.layer, FetchHttpClient.layer)
                )
              )
            )
          )
        );
        expect(resolution.path).toBe(await realpath(executable));
        expect(resolution.installation).toBe(installation);
        expect(resolution.identity.native_build).toBe("native-test");
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  );

  it("reports an absent resolved binary as its exact domain variant", async () => {
    const missing = "/definitely-missing/magnitude/installation.json";
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const resolver = yield* IcnBinaryResolver;
        return yield* Effect.either(resolver.resolve(
          new IcnBinaryResolutionConfig({
            source: { _tag: "Installation", path: missing },
            supportedApiVersion: 1,
            expectedNativeBuild: Option.none(),
            expectedTarget: Option.none(),
            requiredCapabilities: [],
            probeTimeout: Duration.seconds(2),
          }),
        ));
      }).pipe(
        Effect.provide(
          makeIcnBinaryResolver().pipe(
            Layer.provide(Layer.succeed(IcnPreparationReporter, {
              report: () => Effect.void,
            })),
            Layer.provideMerge(Layer.merge(BunContext.layer, FetchHttpClient.layer)),
          ),
        ),
      ),
    );

    expect(result).toMatchObject({
      _tag: "Left",
      left: {
        _tag: "IcnBinaryNotFound",
        path: join(
          "/definitely-missing/magnitude",
          "bin",
          `magnitude-icn${process.platform === "win32" ? ".exe" : ""}`,
        ),
      },
    });
    if (result._tag === "Left") expect(result.left).toBeInstanceOf(IcnBinaryNotFound);
  });

  it("propagates filesystem failures while checking binary existence", async () => {
    const failure = new SystemError({
      reason: "PermissionDenied",
      module: "FileSystem",
      method: "exists",
      pathOrDescriptor: "/restricted/magnitude/installation.json",
    });
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const baseFs = yield* FileSystem.FileSystem;
        const failingFs: FileSystem.FileSystem = {
          ...baseFs,
          exists: () => Effect.fail(failure),
        };
        return yield* Effect.gen(function* () {
          const resolver = yield* IcnBinaryResolver;
          return yield* Effect.either(resolver.resolve(
            new IcnBinaryResolutionConfig({
              source: { _tag: "Installation", path: failure.pathOrDescriptor as string },
              supportedApiVersion: 1,
              expectedNativeBuild: Option.none(),
              expectedTarget: Option.none(),
              requiredCapabilities: [],
              probeTimeout: Duration.seconds(2),
            }),
          ));
        }).pipe(
          Effect.provide(
            makeIcnBinaryResolver().pipe(
              Layer.provide(Layer.succeed(FileSystem.FileSystem, failingFs)),
              Layer.provide(Layer.succeed(IcnPreparationReporter, {
                report: () => Effect.void,
              })),
              Layer.provideMerge(Layer.merge(BunContext.layer, FetchHttpClient.layer)),
            ),
          ),
        );
      }).pipe(Effect.provide(BunContext.layer)),
    );

    expect(result._tag).toBe("Left");
    if (result._tag === "Left") expect(result.left).toBe(failure);
  });

});
