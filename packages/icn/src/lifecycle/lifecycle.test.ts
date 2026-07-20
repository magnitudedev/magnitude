import { Duration, Effect, Layer, Option } from "effect";
import * as BunContext from "@effect/platform-bun/BunContext";
import { chmod, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  IcnBinaryResolutionConfig,
  IcnBinaryResolver,
  IcnBinaryResolverLive,
  IcnLifecycleConfig,
  IcnStorageConfig,
  renderIcnArguments,
} from "./index.js";

const config = (host: "127.0.0.1" | "::1" = "127.0.0.1") =>
  new IcnLifecycleConfig({
    binary: new IcnBinaryResolutionConfig({
      source: { _tag: "Explicit", path: "/opt/magnitude/magnitude-icn" },
      supportedApiVersion: 1,
      expectedNativeBuild: Option.none(),
      expectedTarget: Option.none(),
      requiredCapabilities: ["runtime_model_control"],
      allowBuildMismatch: false,
      probeTimeout: Duration.seconds(2),
    }),
    storage: new IcnStorageConfig({
      modelStore: Option.some("/data/models"),
      legacyStore: Option.none(),
      modelSources: ["/read-only/models"],
      huggingFaceCaches: ["/read-only/hf"],
    }),
    host,
    parentPid: 42,
    startupTimeout: Duration.seconds(30),
    gracefulShutdownTimeout: Duration.seconds(5),
    forceShutdownTimeout: Duration.seconds(2),
    outputLimitBytes: 64 * 1024,
  });

describe("ICN managed launch", () => {
  it("renders a model-free, owner-bound, port-zero command", () => {
    const args = renderIcnArguments(config(), "instance-1", 42);
    expect(args).toEqual([
      "serve",
      "--bind",
      "127.0.0.1:0",
      "--instance-id",
      "instance-1",
      "--parent-pid",
      "42",
      "--model-store",
      "/data/models",
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
      renderIcnArguments(config("::1"), "instance-2", 42).slice(0, 3)
    ).toEqual(["serve", "--bind", "[::1]:0"]);
  });

  it.runIf(process.platform !== "win32")(
    "resolves and verifies an explicit binary before publication",
    async () => {
      const directory = await mkdtemp(join(tmpdir(), "icn-resolver-test-"));
      const executable = join(directory, "magnitude-icn");
      await writeFile(
        executable,
        `#!/bin/sh\nprintf '%s\\n' '{"version":"1.0.0","api_version":1,"native_build":"native-test","target":"test-target","capabilities":["runtime_model_control"]}'\n`
      );
      await chmod(executable, 0o755);
      try {
        const resolution = await Effect.runPromise(
          Effect.gen(function* () {
            const resolver = yield* IcnBinaryResolver;
            return yield* resolver.resolve(
              new IcnBinaryResolutionConfig({
                source: { _tag: "Explicit", path: executable },
                supportedApiVersion: 1,
                expectedNativeBuild: Option.some("native-test"),
                expectedTarget: Option.some("test-target"),
                requiredCapabilities: ["runtime_model_control"],
                allowBuildMismatch: false,
                probeTimeout: Duration.seconds(2),
              })
            );
          }).pipe(
            Effect.provide(
              IcnBinaryResolverLive.pipe(Layer.provideMerge(BunContext.layer))
            )
          )
        );
        expect(resolution.path).toBe(await realpath(executable));
        expect(resolution.identity.native_build).toBe("native-test");
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  );
});
