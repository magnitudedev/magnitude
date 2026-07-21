import { Duration, Effect, Layer, Option } from "effect";
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
  IcnBinaryResolverLive,
  IcnLifecycleConfig,
  IcnStorageConfig,
  icnReleaseAssetName,
  icnReleaseDownloadUrl,
  renderIcnArguments,
} from "./index.js";

const config = (host: "127.0.0.1" | "::1" = "127.0.0.1") =>
  new IcnLifecycleConfig({
    binary: new IcnBinaryResolutionConfig({
      source: { _tag: "Explicit", path: "/opt/magnitude/magnitude-icn" },
      supportedApiVersion: 1,
      expectedNativeBuild: Option.none(),
      expectedTarget: Option.none(),
      requiredCapabilities: ["model_load_control"],
      allowBuildMismatch: false,
      probeTimeout: Duration.seconds(2),
      downloadTimeout: Duration.seconds(30),
    }),
    storage: new IcnStorageConfig({
      modelStore: Option.some("/data/models"),
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
  it("derives the versioned platform release asset deterministically", () => {
    expect(icnReleaseAssetName("darwin-arm64")).toBe(
      "magnitude-icn-darwin-arm64.tar.gz"
    );
    expect(
      icnReleaseDownloadUrl(
        "https://github.com/magnitudedev/magnitude/releases/download/",
        "1.2.3",
        "darwin-arm64"
      )
    ).toBe(
      "https://github.com/magnitudedev/magnitude/releases/download/%40magnitudedev%2Fcli%401.2.3/magnitude-icn-darwin-arm64.tar.gz"
    );
  });

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
        `#!/bin/sh\nprintf '%s\\n' '{"version":"1.0.0","api_version":1,"native_build":"native-test","target":"test-target","capabilities":["model_load_control"]}'\n`
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
                requiredCapabilities: ["model_load_control"],
                allowBuildMismatch: false,
                probeTimeout: Duration.seconds(2),
                downloadTimeout: Duration.seconds(30),
              })
            );
          }).pipe(
            Effect.provide(
              IcnBinaryResolverLive.pipe(
                Layer.provideMerge(
                  Layer.merge(BunContext.layer, FetchHttpClient.layer)
                )
              )
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

  it.runIf(process.platform !== "win32")(
    "downloads, verifies, publishes, and reuses a release ICN",
    async () => {
      const directory = await mkdtemp(join(tmpdir(), "icn-release-test-"));
      const payload = join(directory, "payload");
      const dataDir = join(directory, "data");
      const executable = join(payload, "magnitude-icn");
      const archive = join(directory, "magnitude-icn-test-target.tar.gz");
      await mkdir(payload, { recursive: true });
      await writeFile(
        executable,
        `#!/bin/sh\nprintf '%s\\n' '{"version":"1.0.0","api_version":1,"native_build":"native-test","target":"test-target","capabilities":["model_load_control"]}'\n`
      );
      await chmod(executable, 0o755);
      const bytes = await Bun.file(executable).arrayBuffer();
      const sha256 = Buffer.from(
        await crypto.subtle.digest("SHA-256", bytes)
      ).toString("hex");
      await writeFile(
        join(payload, "magnitude-icn-manifest.json"),
        JSON.stringify({
          schemaVersion: 1,
          binary: "magnitude-icn",
          sha256,
          apiVersion: 1,
          nativeBuild: "native-test",
          target: "test-target",
        })
      );
      const tar = Bun.spawn([
        "tar",
        "-czf",
        archive,
        "-C",
        payload,
        "magnitude-icn",
        "magnitude-icn-manifest.json",
      ]);
      expect(await tar.exited).toBe(0);
      const server = Bun.serve({
        port: 0,
        fetch: () => new Response(Bun.file(archive)),
      });
      try {
        const releaseConfig = new IcnBinaryResolutionConfig({
          source: {
            _tag: "Release",
            version: "1.0.0",
            platformKey: "test-target",
            dataDir,
            releaseBaseUrl: `http://127.0.0.1:${server.port}`,
          },
          supportedApiVersion: 1,
          expectedNativeBuild: Option.some("native-test"),
          expectedTarget: Option.some("test-target"),
          requiredCapabilities: ["model_load_control"],
          allowBuildMismatch: false,
          probeTimeout: Duration.seconds(2),
          downloadTimeout: Duration.seconds(5),
        });
        const resolveRelease = Effect.gen(function* () {
          const resolver = yield* IcnBinaryResolver;
          return yield* resolver.resolve(releaseConfig);
        }).pipe(
          Effect.provide(
            IcnBinaryResolverLive.pipe(
              Layer.provideMerge(
                Layer.merge(BunContext.layer, FetchHttpClient.layer)
              )
            )
          )
        );
        const first = await Effect.runPromise(resolveRelease);
        expect(first.path).toBe(
          await realpath(
            join(dataDir, "bin", "icn", "1.0.0-test-target", "magnitude-icn")
          )
        );
        server.stop(true);
        const cached = await Effect.runPromise(resolveRelease);
        expect(cached.path).toBe(first.path);
      } finally {
        server.stop(true);
        await rm(directory, { recursive: true, force: true });
      }
    }
  );
});
