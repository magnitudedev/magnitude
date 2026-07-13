import { afterEach, describe, expect, it } from "vitest";
import { FetchHttpClient } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect } from "effect";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { makeLocalDaemonSpawner } from "./local-daemon-spawner";

describe("local daemon spawner rendezvous", () => {
  let server: ReturnType<typeof Bun.serve> | undefined;

  afterEach(() => {
    server?.stop(true);
    server = undefined;
  });

  it("accepts another candidate after its own candidate exits and creates no lock", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-"));
    const version = "test-version";
    const id = "winning-owner";
    const pid = 4321;
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
        }),
    });
    const registryDirectory = join(dataDir, "acn", encodeURIComponent(version));
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => {
      setTimeout(() => {
        void writeFile(
          registryPath,
          JSON.stringify({
            schemaVersion: 1,
            registration: {
              id,
              version,
              url: `http://127.0.0.1:${server!.port}`,
              pid,
              timestamp: Date.now(),
            },
          })
        );
      }, 25);
      return { pid: 9999, exited: Promise.resolve(1) };
    };

    const url = await makeLocalDaemonSpawner(spawnProcess, {
      dataDir,
      version,
      spawnTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(
      await Bun.file(
        join(registryDirectory, "registry.lock", "owner.json")
      ).exists()
    ).toBe(false);
  });
});
