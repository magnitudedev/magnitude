import { afterEach, describe, expect, it } from "vitest";
import { FetchHttpClient } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect, Option, Stream } from "effect";
import {
  chmod,
  mkdir,
  mkdtemp,
  stat,
  utimes,
  writeFile,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { makeLocalDaemonSpawner } from "./local-daemon-spawner";
import { SpawnProcess, type SpawnedProcess } from "./local-daemon-spawner";
import { runDaemonSpawn, type DaemonSpawner } from "./daemon-spawner";

const spawn = (
  spawner: DaemonSpawner,
  command: Option.Option<ReadonlyArray<string>>
) => runDaemonSpawn(spawner.spawn(command));

const testProcess = (
  exited: Promise<number | null> = new Promise(() => {}),
  pid = 9999
): SpawnedProcess => ({
  pid: Option.some(pid),
  exited: Effect.promise(() => exited).pipe(Effect.map((code) => code ?? 1)),
  diagnostic: Effect.succeed(Option.none()),
  kill: () => Effect.void,
});

const testSpawner = (
  spawnProcess: (command: ReadonlyArray<string>) => {
    readonly pid: number;
    readonly exited: Promise<number | null>;
  }
): SpawnProcess => ({
  spawn: (command) =>
    Effect.sync(() => {
      const process = spawnProcess(command);
      return testProcess(process.exited, process.pid);
    }),
});

describe("local daemon spawner rendezvous", () => {
  let server: ReturnType<typeof Bun.serve> | undefined;

  afterEach(() => {
    server?.stop(true);
    server = undefined;
  });

  it.runIf(process.platform !== "win32")(
    "threads the elected data root into the resolved ACN command",
    async () => {
      const dataDir = await mkdtemp(
        join(tmpdir(), "magnitude-spawner-data-root-")
      );
      const version = "1.2.3";
      const id = "data-root-owner";
      const pid = 6543;
      const binary = join(dataDir, "fake-acn");
      await writeFile(binary, `#!/bin/sh\nprintf '%s\\n' '${version}'\n`);
      await chmod(binary, 0o755);
      server = Bun.serve({
        port: 0,
        fetch: () =>
          Response.json({
            service: "magnitude-acn",
            version,
            id,
            pid,
            state: { _tag: "Ready" },
          }),
      });
      const registryDirectory = join(dataDir, "acn");
      const registryPath = join(registryDirectory, "registry.json");
      await mkdir(registryDirectory, { recursive: true });
      let launched: readonly string[] | null = null;

      const spawnProcess = (command: readonly string[]) => {
        launched = command;
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
        return { pid: 9999, exited: new Promise<number | null>(() => {}) };
      };

      const url = await makeLocalDaemonSpawner({
        binaryPath: binary,
        dataDir,
        version,
        publicationTimeoutMs: 2_000,
        probeTimeoutMs: 200,
      }).pipe(
        Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
        Effect.flatMap((spawner) => spawn(spawner, Option.none())),
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
        Effect.runPromise
      );

      expect(url).toBe(`http://127.0.0.1:${server.port}`);
      expect(launched).toEqual([
        binary,
        "serve",
        "--register",
        "--data-dir",
        dataDir,
      ]);
    }
  );

  it("fails immediately when its exact spawned candidate exits", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-"));
    const candidateVersion = "1.0.0";
    const registryDirectory = join(dataDir, "acn");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => ({
      pid: 9999,
      exited: Promise.resolve(1),
    });

    const attempt = makeLocalDaemonSpawner({
      dataDir,
      version: candidateVersion,
      publicationTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow('"exitCode": 1');
    expect(
      await Bun.file(join(registryDirectory, "spawn-election")).exists()
    ).toBe(false);
  });

  it("includes the bounded candidate diagnostic in an exact-exit failure", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-diagnostic-")
    );
    const processSpawner: SpawnProcess = {
      spawn: () =>
        Effect.succeed({
          pid: Option.some(9999),
          exited: Effect.succeed(17),
          diagnostic: Effect.succeed(
            Option.some("local inference contract rejected")
          ),
          kill: () => Effect.void,
        }),
    };

    const attempt = makeLocalDaemonSpawner({
      dataDir,
      version: "1.0.0",
      publicationTimeoutMs: 2_000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(SpawnProcess, processSpawner),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow("local inference contract rejected");
  });

  it("single-flights spawn across independent local spawners", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-election-")
    );
    const version = "test-election-version";
    const id = "election-winner";
    const pid = 7654;
    let spawnCalls = 0;
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
          state: { _tag: "Ready" },
        }),
    });
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => {
      spawnCalls++;
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
      return { pid: 9999, exited: new Promise<number | null>(() => {}) };
    };

    const makeSpawner = () =>
      makeLocalDaemonSpawner({
        dataDir,
        version,
        publicationTimeoutMs: 2000,
        probeTimeoutMs: 200,
      }).pipe(
        Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
        Effect.provide([BunContext.layer, FetchHttpClient.layer])
      );
    const [first, second] = await Effect.all([
      makeSpawner(),
      makeSpawner(),
    ]).pipe(
      Effect.flatMap(([left, right]) =>
        Effect.all(
          [
            spawn(left, Option.some(["ignored"])),
            spawn(right, Option.some(["ignored"])),
          ],
          { concurrency: "unbounded" }
        )
      ),
      Effect.runPromise
    );

    expect(first).toBe(`http://127.0.0.1:${server.port}`);
    expect(second).toBe(first);
    expect(spawnCalls).toBe(1);
    expect(
      await Bun.file(join(registryDirectory, "spawn-election")).exists()
    ).toBe(false);
  });

  it("joins a successor waiting for active ownership without spawning another", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-ownership-wait-")
    );
    const version = "1.0.0";
    const id = "waiting-successor";
    const pid = 7655;
    let ready = false;
    let spawnCalls = 0;
    const reports: string[] = [];
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
          state: ready
            ? { _tag: "Ready" }
            : {
                _tag: "Starting",
                activity: "WaitingForOwnership",
              },
        }),
    });
    const registryDirectory = join(dataDir, "acn");
    await mkdir(registryDirectory, { recursive: true });
    await writeFile(
      join(registryDirectory, "registry.json"),
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          id,
          version,
          url: `http://127.0.0.1:${server.port}`,
          pid,
          timestamp: Date.now(),
        },
      })
    );
    setTimeout(() => {
      ready = true;
    }, 200);

    const attempt = makeLocalDaemonSpawner({
      dataDir,
      version,
      publicationTimeoutMs: 2_000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(
        SpawnProcess,
        testSpawner(() => {
          spawnCalls++;
          return { pid: 9999, exited: new Promise<number | null>(() => {}) };
        })
      ),
      Effect.flatMap((spawner) =>
        runDaemonSpawn(
          spawner.spawn(Option.some(["ignored"])).pipe(
            Stream.tap((event) =>
              event._tag === "Observation"
                ? Effect.sync(() => {
                    if (event.observation._tag === "Starting") {
                      reports.push(event.observation.phase);
                    }
                  })
                : Effect.void
            )
          )
        )
      ),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );
    await Bun.sleep(50);
    expect(
      await Bun.file(join(registryDirectory, "spawn-election")).exists()
    ).toBe(false);
    const url = await attempt;

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(spawnCalls).toBe(0);
    expect(reports).toContain("WaitingForOwner");
  });

  it("joins an ACN installation and forwards its health progress", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-installation-wait-")
    );
    const version = "1.0.0";
    const id = "installing-successor";
    const pid = 7656;
    let ready = false;
    let spawnCalls = 0;
    const reports: Array<{
      readonly phase: string;
      readonly completed: number;
    }> = [];
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
          state: ready
            ? { _tag: "Ready" }
            : {
                _tag: "Installing",
                phase: "DownloadingInferenceEngine",
                plan: {
                  daemonBytes: 30,
                  inferenceEngineBytes: 70,
                  inferenceEngineBytesExact: true,
                },
                progress: {
                  completed: 40,
                  totalBytes: 100,
                  unit: "Bytes",
                },
              },
        }),
    });
    const registryDirectory = join(dataDir, "acn");
    await mkdir(registryDirectory, { recursive: true });
    await writeFile(
      join(registryDirectory, "registry.json"),
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          id,
          version,
          url: `http://127.0.0.1:${server.port}`,
          pid,
          timestamp: Date.now(),
        },
      })
    );
    setTimeout(() => {
      ready = true;
    }, 50);

    const url = await makeLocalDaemonSpawner({
      dataDir,
      version,
      publicationTimeoutMs: 2_000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(
        SpawnProcess,
        testSpawner(() => {
          spawnCalls++;
          return { pid: 9999, exited: new Promise<number | null>(() => {}) };
        })
      ),
      Effect.flatMap((spawner) =>
        runDaemonSpawn(
          spawner.spawn(Option.some(["ignored"])).pipe(
            Stream.tap((event) =>
              event._tag === "Observation"
                ? Effect.sync(() => {
                    const state = event.observation;
                    if (
                      state._tag === "Installing" &&
                      Option.isSome(state.progress)
                    ) {
                      reports.push({
                        phase: state.phase,
                        completed: state.progress.value.completed,
                      });
                    }
                  })
                : Effect.void
            )
          )
        )
      ),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(spawnCalls).toBe(0);
    expect(reports).toContainEqual({
      phase: "DownloadingInferenceEngine",
      completed: 40,
    });
  });

  it("keeps waiting without stealing a live process's election", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-foreign-election-")
    );
    const version = "test-foreign-election-version";
    const electionDirectory = join(dataDir, "acn", "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(
      electionDirectory,
      JSON.stringify({
        token: "foreign-owner",
        pid: process.pid,
      })
    );
    const staleTimestamp = new Date(Date.now() - 120_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    const spawnProcess = () => ({
      pid: 9999,
      exited: new Promise<number | null>(() => {}),
    });
    const attempt = makeLocalDaemonSpawner({
      dataDir,
      version,
      publicationTimeoutMs: 100,
      probeTimeoutMs: 20,
    }).pipe(
      Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.timeoutOption("100 millis"),
      Effect.runPromise
    );

    await expect(attempt).resolves.toEqual(Option.none());
    expect(JSON.parse(await Bun.file(electionDirectory).text())).toEqual({
      token: "foreign-owner",
      pid: process.pid,
    });
  });

  it("recovers a dead election and observes exact candidate exit", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-dead-election-")
    );
    const version = "test-dead-election-version";
    const electionDirectory = join(dataDir, "acn", "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(
      electionDirectory,
      JSON.stringify({
        token: "dead-owner",
        pid: 2_147_483_647,
      })
    );
    const staleTimestamp = new Date(Date.now() - 1_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    let spawnCalls = 0;
    const spawnProcess = () => {
      spawnCalls++;
      return { pid: 9999, exited: Promise.resolve(1) };
    };
    const attempt = makeLocalDaemonSpawner({
      dataDir,
      version,
      publicationTimeoutMs: 100,
      probeTimeoutMs: 20,
    }).pipe(
      Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow('"exitCode": 1');
    expect(spawnCalls).toBe(1);
    expect(await Bun.file(electionDirectory).exists()).toBe(false);
    expect((await stat(`${electionDirectory}.stale-dead-owner`)).isFile()).toBe(
      true
    );
  });

  it("starts the successor without directing the incompatible owner to shut down", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-upgrade-"));
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const old = Bun.spawn({
      cmd: [
        process.execPath,
        "-e",
        "process.on('SIGTERM',()=>setTimeout(()=>process.exit(0),50));setInterval(()=>{},1000)",
      ],
      stdout: "ignore",
      stderr: "ignore",
    });
    let health = { version: "1.0.0", id: "old-owner", pid: old.pid };
    let shutdownRequests = 0;
    server = Bun.serve({
      port: 0,
      fetch: (request) => {
        if (request.method === "POST") {
          shutdownRequests++;
          return new Response(null, { status: 405 });
        }
        return Response.json({
          service: "magnitude-acn",
          ...health,
          ...(health.version === "1.0.0" ? {} : { state: { _tag: "Ready" } }),
        });
      },
    });
    await writeFile(
      registryPath,
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          ...health,
          url: `http://127.0.0.1:${server.port}`,
          timestamp: Date.now(),
        },
      })
    );

    let oldWasGoneAtSpawn = false;
    const spawnProcess = () => {
      try {
        process.kill(old.pid, 0);
      } catch {
        oldWasGoneAtSpawn = true;
      }
      health = { version: "2.0.0", id: "new-owner", pid: 9876 };
      void writeFile(
        registryPath,
        JSON.stringify({
          schemaVersion: 1,
          registration: {
            ...health,
            url: `http://127.0.0.1:${server!.port}`,
            timestamp: Date.now(),
          },
        })
      );
      return { pid: health.pid, exited: new Promise<number | null>(() => {}) };
    };

    const url = await makeLocalDaemonSpawner({
      dataDir,
      version: "2.0.0",
      publicationTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(oldWasGoneAtSpawn).toBe(false);
    expect(shutdownRequests).toBe(0);
    process.kill(old.pid, "SIGTERM");
    expect([0, 143]).toContain(await old.exited);
  });

  it("lets an older client reuse a newer healthy owner without spawning", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-downgrade-")
    );
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });
    const incumbent = { version: "2.0.0", id: "newer-owner", pid: process.pid };
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          ...incumbent,
          state: { _tag: "Ready" },
        }),
    });
    await writeFile(
      registryPath,
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          ...incumbent,
          url: `http://127.0.0.1:${server.port}`,
          timestamp: Date.now(),
        },
      })
    );
    let spawned = false;

    const spawnProcess = () => {
      spawned = true;
      return { pid: 9999, exited: Promise.resolve(1) };
    };
    const url = await makeLocalDaemonSpawner({
      dataDir,
      version: "1.0.0",
      publicationTimeoutMs: 500,
      probeTimeoutMs: 100,
    }).pipe(
      Effect.provideService(SpawnProcess, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(spawned).toBe(false);
  });
});
