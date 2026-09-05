import { Effect, Ref } from "effect";
import { gzipSync } from "node:zlib";
import { pack } from "tar-stream";
import { describe, expect, it } from "vitest";
import {
  awaitIntegrity,
  pluginTarballFilename,
  readTarballEntry,
} from "./plugin-artifacts";

describe("registry read-back", () => {
  const policy = { attempts: 3, interval: "1 millis" } as const;
  const reader = (answers: ReadonlyArray<string | null | Error>) =>
    Effect.map(Ref.make(0), (index) =>
      Effect.flatMap(Ref.getAndUpdate(index, (i) => i + 1), (i) => {
        const answer = answers[Math.min(i, answers.length - 1)];
        return answer instanceof Error
          ? Effect.fail(answer)
          : Effect.succeed(answer ?? null);
      })
    );
  it("tolerates the registry lagging behind a fresh publish", async () => {
    const read = await Effect.runPromise(reader([null, new Error("E503"), "sha512-ok"]));
    await Effect.runPromise(awaitIntegrity(read, "sha512-ok", policy));
  });
  it("treats a different integrity as final and names both values", async () => {
    const read = await Effect.runPromise(reader(["sha512-other"]));
    const result = await Effect.runPromise(
      Effect.either(awaitIntegrity(read, "sha512-ok", policy))
    );
    expect(result._tag).toBe("Left");
    if (result._tag === "Left") {
      expect(result.left.message).toContain("sha512-other");
      expect(result.left.message).toContain("sha512-ok");
    }
  });
  it("gives up after the policy's attempts, reporting the last observation", async () => {
    const read = await Effect.runPromise(reader([null]));
    const result = await Effect.runPromise(
      Effect.either(awaitIntegrity(read, "sha512-ok", policy))
    );
    expect(result._tag).toBe("Left");
    if (result._tag === "Left") expect(result.left.message).toContain("3 attempts");
  });
});

const tarball = async (entries: Record<string, string>): Promise<Uint8Array> => {
  const archive = pack();
  for (const [name, contents] of Object.entries(entries)) archive.entry({ name }, contents);
  archive.finalize();
  const chunks: Buffer[] = [];
  for await (const chunk of archive) chunks.push(chunk as Buffer);
  return new Uint8Array(gzipSync(Buffer.concat(chunks)));
};

describe("published plugin tarballs", () => {
  it("names tarballs the way npm pack does for scoped packages", () => {
    expect(pluginTarballFilename("@magnitudedev/pi-extension", "0.0.1")).toBe(
      "magnitudedev-pi-extension-0.0.1.tgz"
    );
  });
  it("reads one entry out of a gzipped tarball", async () => {
    const bytes = await tarball({
      "package/package.json": "{}",
      "package/dist/magnitude-plugin.json": '{"rpcVersion":1}',
    });
    const entry = await Effect.runPromise(
      readTarballEntry(bytes, "package/dist/magnitude-plugin.json")
    );
    expect(new TextDecoder().decode(entry)).toBe('{"rpcVersion":1}');
    const missing = await Effect.runPromise(
      Effect.either(readTarballEntry(bytes, "package/README.md"))
    );
    expect(missing._tag).toBe("Left");
  });
});
