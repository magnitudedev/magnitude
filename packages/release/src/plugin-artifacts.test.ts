import { Effect } from "effect";
import { gzipSync } from "node:zlib";
import { pack } from "tar-stream";
import { describe, expect, it } from "vitest";
import { pluginTarballFilename, readTarballEntry } from "./plugin-artifacts";

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
