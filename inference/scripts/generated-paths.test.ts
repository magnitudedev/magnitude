import { describe, expect, test } from "vitest";
import { resolve } from "node:path";

describe("generated parity path policy", () => {
  test("workspace ignore rules cover every default parity output/cache root", async () => {
    const repositoryRoot = resolve(import.meta.dirname, "../..");
    const lines = (await Bun.file(resolve(repositoryRoot, ".gitignore")).text())
      .split("\n")
      .map((line) => line.trim());

    expect(lines).toContain("inference/target/");
    expect(lines).toContain("inference/**/target/");
    expect(lines).toContain("inference/.parity-models/");
    expect(lines).toContain("inference/results/parity/");
    expect(lines).toContain("inference/dist/");
  });

  test("the results directory admits only policy documentation", async () => {
    const ignore = await Bun.file(
      resolve(import.meta.dirname, "../results/.gitignore")
    ).text();
    expect(ignore.split("\n")).toEqual(
      expect.arrayContaining(["*", "!.gitignore", "!README.md"])
    );
  });
});
