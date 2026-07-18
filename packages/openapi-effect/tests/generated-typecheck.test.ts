import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { Effect, HashMap } from "effect";
import { afterEach, describe, expect, it } from "vitest";
import { compileOpenApi } from "../src/index.js";
import { sseDocument, streamConfig } from "./fixtures.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true }))
  );
});

describe("generated TypeScript", () => {
  it("type-checks against Magnitude's installed Effect 3 platform", async () => {
    const recursiveDocument = {
      ...sseDocument,
      components: {
        schemas: {
          ...sseDocument.components.schemas,
          Node: {
            type: "object",
            required: ["value"],
            properties: {
              value: { type: "string" },
              next: { $ref: "#/components/schemas/Node" },
            },
            additionalProperties: false,
          },
        },
      },
    } as const;
    const generated = await Effect.runPromise(
      compileOpenApi(recursiveDocument, streamConfig)
    );
    const directory = await mkdtemp(
      resolve(processCwd(), ".generated-typecheck-")
    );
    temporaryDirectories.push(directory);

    for (const [path, source] of HashMap.entries(generated.files)) {
      if (path.endsWith(".ts"))
        await writeFile(resolve(directory, path), source);
    }
    await writeFile(
      resolve(directory, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          target: "ESNext",
          module: "ESNext",
          moduleResolution: "bundler",
          strict: true,
          skipLibCheck: true,
          noEmit: true,
        },
        include: ["*.ts"],
      })
    );

    const subprocess = Bun.spawn(
      ["bunx", "tsc", "--noEmit", "-p", resolve(directory, "tsconfig.json")],
      {
        cwd: processCwd(),
        stdout: "pipe",
        stderr: "pipe",
      }
    );
    const [exitCode, stdout, stderr] = await Promise.all([
      subprocess.exited,
      new Response(subprocess.stdout).text(),
      new Response(subprocess.stderr).text(),
    ]);
    expect(`${stdout}\n${stderr}`, `${stdout}\n${stderr}`).toBe("\n");
    expect(exitCode).toBe(0);
  });
});

const processCwd = (): string => resolve(import.meta.dirname, "..");
