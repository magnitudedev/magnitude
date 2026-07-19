import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  collectChangedPaths,
  expandInputPaths,
  matchDesignDocuments,
  normalizeProjectPath,
  parseDesignDocument,
  type DesignDocument,
} from "./design-docs";

describe("parseDesignDocument", () => {
  test("reads a list of project-relative globs", () => {
    expect(
      parseDesignDocument(
        "design/example.md",
        `---
applies_to:
  - packages/example/**
  - packages/shared/src/contract.ts
---
# Example
`,
      ),
    ).toEqual({
      path: "design/example.md",
      patterns: ["packages/example/**", "packages/shared/src/contract.ts"],
    });
  });

  test("requires non-empty applies_to front matter", () => {
    expect(() => parseDesignDocument("design/example.md", "# Example\n")).toThrow(
      "missing YAML front matter",
    );
    expect(() =>
      parseDesignDocument("design/example.md", "---\napplies_to: []\n---\n"),
    ).toThrow("applies_to must be a non-empty list");
  });

  test("rejects paths that are not project-root-relative", () => {
    expect(() =>
      parseDesignDocument(
        "design/example.md",
        "---\napplies_to:\n  - ../outside/**\n---\n",
      ),
    ).toThrow("must be project-root-relative");
  });
});

describe("matchDesignDocuments", () => {
  const documents: DesignDocument[] = [
    {
      path: "design/engine.md",
      patterns: ["inference/crates/icn-engine/**", "inference/shared.rs"],
    },
    {
      path: "design/api.md",
      patterns: ["inference/crates/icn-api/**"],
    },
  ];

  test("returns all overlapping documents once", () => {
    const result = matchDesignDocuments(documents, [
      "inference/crates/icn-engine/src/lib.rs",
      "inference/shared.rs",
    ]);
    expect(result.map((match) => match.document.path)).toEqual(["design/engine.md"]);
    expect(result[0]?.matches).toHaveLength(2);
  });

  test("treats a changed design document as pertinent to itself", () => {
    const result = matchDesignDocuments(documents, ["design/api.md"]);
    expect(result.map((match) => match.document.path)).toEqual(["design/api.md"]);
  });
});

describe("normalizeProjectPath", () => {
  test("normalizes paths under the project root", () => {
    expect(normalizeProjectPath("/repo", "/repo/packages/example.ts")).toBe(
      "packages/example.ts",
    );
  });

  test("rejects paths outside the project root", () => {
    expect(() => normalizeProjectPath("/repo", "/other/example.ts")).toThrow(
      "outside the project root",
    );
  });
});

describe("Git path collection", () => {
  test("includes staged, unstaged, and untracked files and expands directories", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-design-docs-"));
    try {
      Bun.spawnSync(["git", "init", "-q"], { cwd: root });
      mkdirSync(join(root, "src", "nested"), { recursive: true });
      writeFileSync(join(root, "src", "tracked.ts"), "initial\n");
      Bun.spawnSync(["git", "add", "src/tracked.ts"], { cwd: root });
      writeFileSync(join(root, "src", "tracked.ts"), "changed\n");
      writeFileSync(join(root, "src", "staged.ts"), "staged\n");
      Bun.spawnSync(["git", "add", "src/staged.ts"], { cwd: root });
      writeFileSync(join(root, "src", "nested", "untracked.ts"), "untracked\n");

      expect(collectChangedPaths(root)).toEqual([
        "src/nested/untracked.ts",
        "src/staged.ts",
        "src/tracked.ts",
      ]);
      expect(expandInputPaths(root, ["src"])).toEqual([
        "src/",
        "src/nested/untracked.ts",
        "src/staged.ts",
        "src/tracked.ts",
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
