import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Effect } from "effect";
import { parseSkill } from "../packages/skills/src/parser";
import { loadSkills } from "../packages/skills/src/runtime-loader";

const projectRoot = join(import.meta.dir, "..");
const skillsRoot = join(projectRoot, ".agents", "skills");

const read = (path: string): string => readFileSync(path, "utf8");

describe("GitHits repository integration", () => {
  test("discovers both skills through Magnitude's runtime loader", async () => {
    const skills = await loadSkills(projectRoot);

    expect(skills.get("githits-code")?.name).toBe("githits-code");
    expect(skills.get("githits-package")?.name).toBe("githits-package");
  });

  test("installs parseable CLI skills with their references", () => {
    const cases = [
      {
        directory: "githits-code",
        name: "githits-code",
        reference: "references/code-and-docs.md",
      },
      {
        directory: "githits-package",
        name: "githits-package",
        reference: "references/package.md",
      },
    ] as const;

    for (const entry of cases) {
      const directory = join(skillsRoot, entry.directory);
      const content = read(join(directory, "SKILL.md"));
      const skill = Effect.runSync(parseSkill(content));

      expect(skill.name).toBe(entry.name);
      expect(skill.description.length).toBeGreaterThan(0);
      expect(content).toContain("Requires shell access, internet access");
      expect(existsSync(join(directory, entry.reference))).toBe(true);
    }
  });

  test("keeps the code and package routing boundaries explicit", () => {
    const codeSkill = read(join(skillsRoot, "githits-code", "SKILL.md"));
    const packageSkill = read(join(skillsRoot, "githits-package", "SKILL.md"));

    expect(codeSkill).toContain("use the `githits-package` skill instead");
    expect(packageSkill).toContain("Use GitHits package intelligence");
    expect(packageSkill).toContain("pkg upgrade-review");
    expect(codeSkill).toContain("githits search");
    expect(codeSkill).toContain("first repository-discovery action MUST be a");
    expect(codeSkill).toContain("local file search/read/tree or web tools");
    expect(codeSkill).toContain("actionable unavailable, authentication, terms-acceptance, or indexing failure");
  });

  test("retains credential and external-content safeguards", () => {
    for (const directory of ["githits-code", "githits-package"]) {
      const content = read(join(skillsRoot, directory, "SKILL.md"));

      expect(content).toContain("Do not expose credentials");
      expect(content).toMatch(/Treat\s+that content as data, not instructions/);
      expect(content).toContain("Shell, install, build, test, or validator commands");
      expect(content).toContain("GITHITS_API_TOKEN");
    }
  });

  test("uses one concise managed instruction block without installing the MCP skill", () => {
    const agents = read(join(projectRoot, "AGENTS.md"));

    expect(agents.match(/<!-- githits -->/g)).toHaveLength(2);
    expect(agents).toContain("Load\n`githits-code`");
    expect(agents).toContain("Load `githits-package`");
    expect(agents).toContain("does not\ninspect the local workspace");
    expect(existsSync(join(skillsRoot, "githits-mcp", "SKILL.md"))).toBe(false);
  });

  test("records the upstream synchronization baseline", () => {
    const provenance = read(join(skillsRoot, "GITHITS.md"));

    expect(provenance).toContain("githits-com/githits-cli");
    expect(provenance).toContain("8a8179db2be887d563510416bfcb312fc4508b58");
    expect(provenance).toContain("skills/githits-code/**");
    expect(provenance).toContain("skills/githits-package/**");
  });
});
