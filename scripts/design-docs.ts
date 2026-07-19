#!/usr/bin/env bun

import { statSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";

export interface DesignDocument {
  path: string;
  patterns: string[];
}

export interface DesignMatch {
  document: DesignDocument;
  matches: Array<{ target: string; pattern: string }>;
}

interface CliOptions {
  all: boolean;
  changed: boolean;
  explain: boolean;
  help: boolean;
  paths: string[];
}

const usage = `Usage: bun design-docs [options] [path ...]

Find design documents applicable to project files or directories.

Options:
  --changed   Match all staged, unstaged, and untracked Git changes
  --all       List every design document
  --explain   Show which paths and patterns caused each match
  -h, --help  Show this help

Examples:
  bun design-docs inference/crates/icn-engine/src/scheduler.rs
  bun design-docs inference/crates/icn-engine
  bun design-docs --changed
  bun design-docs --all`;

function decode(output: Uint8Array): string {
  return new TextDecoder().decode(output);
}

function runGit(root: string, args: string[]): string {
  const result = Bun.spawnSync(["git", ...args], {
    cwd: root,
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) {
    const detail = decode(result.stderr).trim();
    throw new Error(detail || `git ${args.join(" ")} failed`);
  }
  return decode(result.stdout);
}

function nulPaths(output: string): string[] {
  return output
    .split("\0")
    .filter(Boolean)
    .map((path) => path.replaceAll("\\", "/"));
}

export function findProjectRoot(cwd = process.cwd()): string {
  return runGit(cwd, ["rev-parse", "--show-toplevel"]).trim();
}

export function parseDesignDocument(path: string, source: string): DesignDocument {
  const frontMatter = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!frontMatter) {
    throw new Error(`${path}: missing YAML front matter`);
  }

  let parsed: unknown;
  try {
    parsed = Bun.YAML.parse(frontMatter[1]);
  } catch (error) {
    throw new Error(`${path}: invalid YAML front matter: ${String(error)}`);
  }

  const appliesTo =
    parsed && typeof parsed === "object" && "applies_to" in parsed
      ? (parsed as { applies_to?: unknown }).applies_to
      : undefined;
  if (!Array.isArray(appliesTo) || appliesTo.length === 0) {
    throw new Error(`${path}: applies_to must be a non-empty list`);
  }

  const patterns = appliesTo.map((pattern, index) => {
    if (typeof pattern !== "string" || pattern.length === 0) {
      throw new Error(`${path}: applies_to[${index}] must be a non-empty string`);
    }
    if (isAbsolute(pattern) || pattern.startsWith("../") || pattern.includes("\\")) {
      throw new Error(
        `${path}: applies_to[${index}] must be project-root-relative and use forward slashes`,
      );
    }
    try {
      new Bun.Glob(pattern);
    } catch (error) {
      throw new Error(`${path}: invalid applies_to glob ${JSON.stringify(pattern)}: ${String(error)}`);
    }
    return pattern;
  });

  return { path, patterns };
}

export async function loadDesignDocuments(root: string): Promise<DesignDocument[]> {
  const paths = Array.from(
    new Bun.Glob("design/**/*.md").scanSync({ cwd: root, onlyFiles: true }),
  )
    .map((path) => path.replaceAll("\\", "/"))
    .filter((path) => !path.endsWith("/AGENTS.md"))
    .sort();

  return Promise.all(
    paths.map(async (path) => parseDesignDocument(path, await Bun.file(resolve(root, path)).text())),
  );
}

export function normalizeProjectPath(root: string, input: string): string {
  const absolute = resolve(root, input);
  const projectRelative = relative(root, absolute);
  if (projectRelative === ".." || projectRelative.startsWith(`..${sep}`) || isAbsolute(projectRelative)) {
    throw new Error(`path is outside the project root: ${input}`);
  }
  return projectRelative.replaceAll(sep, "/") || ".";
}

function gitPaths(root: string, args: string[]): string[] {
  return nulPaths(runGit(root, args));
}

export function collectChangedPaths(root: string): string[] {
  return Array.from(
    new Set([
      ...gitPaths(root, ["diff", "--name-only", "-z", "--"]),
      ...gitPaths(root, ["diff", "--cached", "--name-only", "-z", "--"]),
      ...gitPaths(root, ["ls-files", "--others", "--exclude-standard", "-z", "--"]),
    ]),
  ).sort();
}

export function expandInputPaths(root: string, inputs: string[]): string[] {
  const targets = new Set<string>();
  for (const input of inputs) {
    const path = normalizeProjectPath(root, input);
    const absolute = resolve(root, path);
    let directory = false;
    try {
      directory = statSync(absolute).isDirectory();
    } catch {
      // A missing path may still describe a planned or deleted file and is matched directly.
    }

    if (!directory) {
      targets.add(path);
      continue;
    }

    targets.add(path === "." ? "." : `${path}/`);
    const pathspec = path === "." ? "." : path;
    for (const child of gitPaths(root, ["ls-files", "-co", "--exclude-standard", "-z", "--", pathspec])) {
      targets.add(child);
    }
  }
  return Array.from(targets).sort();
}

export function matchDesignDocuments(
  documents: DesignDocument[],
  targets: string[],
): DesignMatch[] {
  const targetSet = new Set(targets);
  return documents
    .map((document): DesignMatch => {
      const matches: DesignMatch["matches"] = [];
      if (targetSet.has(document.path)) {
        matches.push({ target: document.path, pattern: "(document itself)" });
      }
      for (const pattern of document.patterns) {
        const glob = new Bun.Glob(pattern);
        for (const target of targets) {
          if (glob.match(target)) {
            matches.push({ target, pattern });
          }
        }
      }
      return { document, matches };
    })
    .filter((match) => match.matches.length > 0)
    .sort((left, right) => left.document.path.localeCompare(right.document.path));
}

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    all: false,
    changed: false,
    explain: false,
    help: false,
    paths: [],
  };
  let positionalOnly = false;
  for (const arg of args) {
    if (positionalOnly) {
      options.paths.push(arg);
    } else if (arg === "--") {
      positionalOnly = true;
    } else if (arg === "--all") {
      options.all = true;
    } else if (arg === "--changed") {
      options.changed = true;
    } else if (arg === "--explain") {
      options.explain = true;
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else if (arg.startsWith("-")) {
      throw new Error(`unknown option: ${arg}`);
    } else {
      options.paths.push(arg);
    }
  }
  return options;
}

export async function main(args = process.argv.slice(2)): Promise<number> {
  let options: CliOptions;
  try {
    options = parseArgs(args);
  } catch (error) {
    console.error(String(error instanceof Error ? error.message : error));
    console.error(usage);
    return 2;
  }

  if (options.help) {
    console.log(usage);
    return 0;
  }
  const selectorCount = Number(options.all) + Number(options.changed) + Number(options.paths.length > 0);
  if (selectorCount !== 1) {
    console.error("provide paths, --changed, or --all (exactly one selection mode)");
    console.error(usage);
    return 2;
  }

  try {
    const root = findProjectRoot();
    const documents = await loadDesignDocuments(root);
    if (options.all) {
      for (const document of documents) {
        console.log(document.path);
        if (options.explain) {
          console.log("  selected by --all");
        }
      }
      return 0;
    }

    const targets = options.changed
      ? collectChangedPaths(root)
      : expandInputPaths(root, options.paths);
    for (const match of matchDesignDocuments(documents, targets)) {
      console.log(match.document.path);
      if (options.explain) {
        for (const reason of match.matches) {
          console.log(`  matched ${reason.target}`);
          console.log(`  via ${reason.pattern}`);
        }
      }
    }
    return 0;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 1;
  }
}

if (import.meta.main) {
  process.exitCode = await main();
}
