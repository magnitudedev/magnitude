import {
  appendFileSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

import type { GlobalStoragePaths } from "../paths/global-paths";
import type { StoredLogEntry } from "../types/log";

/**
 * Append entries to a JSONL file. This MUST remain a simple O(1) appendFileSync.
 *
 * Do NOT change this to read-rewrite, atomic write, temp-file+rename, or any
 * other pattern that touches the entire file. This function is called on every
 * log entry and every trace span from every worker. A read-rewrite approach
 * makes each call O(file_size), which causes catastrophic lag spikes that worsen
 * over the lifetime of a session — especially with multiple concurrent workers.
 *
 * JSONL reads fail on malformed lines; recovery belongs at a higher layer.
 */
export function appendJsonLinesSync<T>(
  filePath: string,
  entries: readonly T[]
): void {
  if (entries.length === 0) {
    return;
  }

  mkdirSync(dirname(filePath), { recursive: true });
  appendFileSync(
    filePath,
    entries.map((entry) => JSON.stringify(entry)).join("\n") + "\n",
    "utf-8"
  );
}

export function readJsonLinesSync<T>(filePath: string): T[] {
  let raw: string;
  try {
    raw = readFileSync(filePath, "utf-8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw error;
  }

  const result: T[] = [];

  const lines = raw.split("\n");
  const hasTerminatedTail = raw.endsWith("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "") {
      continue;
    }

    try {
      result.push(JSON.parse(line) as T);
    } catch (error) {
      if (i === lines.length - 1 && !hasTerminatedTail) break;
      throw error;
    }
  }

  return result;
}

export function clearFileSync(filePath: string): void {
  try {
    unlinkSync(filePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
  }
}

export function readJsonFileSync<T>(filePath: string, fallback: T): T {
  try {
    return JSON.parse(readFileSync(filePath, "utf-8")) as T;
  } catch {
    return fallback;
  }
}

export function writeTextFileSync(
  filePath: string,
  content: string,
  options?: { readonly mode?: number }
): void {
  mkdirSync(dirname(filePath), { recursive: true });

  const tempPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    writeFileSync(tempPath, content, {
      encoding: "utf-8",
      ...(options?.mode !== undefined ? { mode: options.mode } : {}),
    });
    renameSync(tempPath, filePath);
  } finally {
    try {
      unlinkSync(tempPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
}

export function writeJsonFileSync(
  filePath: string,
  value: unknown,
  options?: { readonly mode?: number }
): void {
  let content = JSON.stringify(value, null, 2);
  if (!content.endsWith("\n")) {
    content += "\n";
  }

  writeTextFileSync(filePath, content, options);
}

export function writeSecureJsonFileSync(
  filePath: string,
  value: unknown
): void {
  writeJsonFileSync(filePath, value, { mode: 0o600 });
}

// =============================================================================
// Session log sync helpers — used by logger
// =============================================================================

export function appendSessionLogsSync(
  paths: GlobalStoragePaths,
  sessionId: string,
  entries: readonly StoredLogEntry[]
): void {
  if (entries.length === 0) {
    return;
  }

  appendJsonLinesSync(paths.sessionLogFile(sessionId), entries);
}

export function clearSessionLogSync(
  paths: GlobalStoragePaths,
  sessionId: string
): void {
  clearFileSync(paths.sessionLogFile(sessionId));
}

export function getSessionLogPath(
  paths: GlobalStoragePaths,
  sessionId: string
): string {
  return paths.sessionLogFile(sessionId);
}
