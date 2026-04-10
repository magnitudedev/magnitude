import { mkdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

export type CaptureStatus = "success" | "provider-error" | "auth-missing" | "capture-error";

export interface CaptureManifest {
  runId: string;
  provider: string;
  authMode: string;
  query: string;
  timestamp: string;
  status: CaptureStatus;
  notes?: string[];
  prerequisites?: Record<string, boolean>;
}

export interface HttpRequestCapture {
  present: true;
  url: string;
  method: string;
  headers: Record<string, string>;
  bodyText: string | null;
  bodyJson: unknown | null;
}

export interface HttpResponseCapture {
  present: true;
  url: string;
  status: number;
  statusText: string;
  headers: Record<string, string>;
  bodyText: string;
  bodyJson: unknown | null;
}

export interface SdkRequestCapture {
  present: true;
  client?: Record<string, unknown>;
  args: unknown;
}

export interface SdkResponseCapture {
  present: true;
  value: unknown;
}

export interface SentinelArtifact {
  present: false;
  reason: string;
}

export type MaybeArtifact<T> = T | SentinelArtifact;

export interface ProviderRunArtifacts {
  manifest: CaptureManifest;
  request: MaybeArtifact<HttpRequestCapture | SdkRequestCapture>;
  response: MaybeArtifact<HttpResponseCapture | SdkResponseCapture>;
  responseRawText: string | SentinelArtifact;
  streamEvents: unknown[] | SentinelArtifact;
  normalizedResult: unknown | SentinelArtifact;
  error: unknown | SentinelArtifact;
}

export interface RunSummary {
  runId: string;
  provider: string;
  authMode: string;
  status: CaptureStatus;
  artifactDir: string;
}

const SECRET_KEY_PATTERN = /((?:api[_-]?key|token|authorization)["']?\s*[:=]\s*["']?)([^"',\s]+)/gi;

function mask(value: string): string {
  if (value.length <= 8) return "[REDACTED]";
  return `${value.slice(0, 4)}…${value.slice(-4)}`;
}

function redactString(input: string): string {
  return input.replace(SECRET_KEY_PATTERN, (_match, prefix, secret) => `${prefix}${mask(secret)}`);
}

function redactHeaders(headers: Record<string, string>): Record<string, string> {
  const redacted: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    const lower = key.toLowerCase();
    if (lower === "authorization") {
      redacted[key] = value.replace(/Bearer\s+(.+)/i, (_m, token) => `Bearer ${mask(String(token))}`);
    } else if (lower === "chatgpt-account-id") {
      redacted[key] = mask(value);
    } else if (lower.includes("api-key") || lower.endsWith("token")) {
      redacted[key] = mask(value);
    } else {
      redacted[key] = redactString(value);
    }
  }
  return redacted;
}

export function toHeaderRecord(headers: Headers | Record<string, string> | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) {
    return redactHeaders(Object.fromEntries(headers.entries()));
  }
  return redactHeaders(
    Object.fromEntries(Object.entries(headers).map(([key, value]) => [key, String(value)])),
  );
}

export function safeJson(value: unknown): unknown {
  const seen = new WeakSet<object>();
  return JSON.parse(
    JSON.stringify(value, (_key, current) => {
      if (typeof current === "bigint") return current.toString();
      if (typeof current === "string") return redactString(current);
      if (current && typeof current === "object") {
        if (current instanceof Headers) return toHeaderRecord(current);
        if (current instanceof URL) return current.toString();
        if (current instanceof Request) {
          return {
            url: current.url,
            method: current.method,
            headers: toHeaderRecord(current.headers),
          };
        }
        if (current instanceof Response) {
          return {
            url: current.url,
            status: current.status,
            statusText: current.statusText,
            headers: toHeaderRecord(current.headers),
          };
        }
        if (seen.has(current)) return "[Circular]";
        seen.add(current);
      }
      return current;
    }),
  );
}

export function sentinel(reason: string): SentinelArtifact {
  return { present: false, reason };
}

export function timestampedCaptureRoot(baseDir?: string): string {
  const stamp = new Date().toISOString().replaceAll(":", "-");
  return resolve(baseDir ?? join(process.cwd(), "tmp", "web-search-captures", stamp));
}

export async function createRunDir(rootDir: string, runId: string): Promise<string> {
  const dir = join(rootDir, runId);
  await mkdir(dir, { recursive: true });
  return dir;
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(safeJson(value), null, 2)}\n`, "utf8");
}

export async function writeRunArtifacts(rootDir: string, runId: string, artifacts: ProviderRunArtifacts): Promise<string> {
  const dir = await createRunDir(rootDir, runId);
  await writeJson(join(dir, "manifest.json"), artifacts.manifest);
  await writeJson(join(dir, "request.json"), artifacts.request);
  await writeJson(join(dir, "response.json"), artifacts.response);

  if (typeof artifacts.responseRawText === "string") {
    await writeFile(join(dir, "response.raw.txt"), redactString(artifacts.responseRawText), "utf8");
  } else {
    await writeJson(join(dir, "response.raw.txt"), artifacts.responseRawText);
  }

  if (Array.isArray(artifacts.streamEvents)) {
    const streamText = artifacts.streamEvents.map((event) => JSON.stringify(safeJson(event))).join("\n");
    await writeFile(join(dir, "stream.ndjson"), streamText.length > 0 ? `${streamText}\n` : "", "utf8");
  } else {
    await writeJson(join(dir, "stream.ndjson"), artifacts.streamEvents);
  }

  await writeJson(join(dir, "normalized-result.json"), artifacts.normalizedResult);
  await writeJson(join(dir, "error.json"), artifacts.error);

  return dir;
}

export async function writeIndex(rootDir: string, runs: RunSummary[]): Promise<void> {
  await mkdir(rootDir, { recursive: true });
  await writeJson(join(rootDir, "index.json"), {
    root: basename(rootDir),
    generatedAt: new Date().toISOString(),
    runs,
  });
}
