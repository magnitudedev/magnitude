import { existsSync, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { Effect, Layer } from "effect";
import { ProviderAuth, ProviderState, type AuthInfo } from "@magnitudedev/providers";
import { webSearch, webSearchStream, type SearchAuth, type SearchOptions } from "../src/tools/web-search";
import { openaiWebSearch } from "../src/tools/web-search-openai";
import { openrouterWebSearch } from "../src/tools/web-search-openrouter";
import { vercelWebSearch } from "../src/tools/web-search-vercel";
import { copilotWebSearch } from "../src/tools/web-search-copilot";
import { runVercelAiSdkCapture } from "./web-search-capture/vercel-ai-sdk-capture";
import { geminiWebSearch } from "../src/tools/web-search-gemini";
import { anthropicWebSearch } from "../src/tools/web-search-anthropic";
import {
  sentinel,
  timestampedCaptureRoot,
  writeIndex,
  writeRunArtifacts,
  type CaptureManifest,
  type CaptureStatus,
  type ProviderRunArtifacts,
  type RunSummary,
} from "./web-search-capture/capture-harness";
import { withFetchInterceptor, type FetchCaptureState } from "./web-search-capture/interceptors/fetch-interceptor";
import { withOpenAISdkInterceptor, type OpenAISdkCaptureState } from "./web-search-capture/interceptors/openai-sdk-interceptor";
import { withGoogleSdkInterceptor, type GoogleSdkCaptureState } from "./web-search-capture/interceptors/google-sdk-interceptor";
import { withAnthropicSdkInterceptor, type AnthropicSdkCaptureState } from "./web-search-capture/interceptors/anthropic-sdk-interceptor";

type ProviderCli = "openai" | "openrouter" | "vercel" | "github-copilot" | "gemini" | "anthropic" | "all";
type OpenAIAuthMode = "api" | "oauth" | "both";

interface CliConfig {
  provider: ProviderCli;
  query: string;
  outDir?: string;
  openaiAuth: OpenAIAuthMode;
  streamAnthropic: boolean;
  directAdapter: boolean;
  system?: string;
  allowedDomains: string[];
  blockedDomains: string[];
}

interface RunSpec {
  runId: string;
  providerSlot: "openai" | "openrouter" | "vercel" | "github-copilot" | "google" | "anthropic";
  providerLabel: string;
  authMode: "api" | "oauth";
  authInfo?: AuthInfo;
  authError?: string;
}

function parseArgs(argv: string[]): CliConfig {
  const config: CliConfig = {
    provider: "all",
    query: "What are the top AI news headlines today? Include sources.",
    openaiAuth: "both",
    streamAnthropic: false,
    directAdapter: false,
    allowedDomains: [],
    blockedDomains: [],
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    const next = argv[index + 1];

    if (arg === "--provider" && next) {
      config.provider = next as ProviderCli;
      index += 1;
      continue;
    }
    if (arg === "--query" && next) {
      config.query = next;
      index += 1;
      continue;
    }
    if (arg === "--out" && next) {
      config.outDir = next;
      index += 1;
      continue;
    }
    if (arg === "--openai-auth" && next) {
      config.openaiAuth = next as OpenAIAuthMode;
      index += 1;
      continue;
    }

    if (arg === "--system" && next) {
      config.system = next;
      index += 1;
      continue;
    }
    if (arg === "--allowed-domain" && next) {
      config.allowedDomains.push(next);
      index += 1;
      continue;
    }
    if (arg === "--blocked-domain" && next) {
      config.blockedDomains.push(next);
      index += 1;
      continue;
    }
    if (arg === "--stream-anthropic") {
      config.streamAnthropic = true;
      continue;
    }
    if (arg === "--direct-adapter") {
      config.directAdapter = true;
      continue;
    }
    if (arg === "--help") {
      printHelp();
      process.exit(0);
    }
  }

  return config;
}

function printHelp(): void {
  console.log(`
Usage:
  bun run scripts/web-search-capture.ts [options]

Options:
  --provider <openai|openrouter|vercel|github-copilot|gemini|anthropic|all>
  --query "<text>"
  --out <dir>
  --openai-auth <api|oauth|both>
  --stream-anthropic
  --direct-adapter
  --system "<text>"
  --allowed-domain <domain>   (repeatable)
  --blocked-domain <domain>   (repeatable)

Credentials are read from ~/.magnitude/auth.json by default.
Environment variables still override when set.
`);
}

function getSearchOptions(config: CliConfig): SearchOptions | undefined {
  const options: SearchOptions = {};
  if (config.system) options.system = config.system;
  if (config.allowedDomains.length > 0) options.allowed_domains = config.allowedDomains;
  if (config.blockedDomains.length > 0) options.blocked_domains = config.blockedDomains;
  return Object.keys(options).length > 0 ? options : undefined;
}

function oauthAuth(accessToken: string, accountId?: string): AuthInfo {
  return {
    type: "oauth",
    accessToken,
    refreshToken: "capture-script",
    expiresAt: Date.now() + 60 * 60 * 1000,
    ...(accountId ? { accountId } : {}),
  };
}

function apiAuth(key: string): AuthInfo {
  return { type: "api", key };
}

function apiKeyFromAuthInfo(authInfo: AuthInfo | undefined): string | undefined {
  return authInfo?.type === "api" ? authInfo.key : undefined;
}

function oauthTokenFromAuthInfo(authInfo: AuthInfo | undefined): string | undefined {
  return authInfo?.type === "oauth" ? authInfo.accessToken : undefined;
}

function oauthAccountIdFromAuthInfo(authInfo: AuthInfo | undefined): string | undefined {
  return authInfo?.type === "oauth" ? authInfo.accountId : undefined;
}

function parseGlobalAuthFile(): Record<string, AuthInfo> {
  const home = process.env.HOME;
  if (!home) return {};
  const authPath = resolve(home, ".magnitude/auth.json");
  if (!existsSync(authPath)) return {};

  try {
    const raw = JSON.parse(readFileSync(authPath, "utf8"));
    if (!raw || typeof raw !== "object") return {};
    return raw as Record<string, AuthInfo>;
  } catch {
    return {};
  }
}

function buildRunSpecs(config: CliConfig): RunSpec[] {
  const specs: RunSpec[] = [];
  const include = (provider: Exclude<ProviderCli, "all">) => config.provider === "all" || config.provider === provider;
  const globalAuth = parseGlobalAuthFile();

  if (include("openai")) {
    const openaiAuthInfo = globalAuth.openai;
    const openaiApiKey = process.env.OPENAI_API_KEY ?? apiKeyFromAuthInfo(openaiAuthInfo);
    const openaiOauthToken = process.env.OPENAI_OAUTH_TOKEN ?? process.env.OPENAI_ACCESS_TOKEN ?? oauthTokenFromAuthInfo(openaiAuthInfo);
    const openaiAccountId = process.env.OPENAI_ACCOUNT_ID ?? oauthAccountIdFromAuthInfo(openaiAuthInfo);

    if (config.openaiAuth === "api" || config.openaiAuth === "both") {
      specs.push(openaiApiKey
        ? {
            runId: "openai-api",
            providerSlot: "openai",
            providerLabel: "openai",
            authMode: "api",
            authInfo: apiAuth(openaiApiKey),
          }
        : {
            runId: "openai-api",
            providerSlot: "openai",
            providerLabel: "openai",
            authMode: "api",
            authError: "Missing OpenAI API auth (OPENAI_API_KEY or ~/.magnitude/auth.json:openai)",
          });
    }

    if (config.openaiAuth === "oauth" || config.openaiAuth === "both") {
      specs.push(openaiOauthToken
        ? {
            runId: "openai-oauth",
            providerSlot: "openai",
            providerLabel: "openai",
            authMode: "oauth",
            authInfo: oauthAuth(openaiOauthToken, openaiAccountId),
          }
        : {
            runId: "openai-oauth",
            providerSlot: "openai",
            providerLabel: "openai",
            authMode: "oauth",
            authError: "Missing OpenAI OAuth auth (OPENAI_OAUTH_TOKEN/OPENAI_ACCESS_TOKEN or ~/.magnitude/auth.json:openai)",
          });
    }
  }

  if (include("openrouter")) {
    const key = process.env.OPENROUTER_API_KEY ?? apiKeyFromAuthInfo(globalAuth.openrouter);
    specs.push(key
      ? { runId: "openrouter", providerSlot: "openrouter", providerLabel: "openrouter", authMode: "api", authInfo: apiAuth(key) }
      : { runId: "openrouter", providerSlot: "openrouter", providerLabel: "openrouter", authMode: "api", authError: "Missing OpenRouter auth (OPENROUTER_API_KEY or ~/.magnitude/auth.json:openrouter)" });
  }

  if (include("vercel")) {
    const key = process.env.AI_GATEWAY_API_KEY ?? process.env.VERCEL_API_KEY ?? apiKeyFromAuthInfo(globalAuth.vercel);
    specs.push(key
      ? { runId: "vercel", providerSlot: "vercel", providerLabel: "vercel", authMode: "api", authInfo: apiAuth(key) }
      : { runId: "vercel", providerSlot: "vercel", providerLabel: "vercel", authMode: "api", authError: "Missing Vercel auth (AI_GATEWAY_API_KEY/VERCEL_API_KEY or ~/.magnitude/auth.json:vercel)" });
  }

  if (include("github-copilot")) {
    const copilotAuthInfo = globalAuth["github-copilot"];
    const token = process.env.GITHUB_COPILOT_TOKEN ?? process.env.COPILOT_OAUTH_TOKEN ?? oauthTokenFromAuthInfo(copilotAuthInfo);
    specs.push(token
      ? { runId: "github-copilot", providerSlot: "github-copilot", providerLabel: "github-copilot", authMode: "oauth", authInfo: oauthAuth(token) }
      : { runId: "github-copilot", providerSlot: "github-copilot", providerLabel: "github-copilot", authMode: "oauth", authError: "Missing Copilot auth (GITHUB_COPILOT_TOKEN/COPILOT_OAUTH_TOKEN or ~/.magnitude/auth.json:github-copilot)" });
  }

  if (include("gemini")) {
    const key = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY ?? apiKeyFromAuthInfo(globalAuth.google);
    specs.push(key
      ? { runId: "gemini", providerSlot: "google", providerLabel: "gemini", authMode: "api", authInfo: apiAuth(key) }
      : { runId: "gemini", providerSlot: "google", providerLabel: "gemini", authMode: "api", authError: "Missing Gemini auth (GOOGLE_API_KEY/GEMINI_API_KEY or ~/.magnitude/auth.json:google)" });
  }

  if (include("anthropic")) {
    const anthropicAuthInfo = globalAuth.anthropic;
    const apiKey = process.env.ANTHROPIC_API_KEY ?? apiKeyFromAuthInfo(anthropicAuthInfo);
    specs.push(apiKey
      ? { runId: "anthropic-api", providerSlot: "anthropic", providerLabel: "anthropic", authMode: "api", authInfo: apiAuth(apiKey) }
      : { runId: "anthropic-api", providerSlot: "anthropic", providerLabel: "anthropic", authMode: "api", authError: "Missing Anthropic auth (ANTHROPIC_API_KEY or ~/.magnitude/auth.json:anthropic)" });
  }

  return specs;
}

function createProviderStateLayer(providerId: RunSpec["providerSlot"]) {
  return Layer.succeed(ProviderState, {
    peek: (slot: string) => {
      if (slot !== "lead") return Effect.succeed(null);
      return Effect.succeed({ model: { providerId, id: `${providerId}-capture` } });
    },
  } as any);
}

function createProviderAuthLayer(spec: RunSpec) {
  const authMap: Record<string, AuthInfo | undefined> = {
    openai: spec.providerSlot === "openai" ? spec.authInfo : undefined,
    openrouter: spec.providerSlot === "openrouter" ? spec.authInfo : undefined,
    vercel: spec.providerSlot === "vercel" ? spec.authInfo : undefined,
    "github-copilot": spec.providerSlot === "github-copilot" ? spec.authInfo : undefined,
    google: spec.providerSlot === "google" ? spec.authInfo : undefined,
    anthropic: spec.providerSlot === "anthropic" ? spec.authInfo : undefined,
  };

  return Layer.succeed(ProviderAuth, {
    getAuth: (providerId: string) => Effect.succeed(authMap[providerId]),
  } as any);
}

function toSearchAuth(spec: RunSpec): SearchAuth {
  if (!spec.authInfo) throw new Error("Cannot resolve SearchAuth without authInfo");
  if (spec.authInfo.type === "api") return { type: "api-key", value: spec.authInfo.key };
  if (spec.authInfo.type === "oauth") {
    return { type: "oauth-token", value: spec.authInfo.accessToken, accountId: spec.authInfo.accountId };
  }
  throw new Error(`Unsupported auth info type for capture script: ${String((spec.authInfo as any).type)}`);
}

async function executeViaRouter(spec: RunSpec, query: string, options?: SearchOptions) {
  return Effect.runPromise(
    webSearch(query, options).pipe(
      Effect.provide(Layer.mergeAll(createProviderStateLayer(spec.providerSlot), createProviderAuthLayer(spec))),
    ) as any,
  );
}

async function executeViaDirectAdapter(spec: RunSpec, query: string, options?: SearchOptions) {
  const auth = toSearchAuth(spec);
  switch (spec.providerSlot) {
    case "openai":
      return openaiWebSearch(query, auth, options);
    case "openrouter":
      return openrouterWebSearch(query, auth, options);
    case "vercel":
      return vercelWebSearch(query, auth, options);
    case "github-copilot":
      return copilotWebSearch(query, auth, options);
    case "google":
      return geminiWebSearch(query, auth, options);
    case "anthropic":
      return anthropicWebSearch(query, auth, options);
  }
}

async function executeAnthropicStream(spec: RunSpec, query: string, options?: SearchOptions) {
  const auth = toSearchAuth(spec);
  const streamEvents: unknown[] = [];
  let doneResponse: unknown = null;
  for await (const event of webSearchStream(query, auth, options)) {
    streamEvents.push(event);
    if (event.type === "done") {
      doneResponse = event.response;
    }
  }
  return { streamEvents, normalizedResult: doneResponse };
}

async function withInterceptorsForSpec<T>(spec: RunSpec, run: () => Promise<T>): Promise<{
  value: T;
  request: unknown;
  response: unknown;
  responseRawText: string | null;
  streamEvents: unknown[];
}> {
  const fetchState: FetchCaptureState = { streamEvents: [] };
  const openaiState: OpenAISdkCaptureState = {};
  const googleState: GoogleSdkCaptureState = {};
  const anthropicState: AnthropicSdkCaptureState = { streamEvents: [] };

  const base = async () => run();

  const fetchWrapped =
    spec.providerSlot === "openai" && spec.authMode === "oauth" ||
    spec.providerSlot === "openrouter" ||
    spec.providerSlot === "github-copilot" ||
    spec.providerSlot === "vercel"
      ? () => withFetchInterceptor(fetchState, base)
      : base;

  const openaiWrapped =
    spec.providerSlot === "openai" && spec.authMode === "api"
      ? () => withOpenAISdkInterceptor(openaiState, fetchWrapped)
      : fetchWrapped;

  const googleWrapped =
    spec.providerSlot === "google"
      ? () => withGoogleSdkInterceptor(googleState, openaiWrapped)
      : openaiWrapped;

  const anthropicWrapped =
    spec.providerSlot === "anthropic"
      ? () => withAnthropicSdkInterceptor(anthropicState, googleWrapped)
      : googleWrapped;

  const value = await anthropicWrapped();

  const mergedStream = [
    ...fetchState.streamEvents,
    ...anthropicState.streamEvents,
  ];

  const request =
    fetchState.request ??
    openaiState.request ??
    googleState.request ??
    anthropicState.request ??
    sentinel("no-captured-request");

  const response =
    fetchState.response ??
    openaiState.response ??
    googleState.response ??
    anthropicState.response ??
    sentinel("no-captured-response");

  const responseRawText = fetchState.responseRawText ?? null;

  return { value, request, response, responseRawText, streamEvents: mergedStream };
}

function getStatusFromError(error: unknown): CaptureStatus {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("No ") && message.toLowerCase().includes("auth")) return "auth-missing";
  return "provider-error";
}

async function runOne(
  rootDir: string,
  spec: RunSpec,
  config: CliConfig,
  options?: SearchOptions,
): Promise<RunSummary> {
  const manifestBase: Omit<CaptureManifest, "status"> = {
    runId: spec.runId,
    provider: spec.providerLabel,
    authMode: spec.authMode,
    query: config.query,
    timestamp: new Date().toISOString(),
    notes: [config.directAdapter ? "mode:direct-adapter" : "mode:router"],
    prerequisites: {
      authPresent: Boolean(spec.authInfo),
    },
  };

  if (!spec.authInfo) {
    const artifacts: ProviderRunArtifacts = {
      manifest: { ...manifestBase, status: "auth-missing", notes: [...(manifestBase.notes ?? []), spec.authError ?? "missing auth"] },
      request: sentinel("no-network-attempt"),
      response: sentinel("no-network-attempt"),
      responseRawText: sentinel("no-network-attempt"),
      streamEvents: sentinel("no-network-attempt"),
      normalizedResult: sentinel("no-network-attempt"),
      error: { message: spec.authError ?? "Missing credentials" },
    };

    const artifactDir = await writeRunArtifacts(rootDir, spec.runId, artifacts);
    return {
      runId: spec.runId,
      provider: spec.providerLabel,
      authMode: spec.authMode,
      status: "auth-missing",
      artifactDir,
    };
  }

  try {
    const vercelSpecialCase =
      spec.providerSlot === "vercel" && spec.authInfo?.type === "api" && !config.directAdapter;

    const { value, request, response, responseRawText, streamEvents } = await withInterceptorsForSpec(spec, async () => {
      if (vercelSpecialCase) {
        return runVercelAiSdkCapture(config.query, spec.authInfo!.key, options);
      }
      if (spec.providerSlot === "anthropic" && config.streamAnthropic) {
        return executeAnthropicStream(spec, config.query, options);
      }
      if (config.directAdapter) {
        return executeViaDirectAdapter(spec, config.query, options);
      }
      return executeViaRouter(spec, config.query, options);
    });

    const artifacts: ProviderRunArtifacts = {
      manifest: {
        ...manifestBase,
        status: "success",
        ...(vercelSpecialCase
          ? { notes: [...(manifestBase.notes ?? []), "vercel:ai-sdk-capture-path"] }
          : {}),
      },
      request: vercelSpecialCase
        ? ((value as any).request as any)
        : (request as any)?.present === false
          ? request as any
          : ({ present: true, ...(request as any) } as any),
      response: vercelSpecialCase
        ? ((value as any).response as any)
        : (response as any)?.present === false
          ? response as any
          : ({ present: true, ...(response as any) } as any),
      responseRawText: vercelSpecialCase
        ? ((value as any).responseRawText ?? sentinel("not-text-response"))
        : responseRawText ?? sentinel("not-text-response"),
      streamEvents: vercelSpecialCase
        ? (((value as any).streamEvents?.length ?? 0) > 0
          ? (value as any).streamEvents
          : sentinel("non-streaming-provider-or-no-events"))
        : streamEvents.length > 0
          ? streamEvents
          : sentinel("non-streaming-provider-or-no-events"),
      normalizedResult: vercelSpecialCase
        ? (value as any).normalizedResult
        : spec.providerSlot === "anthropic" && config.streamAnthropic
          ? (value as any).normalizedResult ?? sentinel("missing-done-response")
          : value,
      error: sentinel("none"),
    };

    const artifactDir = await writeRunArtifacts(rootDir, spec.runId, artifacts);
    return {
      runId: spec.runId,
      provider: spec.providerLabel,
      authMode: spec.authMode,
      status: "success",
      artifactDir,
    };
  } catch (error) {
    const status = getStatusFromError(error);
    const artifacts: ProviderRunArtifacts = {
      manifest: { ...manifestBase, status },
      request: sentinel("capture-request-unavailable-due-to-error"),
      response: sentinel("capture-response-unavailable-due-to-error"),
      responseRawText: sentinel("capture-response-unavailable-due-to-error"),
      streamEvents: sentinel("capture-stream-unavailable-due-to-error"),
      normalizedResult: sentinel("no-normalized-result"),
      error: {
        name: error instanceof Error ? error.name : "Error",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
      },
    };

    const artifactDir = await writeRunArtifacts(rootDir, spec.runId, artifacts);
    return {
      runId: spec.runId,
      provider: spec.providerLabel,
      authMode: spec.authMode,
      status,
      artifactDir,
    };
  }
}

async function main(): Promise<void> {
  const config = parseArgs(Bun.argv.slice(2));
  const options = getSearchOptions(config);
  const rootDir = timestampedCaptureRoot(config.outDir ? resolve(config.outDir) : undefined);
  await mkdir(rootDir, { recursive: true });

  const specs = buildRunSpecs(config);
  if (specs.length === 0) {
    console.error("No provider runs resolved from CLI options.");
    process.exit(1);
  }

  const summaries: RunSummary[] = [];
  for (const spec of specs) {
    const summary = await runOne(rootDir, spec, config, options);
    summaries.push(summary);
    console.log(`[web-search-capture] ${summary.runId}: ${summary.status}`);
  }

  await writeIndex(rootDir, summaries);
  console.log(`\nCapture artifacts written to: ${rootDir}`);
}

if (import.meta.main) {
  await main();
}
