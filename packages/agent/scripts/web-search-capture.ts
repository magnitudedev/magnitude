import { existsSync, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { Effect, Layer } from "effect";
import { ProviderAuth, ProviderState, type AuthInfo } from "@magnitudedev/providers";
import { webSearch, webSearchStream, type SearchAuth, type SearchOptions } from "../src/tools/web-search";
import { runDirectAdapter } from "./web-search-capture/runners";
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
import { withAnthropicSdkInterceptor, type AnthropicSdkCaptureState } from "./web-search-capture/interceptors/anthropic-sdk-interceptor";

type ProviderCli = "openai" | "openrouter" | "vercel" | "anthropic" | "all";
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
  providerSlot: "openai" | "openrouter" | "vercel" | "anthropic";
  providerLabel: string;
  authMode: "api" | "oauth";
  authInfo?: AuthInfo;
  authError?: string;
  authSource?: "env" | "stored";
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
  --provider <openai|openrouter|vercel|anthropic|all>
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

function oauthAuth(
  accessToken: string,
  accountId?: string,
  refreshToken = "capture-script",
  expiresAt = Date.now() + 60 * 60 * 1000,
): AuthInfo {
  return {
    type: "oauth",
    accessToken,
    refreshToken,
    expiresAt,
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
      const refreshToken = openaiAuthInfo?.type === "oauth" ? openaiAuthInfo.refreshToken : "capture-script";
      const expiresAt = openaiAuthInfo?.type === "oauth" ? openaiAuthInfo.expiresAt : Date.now() + 60 * 60 * 1000;
      specs.push(openaiOauthToken
        ? {
            runId: "openai-oauth",
            providerSlot: "openai",
            providerLabel: "openai",
            authMode: "oauth",
            authInfo: oauthAuth(openaiOauthToken, openaiAccountId, refreshToken, expiresAt),
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
    const envToken = process.env.GITHUB_COPILOT_TOKEN ?? process.env.COPILOT_OAUTH_TOKEN;
    const storedToken = oauthTokenFromAuthInfo(copilotAuthInfo);
    const token = envToken ?? storedToken;
    const refreshToken = copilotAuthInfo?.type === "oauth" ? copilotAuthInfo.refreshToken : "capture-script";
    const expiresAt = copilotAuthInfo?.type === "oauth" ? copilotAuthInfo.expiresAt : Date.now() + 60 * 60 * 1000;
    specs.push(token
      ? {
          runId: "github-copilot",
          providerSlot: "github-copilot",
          providerLabel: "github-copilot",
          authMode: "oauth",
          authInfo: oauthAuth(token, undefined, refreshToken, expiresAt),
          authSource: envToken ? "env" : "stored",
        }
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
    setAuth: (providerId: string, auth: AuthInfo) =>
      Effect.sync(() => {
        authMap[providerId] = auth;
      }),
    refresh: (providerId: string, refreshToken: string) =>
      Effect.tryPromise({
        try: async () => {
          return null;
        },
        catch: (error) => (error instanceof Error ? error : new Error(String(error))),
      }),
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
  return runDirectAdapter(spec.providerSlot, query, toSearchAuth(spec), options);
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

type InterceptorCapture = {
  request: unknown;
  response: unknown;
  responseRawText: string | null;
  streamEvents: unknown[];
};

type CaptureWrappedError = Error & {
  capture?: InterceptorCapture;
};

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
    spec.providerSlot === "vercel" ||
    spec.providerSlot === "github-copilot"
      ? () => withFetchInterceptor(fetchState, base)
      : base;

  const openaiWrapped =
    (spec.providerSlot === "openai" && spec.authMode === "api") || spec.providerSlot === "vercel"
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

  const buildCapture = (): InterceptorCapture => {
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

    return { request, response, responseRawText, streamEvents: mergedStream };
  };

  try {
    const value = await anthropicWrapped();
    const capture = buildCapture();
    return { value, ...capture };
  } catch (error) {
    const wrapped = (error instanceof Error ? error : new Error(String(error))) as CaptureWrappedError;
    wrapped.capture = buildCapture();
    throw wrapped;
  }
}

function getStatusFromError(error: unknown): CaptureStatus {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("No ") && message.toLowerCase().includes("auth")) return "auth-missing";
  return "provider-error";
}

function isCopilotTokenExpiredError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("GitHub Copilot web search error 401") &&
    message.toLowerCase().includes("token expired")
  );
}

function hasNonEmptyToolsPayload(request: unknown): boolean {
  const args = (request as any)?.args;
  if (args && typeof args === "object") {
    const tools = (args as any).tools;
    if (tools && typeof tools === "object" && !Array.isArray(tools)) {
      return Object.keys(tools).length > 0;
    }
  }

  const bodyJsonTools = (request as any)?.bodyJson?.tools;
  if (Array.isArray(bodyJsonTools)) {
    return bodyJsonTools.length > 0;
  }

  return false;
}

function hasStructuredSources(normalizedResult: unknown): boolean {
  const results = (normalizedResult as any)?.results;
  if (!Array.isArray(results) || results.length === 0) return false;
  for (const entry of results) {
    const content = (entry as any)?.content;
    if (!Array.isArray(content)) continue;
    if (content.some((source) => typeof source?.url === "string" && source.url.length > 0)) {
      return true;
    }
  }
  return false;
}

function getWebSearchRequests(normalizedResult: unknown): number {
  const value = (normalizedResult as any)?.usage?.web_search_requests;
  return typeof value === "number" ? value : 0;
}

function getVercelSpecialCaseStatusAndNotes(value: unknown): { status: CaptureStatus; notes: string[] } {
  const diagnostics = (value as any)?.diagnostics as
    | {
        unsupportedToolWarning?: boolean;
        requestToolsDropped?: boolean;
        requestedWebSearchTool?: boolean;
        hasCitations?: boolean;
        warningTypes?: string[];
      }
    | undefined;
  const request = (value as any)?.request;
  const response = (value as any)?.response;
  const normalizedResult = (value as any)?.normalizedResult;

  const warningTypes: string[] = [];
  if (Array.isArray(diagnostics?.warningTypes)) {
    warningTypes.push(...diagnostics.warningTypes);
  }
  const responseWarnings = (response as any)?.bodyJson?.warnings;
  if (Array.isArray(responseWarnings)) {
    for (const warning of responseWarnings) {
      const type = typeof warning?.type === "string" ? warning.type : null;
      if (type && !warningTypes.includes(type)) warningTypes.push(type);
    }
  }

  const unsupportedToolWarning =
    Boolean(diagnostics?.unsupportedToolWarning) || warningTypes.includes("unsupported-tool");

  const requestToolsDropped = Boolean(diagnostics?.requestToolsDropped) || (
    Array.isArray((request as any)?.bodyJson?.tools) && (request as any).bodyJson.tools.length === 0
  );

  const notes: string[] = [];
  if (unsupportedToolWarning) {
    notes.push("vercel:unsupported-tool-warning");
  }
  if (requestToolsDropped) {
    notes.push("vercel:request-tools-dropped");
  }
  if (warningTypes.length > 0) {
    notes.push(`vercel:warning-types=${warningTypes.join(",")}`);
  }

  if (!hasNonEmptyToolsPayload(request)) {
    notes.push("vercel:missing-tool-payload");
  }
  if (!hasStructuredSources(normalizedResult)) {
    notes.push("vercel:missing-structured-sources");
  }
  if (getWebSearchRequests(normalizedResult) < 1) {
    notes.push("vercel:web-search-requests-lt-1");
  }

  const hasFailureSignal =
    unsupportedToolWarning ||
    requestToolsDropped ||
    !hasNonEmptyToolsPayload(request) ||
    !hasStructuredSources(normalizedResult) ||
    getWebSearchRequests(normalizedResult) < 1;

  return {
    status: hasFailureSignal ? "capture-error" : "success",
    notes,
  };
}

function hasCopilotWebSearchTool(request: unknown): boolean {
  const tools = (request as any)?.bodyJson?.tools;
  if (!Array.isArray(tools)) return false;
  return tools.some((tool) => tool && typeof tool === "object" && (tool as any).type === "web_search");
}

function hasCopilotSourcesInclude(request: unknown): boolean {
  const include = (request as any)?.bodyJson?.include;
  return Array.isArray(include) && include.includes("web_search_call.action.sources");
}

function getCopilotSignals(response: unknown) {
  const output = Array.isArray((response as any)?.bodyJson?.output) ? (response as any).bodyJson.output : [];
  let hasWebSearchCallSources = false;
  let hasUrlCitation = false;

  for (const item of output) {
    if (item?.type === "web_search_call") {
      const sources = item?.action?.sources;
      if (Array.isArray(sources) && sources.length > 0) {
        hasWebSearchCallSources = true;
      }
    }
    if (item?.type === "message") {
      const content = Array.isArray(item?.content) ? item.content : [];
      for (const entry of content) {
        const annotations = Array.isArray(entry?.annotations) ? entry.annotations : [];
        if (annotations.some((annotation) => annotation?.type === "url_citation")) {
          hasUrlCitation = true;
        }
      }
    }
  }

  return { hasWebSearchCallSources, hasUrlCitation };
}

function getCopilotStatusAndNotes(value: {
  request: unknown;
  response: unknown;
  normalizedResult: unknown;
  authSource?: "env" | "stored";
}): { status: CaptureStatus; notes: string[] } {
  const notes: string[] = [
    `copilot:auth-source=${value.authSource ?? "unknown"}`,
  ];

  const hasTool = hasCopilotWebSearchTool(value.request);
  const hasInclude = hasCopilotSourcesInclude(value.request);
  const webSearchRequests = getWebSearchRequests(value.normalizedResult);
  const signals = getCopilotSignals(value.response);

  notes.push(`copilot:returned-web-search-sources=${String(signals.hasWebSearchCallSources)}`);
  notes.push(`copilot:returned-url-citation=${String(signals.hasUrlCitation)}`);

  if (!hasTool) notes.push("copilot:missing-web-search-tool");
  if (!hasInclude) notes.push("copilot:missing-include-sources");
  if (!signals.hasWebSearchCallSources && !signals.hasUrlCitation) notes.push("copilot:missing-response-sources");
  if (webSearchRequests < 1) notes.push("copilot:web-search-requests-lt-1");

  const hasFailure =
    !hasTool ||
    !hasInclude ||
    (!signals.hasWebSearchCallSources && !signals.hasUrlCitation) ||
    webSearchRequests < 1;

  return {
    status: hasFailure ? "capture-error" : "success",
    notes,
  };
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
    const vercelSpecialCase = spec.providerSlot === "vercel";
    const copilotSpecialCase = spec.providerSlot === "github-copilot";

    const executeWithMode = async () => {
      if (spec.providerSlot === "anthropic" && config.streamAnthropic) {
        return executeAnthropicStream(spec, config.query, options);
      }
      if (config.directAdapter) {
        return executeViaDirectAdapter(spec, config.query, options);
      }
      return executeViaRouter(spec, config.query, options);
    };

    let captured: Awaited<ReturnType<typeof withInterceptorsForSpec<unknown>>>;
    try {
      captured = await withInterceptorsForSpec(spec, executeWithMode);
    } catch (error) {
      const shouldRefreshCopilotToken =
        spec.providerSlot === "github-copilot" &&
        spec.authInfo?.type === "oauth" &&
        spec.authSource !== "env" &&
        typeof spec.authInfo.refreshToken === "string" &&
        spec.authInfo.refreshToken.length > 0 &&
        isCopilotTokenExpiredError(error);

      if (!shouldRefreshCopilotToken) {
        throw error;
      }

      captured = await withInterceptorsForSpec(spec, executeWithMode);
    }

    const { value, request, response, responseRawText, streamEvents } = captured;

    const normalizedResult =
      spec.providerSlot === "anthropic" && config.streamAnthropic
        ? (value as any).normalizedResult ?? sentinel("missing-done-response")
        : value;

    const vercelAssessment = vercelSpecialCase
      ? getVercelSpecialCaseStatusAndNotes({
          diagnostics: (value as any)?.diagnostics,
          request,
          response,
          normalizedResult,
        })
      : null;

    const copilotAssessment = copilotSpecialCase
      ? getCopilotStatusAndNotes({
          request,
          response,
          normalizedResult,
          authSource: spec.authSource,
        })
      : null;

    const assessment = vercelAssessment ?? copilotAssessment ?? { status: "success" as const, notes: [] };

    const artifacts: ProviderRunArtifacts = {
      manifest: {
        ...manifestBase,
        status: assessment.status,
        notes: [
          ...(manifestBase.notes ?? []),
          ...assessment.notes,
        ],
      },
      request: (request as any)?.present === false
        ? request as any
        : ({ present: true, ...(request as any) } as any),
      response: (response as any)?.present === false
        ? response as any
        : ({ present: true, ...(response as any) } as any),
      responseRawText: responseRawText ?? sentinel("not-text-response"),
      streamEvents: streamEvents.length > 0
        ? streamEvents
        : sentinel("non-streaming-provider-or-no-events"),
      normalizedResult,
      error: sentinel("none"),
    };

    const artifactDir = await writeRunArtifacts(rootDir, spec.runId, artifacts);
    return {
      runId: spec.runId,
      provider: spec.providerLabel,
      authMode: spec.authMode,
      status: assessment.status,
      artifactDir,
    };
  } catch (error) {
    const status = getStatusFromError(error);
    const wrapped = error as CaptureWrappedError;
    const artifacts: ProviderRunArtifacts = {
      manifest: { ...manifestBase, status },
      request: wrapped.capture?.request ?? sentinel("capture-request-unavailable-due-to-error"),
      response: wrapped.capture?.response ?? sentinel("capture-response-unavailable-due-to-error"),
      responseRawText: wrapped.capture?.responseRawText ?? sentinel("capture-response-unavailable-due-to-error"),
      streamEvents: wrapped.capture?.streamEvents ?? sentinel("capture-stream-unavailable-due-to-error"),
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

export const __testOnly = {
  getVercelSpecialCaseStatusAndNotes,
  getCopilotStatusAndNotes,
  hasCopilotWebSearchTool,
  hasCopilotSourcesInclude,
  getCopilotSignals,
};
