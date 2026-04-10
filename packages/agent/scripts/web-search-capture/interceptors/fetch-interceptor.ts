import { safeJson, toHeaderRecord } from "../capture-harness";

export interface FetchCaptureState {
  request?: unknown;
  response?: unknown;
  responseRawText?: string;
  streamEvents: unknown[];
}

function cloneRequestHeaders(init?: RequestInit): Record<string, string> {
  if (!init?.headers) return {};
  if (init.headers instanceof Headers) return toHeaderRecord(init.headers);
  if (Array.isArray(init.headers)) return toHeaderRecord(Object.fromEntries(init.headers.map(([k, v]) => [k, String(v)])));
  return toHeaderRecord(init.headers as Record<string, string>);
}

function createStreamFromText(text: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
}

export async function withFetchInterceptor<T>(
  state: FetchCaptureState,
  run: () => Promise<T>,
): Promise<T> {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const requestUrl = typeof input === "string" || input instanceof URL ? String(input) : input.url;
    const method = init?.method ?? (input instanceof Request ? input.method : "GET");
    const bodyText =
      typeof init?.body === "string"
        ? init.body
        : init?.body
          ? String(init.body)
          : null;

    let bodyJson: unknown = null;
    if (bodyText) {
      try {
        bodyJson = JSON.parse(bodyText);
      } catch {
        bodyJson = null;
      }
    }

    state.request = safeJson({
      present: true,
      url: requestUrl,
      method,
      headers: cloneRequestHeaders(init),
      bodyText,
      bodyJson,
    });

    const response = await originalFetch(input as any, init);
    const rawText = await response.clone().text();
    state.responseRawText = rawText;
    let bodyJsonResponse: unknown = null;
    try {
      bodyJsonResponse = rawText ? JSON.parse(rawText) : null;
    } catch {
      bodyJsonResponse = null;
    }

    state.response = safeJson({
      present: true,
      url: response.url || requestUrl,
      status: response.status,
      statusText: response.statusText,
      headers: toHeaderRecord(response.headers),
      bodyText: rawText,
      bodyJson: bodyJsonResponse,
    });

    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("text/event-stream")) {
      for (const line of rawText.split(/\r?\n/)) {
        if (line.length > 0) state.streamEvents.push({ line });
      }
      return new Response(createStreamFromText(rawText), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return new Response(rawText, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }) as typeof fetch;

  try {
    return await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}
