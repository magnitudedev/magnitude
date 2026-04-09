import { safeJson } from "../capture-harness";
import { __captureOnly } from "../../../src/tools/web-search-anthropic";

export interface AnthropicSdkCaptureState {
  request?: unknown;
  response?: unknown;
  streamEvents: unknown[];
}

export async function withAnthropicSdkInterceptor<T>(
  state: AnthropicSdkCaptureState,
  run: () => Promise<T>,
): Promise<T> {
  return __captureOnly.withInterceptor(
    {
      onCreate: async ({ client, args }, next) => {
        state.request = safeJson({
          present: true,
          client,
          args,
        });
        const result = await next();
        state.response = safeJson({
          present: true,
          value: result,
        });
        return result;
      },
      onStream: async ({ client, args }, next) => {
        state.request = safeJson({
          present: true,
          client,
          args,
        });
        const stream = await next();
        state.streamEvents.push({ type: "stream-opened" });
        return __captureOnly.wrapStream(stream, (event) => {
          state.streamEvents.push(safeJson(event));
        }, async (finalMessage) => {
          state.response = safeJson({
            present: true,
            value: finalMessage,
          });
        });
      },
    },
    run,
  );
}
