import { safeJson } from "../capture-harness";
import { __captureOnly } from "../../../src/tools/web-search-gemini";

export interface GoogleSdkCaptureState {
  request?: unknown;
  response?: unknown;
}

export async function withGoogleSdkInterceptor<T>(
  state: GoogleSdkCaptureState,
  run: () => Promise<T>,
): Promise<T> {
  return __captureOnly.withGenerateContentInterceptor(async ({ client, args }, next) => {
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
  }, run);
}
