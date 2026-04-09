import OpenAI from "openai";
import { safeJson } from "../capture-harness";

export interface OpenAISdkCaptureState {
  request?: unknown;
  response?: unknown;
}

export async function withOpenAISdkInterceptor<T>(
  state: OpenAISdkCaptureState,
  run: () => Promise<T>,
): Promise<T> {
  const originalPost = OpenAI.prototype.post;

  OpenAI.prototype.post = (function patchedPost(this: any, path: string, options: any) {
    const shouldCapture = path === "/responses";
    if (shouldCapture) {
      state.request = safeJson({
        present: true,
        client: {
          baseURL: this?.baseURL,
        },
        args: options?.body ?? options,
      });
    }

    const result = originalPost.call(this, path, options);

    if (!shouldCapture) return result;

    return Promise.resolve(result).then((value: unknown) => {
      state.response = safeJson({
        present: true,
        value,
      });
      return value;
    });
  }) as typeof OpenAI.prototype.post;

  try {
    return await run();
  } finally {
    OpenAI.prototype.post = originalPost;
  }
}
