import { ConfigProvider, Effect, Option } from "effect";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readPublicBaseline } from "./public-baseline";

const release = (
  version: string,
  published_at: string,
  prerelease = false
) => ({
  draft: false,
  prerelease,
  published_at,
  tag_name: `@magnitudedev/cli@${version}`,
  assets: [
    {
      name: "magnitude-release.json",
      url: `https://api.github.com/assets/${version}`,
    },
  ],
});
const run = () =>
  Effect.runPromise(
    readPublicBaseline.pipe(
      Effect.withConfigProvider(ConfigProvider.fromMap(new Map()))
    )
  );
afterEach(() => vi.unstubAllGlobals());
describe("stable public baseline", () => {
  it("ignores alpha/beta, unflagged prerelease tags, drafts, and unpublished entries", async () => {
    const requests: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      requests.push(url);
      return Response.json(
        url.includes("/releases?")
          ? [
              release("1.0.0", "2026-01-01"),
              release("0.9.0", "2025-12-01"),
              { ...release("3.0.0", "2026-05-01"), published_at: null },
              release("1.1.0-alpha.0", "2026-02-01", true),
              release("1.1.0-beta.0", "2026-03-01"),
              { ...release("2.0.0", "2026-04-01"), draft: true },
            ]
          : {
              tag: "@magnitudedev/cli@1.0.0",
              version: "1.0.0",
              sourceCommit: "a".repeat(40),
            }
      );
    });
    expect(Option.getOrThrow(await run()).cliVersion).toBe("1.0.0");
    expect(requests[1]).toBe("https://api.github.com/assets/1.0.0");
  });
  it("does not invent a stable baseline when only prereleases exist", async () => {
    vi.stubGlobal("fetch", async () =>
      Response.json([release("1.0.0-alpha.0", "2026-02-01", true)])
    );
    expect(Option.isNone(await run())).toBe(true);
  });
});
