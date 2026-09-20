import { Effect } from "effect"
import { expect, test } from "vitest"
import { selectBuildBackendPacks } from "../src/native-build-selection"
import { targets } from "../src/catalog"

test("every lab target includes required distribution packs without changing execution acceptance", () => Effect.runPromise(Effect.gen(function* () {
  for (const target of targets) {
    const packs = yield* selectBuildBackendPacks(target.artifactHost, target.backend)
    const apple = target.artifactHost === "darwin-arm64"
    expect(packs).toHaveLength(target.backend === "cpu" && !apple ? 0 : 1)
    for (const pack of packs) {
      expect(pack.host).toBe(target.artifactHost)
      expect(pack.backend).toBe(apple && target.backend === "cpu" ? "metal" : target.backend)
      if (pack.backend === "cuda") expect(pack.cuda.toolkitVersion).toBe("12.9")
    }
  }
  expect((yield* selectBuildBackendPacks("linux-x64-gnu", "metal").pipe(Effect.either))._tag).toBe("Left")
  expect((yield* selectBuildBackendPacks("darwin-arm64", "cuda").pipe(Effect.either))._tag).toBe("Left")
})))
