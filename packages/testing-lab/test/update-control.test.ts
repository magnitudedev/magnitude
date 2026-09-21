import { Effect, Option } from "effect"
import { expect, test } from "vitest"
import { NativeHost } from "../../daemon-management/src/desktop-native"
import { updateControlEndpoint } from "../src/update-control"

test("Windows update observations discover the replacement pipe instead of reusing the old owner's pipe", async () => {
  let current: Option.Option<string> = Option.none()
  const native = { inspectEndpoint: (directory: string) => Effect.sync(() => {
    expect(directory).toBe("private-state")
    return current
  }) } as unknown as NativeHost
  const observe = updateControlEndpoint("private-state", true).pipe(Effect.provideService(NativeHost, native))
  const missing = await Effect.runPromise(Effect.either(observe))
  expect(missing._tag === "Left" && missing.left._tag).toBe("ApplicationControlUnavailable")
  current = Option.some("\\\\.\\pipe\\previous")
  expect(await Effect.runPromise(observe)).toBe(Option.getOrThrow(current))
  current = Option.some("\\\\.\\pipe\\replacement")
  expect(await Effect.runPromise(observe)).toBe(Option.getOrThrow(current))
})

test("Unix update observations use the owned state directory socket", async () => {
  const endpoint = await Effect.runPromise(updateControlEndpoint("/tmp/owned", false).pipe(
    Effect.provideService(NativeHost, { inspectEndpoint: () => Effect.die("Must not load Windows host") } as unknown as NativeHost)))
  expect(endpoint).toBe("/tmp/owned/application.sock")
})
