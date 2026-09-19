import { Effect, Option } from "effect"
import { expect, test } from "vitest"
import { admitGuestUser } from "../src/guest-user"

test("only disposable cloud allocations with the actual non-root account can grant installed context", () => Effect.runPromise(Effect.gen(function* () {
  const identity = { home: "/home/labworker", username: "labworker", uid: 1000 }, environment = { HOME: "/home/labworker" }
  for (const provider of ["azure", "namespace"] as const) expect(Option.isSome(yield* admitGuestUser(provider, true, identity, environment, "linux"))).toBe(true)
  for (const provider of ["local", "spark"] as const) expect((yield* admitGuestUser(provider, true, identity, environment, "linux").pipe(Effect.either))._tag).toBe("Left")
  expect(Option.isNone(yield* admitGuestUser("local", false, identity, environment, "linux"))).toBe(true)
  for (const user of [{ ...identity, uid: 0 }, { ...identity, home: "relative" }]) expect((yield* admitGuestUser("azure", true, user, environment, "linux").pipe(Effect.either))._tag).toBe("Left")
  expect((yield* admitGuestUser("azure", true, identity, { HOME: "/somewhere-else" }, "linux").pipe(Effect.either))._tag).toBe("Left")
})))

test("Windows uses the actual account profile and rejects a conflicting HOME override", () => Effect.runPromise(Effect.gen(function* () {
  const identity = { home: "C:\\Users\\labworker", username: "labworker", uid: -1 }
  expect(Option.isSome(yield* admitGuestUser("azure", true, identity, { USERPROFILE: identity.home }, "win32"))).toBe(true)
  const invalidEnvironments: Readonly<Record<string, string>>[] = [{ HOME: identity.home }, { USERPROFILE: identity.home, HOME: "C:\\Other" }]
  for (const environment of invalidEnvironments) {
    expect((yield* admitGuestUser("azure", true, identity, environment, "win32").pipe(Effect.either))._tag).toBe("Left")
  }
})))
