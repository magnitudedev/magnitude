import { ClientIdSchema, type ClientId } from "@magnitudedev/acn-protocol";
import { Duration, Effect, TestClock, TestContext } from "effect";
import { describe, expect, it } from "vitest";
import { makeAcnClientLeaseOwner } from "./acn-recovering-client";

describe("AcnClientLeaseOwner", () => {
  it("renews immediately and every fifteen seconds, then releases once", async () => {
    const renewals: string[] = [];
    const releases: string[] = [];
    await Effect.runPromise(
      Effect.provide(
        Effect.scoped(
          Effect.gen(function* () {
            const clientId = ClientIdSchema.make("client-1");
            const client = {
              RenewClientLease: ({
                clientId: renewed,
              }: {
                clientId: ClientId;
              }) =>
                Effect.sync(() => {
                  renewals.push(renewed);
                  return { connectedClientCount: 1 };
                }),
              ReleaseClientLease: ({
                clientId: released,
              }: {
                clientId: ClientId;
              }) =>
                Effect.sync(() => {
                  releases.push(released);
                  return { connectedClientCount: 0 };
                }),
            } as never;
            const owner = yield* makeAcnClientLeaseOwner(clientId);
            yield* owner.establishThrough(client);

            yield* Effect.yieldNow();
            expect(renewals).toEqual([clientId]);
            yield* TestClock.adjust(Duration.seconds(30));
            yield* Effect.yieldNow();
            expect(renewals).toEqual([clientId, clientId, clientId]);

            expect((yield* owner.releaseThrough(client)).connectedClientCount).toBe(0);
            expect((yield* owner.releaseThrough(client)).connectedClientCount).toBe(0);
            expect(releases).toEqual([clientId]);
            yield* TestClock.adjust(Duration.minutes(1));
            expect(renewals).toEqual([clientId, clientId, clientId]);
          })
        ),
        TestContext.TestContext
      )
    );
  });
});
