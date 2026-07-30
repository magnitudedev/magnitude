import { describe, expect, it } from "vitest";
import * as FileSystem from "@effect/platform/FileSystem";
import { BunFileSystem } from "@effect/platform-bun";
import { Effect, Option } from "effect";
import { AcnOwnerIdSchema } from "@magnitudedev/acn-protocol";
import { mkdtemp } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import {
  listRegisteredAcns,
  readRegistration,
  readRegistrationOwnership,
  registrationIsOwnedBy,
  registrationPath,
  writeRegistrationAtomic,
  type AcnRegistration,
} from "./daemon-registration";

const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem>) =>
  Effect.runPromise(effect.pipe(Effect.provide(BunFileSystem.layer)));

const registration = (id: string): AcnRegistration => ({
  id: AcnOwnerIdSchema.make(id),
  version: "1.0.0",
  url: "http://127.0.0.1:1234",
  pid: 1234,
  timestamp: 1,
});

describe("daemon registration", () => {
  it("fails ownership closed for missing and different registrations", () => {
    expect(registrationIsOwnedBy(Option.none(), "owner-1")).toBe(false);
    expect(
      registrationIsOwnedBy(Option.some(registration("owner-2")), "owner-1")
    ).toBe(false);
    expect(
      registrationIsOwnedBy(Option.some(registration("owner-1")), "owner-1")
    ).toBe(true);
  });

  it("writes and reads registration atomically", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-acn-"));
    const path = registrationPath(dir);
    expect(path).toContain(join("acn", "registry.json"));

    await run(writeRegistrationAtomic(path, registration("owner-1")));

    expect(await run(readRegistration(path))).toEqual(
      Option.some(
        expect.objectContaining({
          id: "owner-1",
          version: "1.0.0",
          url: "http://127.0.0.1:1234",
        })
      )
    );
  });

  it("treats invalid registration content as absent without defects", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-acn-"));
    const path = registrationPath(dir);

    await Bun.write(path, "{");
    expect(await run(readRegistration(path))).toEqual(Option.none());

    await Bun.write(
      path,
      JSON.stringify({ schemaVersion: 1, registration: 42 })
    );
    expect(await run(readRegistration(path))).toEqual(Option.none());
  });

  it("reads ownership without depending on evolving registration fields", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-acn-ownership-"));
    const path = registrationPath(dir);
    await Bun.write(
      path,
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          ...registration("owner-1"),
          shutdownToken: "field-from-an-older-generation",
          futureField: { arbitrary: true },
        },
      })
    );

    expect(await run(readRegistrationOwnership(path))).toEqual(
      Option.some({ id: "owner-1" })
    );
  });

  it("lists registered ACNs", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-acn-"));
    await run(
      writeRegistrationAtomic(registrationPath(dir), registration("owner-1"))
    );
    const registrations = await run(listRegisteredAcns(dir));

    expect(registrations.map((entry) => entry.registration.id)).toEqual([
      "owner-1",
    ]);
    expect(registrations.map((entry) => entry.path)).toEqual([
      registrationPath(dir),
    ]);
  });
});
