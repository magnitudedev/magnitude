import { Effect } from "effect"
import { expect, it } from "vitest"
import { linuxUpdateCallerUid } from "./linux-update-maintenance"
it.each([{ PKEXEC_UID: "1000" }, { SUDO_UID: "1000" }, { PKEXEC_UID: "1000", SUDO_UID: "1000" }])("accepts the authorizing user's identity %#", async environment => {
  expect(await Effect.runPromise(linuxUpdateCallerUid(environment))).toBe(1000)
})
it.each([{}, { SUDO_UID: "0" }, { SUDO_UID: "-1" }, { SUDO_UID: "1.5" }, { SUDO_UID: "1e3" },
  { SUDO_UID: "9007199254740992" }, { PKEXEC_UID: "1000", SUDO_UID: "1001" }, { PKEXEC_UID: "", SUDO_UID: "1000" }])("rejects missing, invalid or conflicting authorization identity %#", async environment => {
  expect(await Effect.runPromise(linuxUpdateCallerUid(environment).pipe(Effect.isFailure))).toBe(true)
})
