import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { DesktopProcessRequest, verifyWindowsDesktopProcesses, WindowsDesktopProcesses } from "../src/desktop-process"

const request = Schema.decodeUnknownSync(DesktopProcessRequest)({ launcherPid: 100, applicationPid: 200, executable: "C:\\App\\Magnitude.exe" })
const observed = {
  userSid: "S-1-5-21-123",
  launcher: { pid: 100, birth: "100000000000000000", userSid: "S-1-5-21-123" },
  application: { pid: 200, parentPid: 100, birth: "100000000000000001", userSid: "S-1-5-21-123", executable: "c:\\app\\Magnitude.exe" },
}
const verify = (value: unknown) => Schema.decodeUnknown(WindowsDesktopProcesses)(value).pipe(
  Effect.flatMap(value => verifyWindowsDesktopProcesses(request, value)))

test("Windows application identity belongs to the actual Electron child, not Playwright's shell", async () => {
  expect(await Effect.runPromise(verify(observed))).toBe(200)
})

test("Windows desktop ownership rejects a foreign process, user, executable or reused parent PID", async () => {
  for (const application of [
    { ...observed.application, pid: 201 },
    { ...observed.application, parentPid: 101 },
    { ...observed.application, pid: 100 },
    { ...observed.application, userSid: "S-1-5-21-456" },
    { ...observed.application, executable: "C:\\Other\\Magnitude.exe" },
    { ...observed.application, executable: "Magnitude.exe" },
    { ...observed.application, birth: "99999999999999999" },
  ]) expect((await Effect.runPromise(verify({ ...observed, application }).pipe(Effect.either)))._tag).toBe("Left")
  for (const launcher of [
    { ...observed.launcher, pid: 101 },
    { ...observed.launcher, userSid: "S-1-5-21-456" },
  ]) expect((await Effect.runPromise(verify({ ...observed, launcher }).pipe(Effect.either)))._tag).toBe("Left")
})
