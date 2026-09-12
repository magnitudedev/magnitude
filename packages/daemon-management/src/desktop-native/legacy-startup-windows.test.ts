import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { inspectLegacyWindowsStartup, makeWindowsLegacyStartup } from "./legacy-startup-windows"
import { LegacyStartupFailed } from "./legacy-startup-command"
import { WindowsLegacyTaskSnapshot } from "./windows-task-query"

// Constructed from the historical CLI command and documented Task Scheduler XML.
// Native Windows export/registration acceptance remains a separate requirement.
const sid = "S-1-5-21-123-456-789-1001"
const source = `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Date>2026-09-10T12:00:00</Date><Author>HOST\\user</Author><URI>\\MagnitudeInference</URI></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>${sid}</UserId><LogonType>Password</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>false</StartWhenAvailable><RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable><IdleSettings><Duration>PT10M</Duration><WaitTimeout>PT1H</WaitTimeout><StopOnIdleEnd>true</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings><AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>false</Hidden><RunOnlyIfIdle>false</RunOnlyIfIdle><WakeToRun>false</WakeToRun><ExecutionTimeLimit>PT0S</ExecutionTimeLimit><Priority>7</Priority><RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure></Settings>
  <Actions Context="Author"><Exec><Command>C:\\Users\\user\\Magnitude &amp; 模型\\magnitude-service.exe</Command><Arguments>serve</Arguments></Exec></Actions>
</Task>`
const snapshot = (xml = source) => Schema.decodeUnknownSync(WindowsLegacyTaskSnapshot)({ _tag: "Registered", xml, currentUserSid: sid })
const inspect = (xml = source) => Effect.runPromise(inspectLegacyWindowsStartup(snapshot(xml)))

describe("Windows startup retirement adapter", () => {
  it("preserves the inspected preference and coalesces concurrent missing replay", async () => {
    let current = snapshot()
    const retired: string[] = []
    await Effect.runPromise(Effect.gen(function* () {
      const adapter = yield* makeWindowsLegacyStartup({
        query: Effect.sync(() => current),
        retire: registration => Effect.sleep("1 millis").pipe(Effect.zipRight(Effect.sync(() => {
          retired.push(registration.digest)
          current = { _tag: "Missing" }
        }))),
      })
      const registration = Option.getOrThrow(yield* adapter.inspect)
      expect(registration.enabled).toBe(true)
      yield* Effect.all([adapter.unregister(registration), adapter.unregister(registration)], { concurrency: "unbounded" })
      expect(Option.isNone(yield* adapter.inspect)).toBe(true)
    }))
    expect(retired).toEqual([createHash("sha256").update(source).digest("hex")])
  })

  it("refuses an edited registration without invoking native retirement", async () => {
    const registration = Option.getOrThrow(await inspect())
    let retired = false
    const result = await Effect.runPromise(Effect.gen(function* () {
      const adapter = yield* makeWindowsLegacyStartup({
        query: Effect.succeed(snapshot(source.replace("<Author>HOST\\user</Author>", "<Author>Changed</Author>"))),
        retire: () => Effect.sync(() => { retired = true }),
      })
      return yield* Effect.either(adapter.unregister(registration))
    }))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("changed after migration")
    expect(retired).toBe(false)
  })

  it("propagates native retirement failure without acknowledging completion", async () => {
    const registration = Option.getOrThrow(await inspect())
    const failure = new LegacyStartupFailed({ message: "scheduler denied deletion" })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const adapter = yield* makeWindowsLegacyStartup({ query: Effect.succeed(snapshot()), retire: () => Effect.fail(failure) })
      return yield* Effect.either(adapter.unregister(registration))
    }))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left).toBe(failure)
  })
})

describe("legacy Windows startup recognition", () => {
  it("recognizes the native historical schtasks export with omitted defaults", async () => {
    const xml = readFileSync(new URL("./fixtures/windows-historical-task.xml", import.meta.url), "utf8")
    const task = Option.getOrThrow(await inspect(xml))
    expect(task.enabled).toBe(false)
    expect(task.executable).toBe("C:\\Users\\user\\Magnitude\\magnitude-service.exe")
    expect(task.digest).toBe(createHash("sha256").update(xml).digest("hex"))
  })
  it("preserves the current user's enabled registration and exact XML digest", async () => {
    const task = Option.getOrThrow(await inspect())
    expect(task).toEqual({ _tag: "WindowsScheduledTask", task: "\\MagnitudeInference", enabled: true,
      executable: "C:\\Users\\user\\Magnitude & 模型\\magnitude-service.exe", principalSid: sid,
      digest: createHash("sha256").update(source).digest("hex") })
  })
  it("preserves a disabled task instead of treating it as absent", async () => {
    const changed = source.replace("<AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled>", "<AllowStartOnDemand>true</AllowStartOnDemand><Enabled>false</Enabled>")
    const task = Option.getOrThrow(await inspect(changed))
    expect(task.enabled).toBe(false)
    expect(task.digest).not.toBe(Option.getOrThrow(await inspect()).digest)
  })
  it("recognizes the registration-only policy and an interactive-token task", async () => {
    const changed = source.replace("<RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>", "")
      .replace("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>", "<ExecutionTimeLimit>PT72H</ExecutionTimeLimit>")
      .replace("<LogonType>Password</LogonType>", "<LogonType>InteractiveToken</LogonType>")
    expect(Option.isSome(await inspect(changed))).toBe(true)
  })
  it("accepts explicit absence", async () => {
    expect(await Effect.runPromise(inspectLegacyWindowsStartup({ _tag: "Missing" }))).toEqual(Option.none())
  })
  it.each([
    ["other principal", source.replace(`<UserId>${sid}</UserId>`, "<UserId>S-1-5-18</UserId>")],
    ["elevated principal", source.replace("LeastPrivilege", "HighestAvailable")],
    ["another action context", source.replace('Context="Author"', 'Context="Other"')],
    ["additional action", source.replace("</Actions>", "<Exec><Command>C:\\evil.exe</Command></Exec></Actions>")],
    ["COM handler", source.replace("</Actions>", "<ComHandler><ClassId>x</ClassId></ComHandler></Actions>")],
    ["additional trigger", source.replace("</Triggers>", "<BootTrigger/></Triggers>")],
    ["disabled trigger", source.replace("<LogonTrigger><Enabled>true", "<LogonTrigger><Enabled>false")],
    ["other logon user", source.replace("</LogonTrigger>", "<UserId>S-1-5-18</UserId></LogonTrigger>")],
    ["relative executable", source.replace("C:\\Users\\user\\Magnitude &amp; 模型\\magnitude-service.exe", "magnitude-service.exe")],
    ["shell command", source.replace("magnitude-service.exe", "cmd.exe")],
    ["extra arguments", source.replace("<Arguments>serve</Arguments>", "<Arguments>serve --other</Arguments>")],
    ["working directory", source.replace("</Exec>", "<WorkingDirectory>C:\\other</WorkingDirectory></Exec>")],
    ["unknown setting", source.replace("</Settings>", "<Other>true</Other></Settings>")],
    ["invalid scheduler mode", source.replace("</Settings>", "<UseUnifiedSchedulingEngine>maybe</UseUnifiedSchedulingEngine></Settings>")],
    ["custom delayed logon", source.replace("</LogonTrigger>", "<StartBoundary>2027-09-10T12:00:00</StartBoundary></LogonTrigger>")],
    ["invalid start boundary", source.replace("</LogonTrigger>", "<StartBoundary>not-a-date</StartBoundary></LogonTrigger>")],
    ["incomplete restart", source.replace("<Count>999</Count>", "")],
    ["foreign namespace", source.replace("<Arguments>", '<Arguments xmlns="urn:foreign">')],
    ["unknown attribute", source.replace("<Arguments>", '<Arguments ignored="true">')],
    ["duplicate field", source.replace("</Exec>", "<Arguments>serve</Arguments></Exec>")],
    ["malformed XML", source.replace("</Actions>", "")],
    ["external entity", source.replace("<Task version", '<!DOCTYPE Task [<!ENTITY x SYSTEM "file:///secret">]><Task version')],
  ])("rejects %s without producing a migration registration", async (_name, xml) => {
    const result = await Effect.runPromise(Effect.either(inspectLegacyWindowsStartup(snapshot(xml))))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("LegacyStartupFailed")
  })
})
