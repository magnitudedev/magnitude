import { createHash } from "node:crypto"
import { win32 } from "node:path"
import { DOMParser, type Element } from "@xmldom/xmldom"
import { Effect, Option, Schema } from "effect"
import { LegacyStartupFailed } from "./legacy-startup-command"
import { WindowsLegacyTaskSnapshot } from "./windows-task-query"

const namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
export const LegacyWindowsStartup = Schema.TaggedStruct("WindowsScheduledTask", {
  task: Schema.Literal("\\MagnitudeInference"),
  digest: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("LegacyWindowsStartupDigest")),
  enabled: Schema.Boolean,
  executable: Schema.NonEmptyString,
  principalSid: Schema.String.pipe(Schema.pattern(/^S-1-(?:\d+-)+\d+$/), Schema.brand("WindowsUserSid")),
})
export type LegacyWindowsStartup = typeof LegacyWindowsStartup.Type
const failed = (message: string) => new LegacyStartupFailed({ message })

export const makeWindowsLegacyStartup = (control: {
  readonly query: Effect.Effect<typeof WindowsLegacyTaskSnapshot.Type, LegacyStartupFailed>
  readonly retire: (registration: LegacyWindowsStartup) => Effect.Effect<void, LegacyStartupFailed>
}) => Effect.gen(function* () {
  const mutation = yield* Effect.makeSemaphore(1)
  const inspect = control.query.pipe(Effect.flatMap(inspectLegacyWindowsStartup))
  const unregister = (expected: LegacyWindowsStartup) => mutation.withPermits(1)(Effect.gen(function* () {
    const current = yield* inspect
    if (Option.isNone(current)) return
    const observed = current.value
    if (observed.digest !== expected.digest || observed.principalSid !== expected.principalSid ||
        observed.executable !== expected.executable || observed.enabled !== expected.enabled || observed.task !== expected.task) {
      return yield* failed("Legacy Windows startup registration changed after migration was prepared")
    }
    yield* control.retire(expected)
  }))
  return { inspect, unregister }
})

/** The XML boundary preserves paths/attributes and rejects ambiguity instead of repairing input. */
const decodeXml = (source: string) => Effect.try({
  try: () => {
    if (source.length > 65536 || /<!DOCTYPE|<!ENTITY/i.test(source) || source.includes("\0")) throw new Error("Unsafe XML")
    const doc = new DOMParser({ onError: () => { throw new Error("Invalid XML") } }).parseFromString(source, "application/xml")
    const root = doc.documentElement
    if (!root || root.tagName !== "Task" || doc.doctype) throw new Error("Invalid task root")
    const fields = new Map<string, string>()
    const elements = new Set<string>()
    const walk = (element: Element, path: string, depth: number) => {
      if (depth > 8 || elements.size >= 256 || elements.has(path) || element.namespaceURI !== namespace) throw new Error("Ambiguous task structure")
      elements.add(path)
      for (let index = 0; index < element.attributes.length; index++) {
        const attribute = element.attributes.item(index)!
        fields.set(`${path}@${attribute.name}`, attribute.value)
      }
      let content = "", nested = false
      for (let child = element.firstChild; child !== null; child = child.nextSibling) {
        if (child.nodeType === 1) { nested = true; walk(child as Element, `${path}/${child.nodeName}`, depth + 1) }
        else if (child.nodeType === 3 || child.nodeType === 4) content += child.nodeValue ?? ""
        else if (child.nodeType !== 8) throw new Error("Unsupported XML node")
      }
      if (nested && content.trim()) throw new Error("Mixed task content")
      if (!nested) fields.set(path, content.trim())
    }
    walk(root, "Task", 0)
    return { fields, elements }
  },
  catch: () => failed("Legacy Windows task contains malformed, ambiguous, or unsupported XML"),
})

const boolean = (value: string) => value === "true" || value === "false"
const oneOf = (...values: readonly string[]) => (value: string) => values.includes(value)
const anyText = (_value: string) => true
const taskDateTime = Schema.String.pipe(Schema.pattern(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$/),
  Schema.filter(value => Number.isFinite(Date.parse(value))))
const settings = new Map<string, (value: string) => boolean>([
  ["MultipleInstancesPolicy", oneOf("IgnoreNew")],
  ["DisallowStartIfOnBatteries", boolean], ["StopIfGoingOnBatteries", boolean],
  ["AllowHardTerminate", oneOf("true")], ["StartWhenAvailable", oneOf("false")],
  ["RunOnlyIfNetworkAvailable", oneOf("false")],
  ["IdleSettings/Duration", oneOf("PT10M")], ["IdleSettings/WaitTimeout", oneOf("PT1H")],
  ["IdleSettings/StopOnIdleEnd", oneOf("true")], ["IdleSettings/RestartOnIdle", oneOf("false")],
  ["AllowStartOnDemand", oneOf("true")], ["Enabled", boolean], ["Hidden", oneOf("false")],
  ["RunOnlyIfIdle", oneOf("false")], ["WakeToRun", oneOf("false")],
  ["ExecutionTimeLimit", oneOf("PT72H", "PT0S")], ["Priority", oneOf("7")],
  ["RestartOnFailure/Interval", oneOf("PT1M")], ["RestartOnFailure/Count", oneOf("999")],
  ["UseUnifiedSchedulingEngine", boolean], ["DisallowStartOnRemoteAppSession", oneOf("false")],
])

/** This validates a snapshot only. It grants no authority to kill a process or delete a task. */
export const inspectLegacyWindowsStartup = (snapshot: typeof WindowsLegacyTaskSnapshot.Type) => Effect.gen(function* () {
  if (snapshot._tag === "Missing") return Option.none<LegacyWindowsStartup>()
  const { fields, elements } = yield* decodeXml(snapshot.xml)
  const principalId = fields.get("Task/Principals/Principal@id")
  const allowed = new Map<string, (value: string) => boolean>([
    ["Task@xmlns", oneOf(namespace)], ["Task@version", oneOf("1.2", "1.3", "1.4")],
    ["Task/RegistrationInfo/Date", anyText], ["Task/RegistrationInfo/Author", anyText],
    ["Task/RegistrationInfo/URI", oneOf("\\MagnitudeInference")],
    ["Task/Principals/Principal@id", value => value.length > 0],
    ["Task/Principals/Principal/UserId", oneOf(snapshot.currentUserSid)],
    ["Task/Principals/Principal/LogonType", oneOf("Password", "InteractiveToken")],
    ["Task/Principals/Principal/RunLevel", oneOf("LeastPrivilege")],
    ["Task/Triggers/LogonTrigger/Enabled", oneOf("true")],
    ["Task/Triggers/LogonTrigger/StartBoundary", value => Schema.is(taskDateTime)(value) &&
      value.slice(0, 16) === fields.get("Task/RegistrationInfo/Date")?.slice(0, 16)],
    ["Task/Triggers/LogonTrigger/UserId", oneOf(snapshot.currentUserSid)],
    ["Task/Actions@Context", value => value === principalId],
    ["Task/Actions/Exec/Command", anyText],
    ["Task/Actions/Exec/Arguments", oneOf("serve")],
    ["Task/Actions/Exec/WorkingDirectory", oneOf("")],
    ...[...settings].map(([key, predicate]) => [`Task/Settings/${key}`, predicate] as const),
  ])
  const containers = new Set(["Task", "Task/RegistrationInfo", "Task/Principals", "Task/Principals/Principal", "Task/Triggers", "Task/Triggers/LogonTrigger", "Task/Actions", "Task/Actions/Exec", "Task/Settings", "Task/Settings/IdleSettings", "Task/Settings/RestartOnFailure"])
  for (const [path, value] of fields) {
    // An empty generated container has no semantic leaf value.
    if (containers.has(path) && value === "") continue
    if (!allowed.get(path)?.(value)) return yield* failed(`Legacy Windows task has an unrecognized setting: ${path}`)
  }
  for (const path of elements) if (!containers.has(path) && !allowed.has(path)) return yield* failed(`Legacy Windows task has an unrecognized element: ${path}`)
  // Native schtasks /RL LIMITED exports may omit the default LeastPrivilege run level.
  for (const required of ["Task/Principals/Principal@id", "Task/Principals/Principal/UserId", "Task/Principals/Principal/LogonType", "Task/Actions/Exec/Command", "Task/Actions/Exec/Arguments"]) {
    if (!fields.has(required)) return yield* failed(`Legacy Windows task is missing ${required}`)
  }
  if (!elements.has("Task/Triggers/LogonTrigger")) return yield* failed("Legacy Windows task is not a logon registration")
  const command = fields.get("Task/Actions/Exec/Command")!
  const executable = command.startsWith('"') && command.endsWith('"') ? command.slice(1, -1) : command
  if (!/^[A-Za-z]:[\\/]/.test(executable) || /["\r\n%]/.test(executable) || win32.basename(executable).toLowerCase() !== "magnitude-service.exe") {
    return yield* failed("Legacy Windows task does not directly execute a local Magnitude service")
  }
  const restart = elements.has("Task/Settings/RestartOnFailure")
  if (restart && (!fields.has("Task/Settings/RestartOnFailure/Interval") || !fields.has("Task/Settings/RestartOnFailure/Count"))) return yield* failed("Legacy Windows task has incomplete restart settings")
  return Option.some(LegacyWindowsStartup.make({
    task: "\\MagnitudeInference", executable, principalSid: snapshot.currentUserSid,
    digest: LegacyWindowsStartup.fields.digest.make(createHash("sha256").update(snapshot.xml).digest("hex")),
    enabled: fields.get("Task/Settings/Enabled") !== "false",
  }))
})
