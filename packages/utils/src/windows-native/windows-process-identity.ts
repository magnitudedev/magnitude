import { Schema } from "effect"

export const WindowsProcessId = Schema.Int.pipe(Schema.between(1, 0xffffffff), Schema.brand("WindowsProcessId"))
export type WindowsProcessId = typeof WindowsProcessId.Type
export const WindowsProcessIdentity = Schema.Struct({
  pid: WindowsProcessId,
  creationTime: Schema.String.pipe(Schema.pattern(/^[0-9a-f]{16}$/), Schema.brand("WindowsProcessCreationTime")),
})
export type WindowsProcessIdentity = typeof WindowsProcessIdentity.Type
