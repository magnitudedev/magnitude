import { Group, Mutation } from "@magnitudedev/effect-query"
import { RunBashPayload, RunBashResult } from "../schemas/shell"
import { SessionError } from "../errors"

const RunBash = Mutation.make("RunBash", {
  policy: { recovery: "AtMostOnce" },
  payload: RunBashPayload,
  success: RunBashResult,
  error: SessionError,
})

export const Shell = Group.make({ RunBash })
