import { Rpc } from "@effect/rpc"
import { RunBashPayload, RunBashResult } from "../schemas/shell"
import { SessionError } from "../errors"

export const RunBash = Rpc.make("RunBash", {
  payload: RunBashPayload,
  success: RunBashResult,
  error: SessionError
})
