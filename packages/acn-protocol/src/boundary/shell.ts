import { Rpc } from "@effect/rpc"
import { atMostOnce } from "../transport/recovery"
import { RunBashPayload, RunBashResult } from "../schemas/shell"
import { SessionError } from "../errors"

const RunBash = Rpc.make("RunBash", {
  payload: RunBashPayload,
  success: RunBashResult,
  error: SessionError,
}).pipe(atMostOnce)

export const Shell = {
  runBash: RunBash,
}
