import { RpcMiddleware } from "@effect/rpc"

export class AcnRpcCommandActivity extends RpcMiddleware.Tag<AcnRpcCommandActivity>()(
  "AcnRpcCommandActivity",
  { wrap: true },
) {}
