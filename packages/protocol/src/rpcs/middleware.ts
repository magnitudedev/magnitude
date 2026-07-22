import { RpcMiddleware } from "@effect/rpc"

export class AcnRpcDemand extends RpcMiddleware.Tag<AcnRpcDemand>()("AcnRpcDemand", {
  wrap: true,
}) {}
