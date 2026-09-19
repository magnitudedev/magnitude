import { Effect } from "effect"
import { createServer, type AddressInfo, type Server } from "node:net"
import { InfrastructureFailure } from "./domain"

/** Occupy only the isolated test service port; never terminate another port owner. */
export const occupyServicePort = (port: number) => Effect.acquireRelease(
  Effect.async<Server, InfrastructureFailure>(resume => {
    const server = createServer(socket => { socket.destroy() })
    server.once("error", () => resume(Effect.fail(new InfrastructureFailure({ operation: "port-fault", message: "Could not reserve the isolated service port" }))))
    server.listen({ port, host: "127.0.0.1", exclusive: true }, () => resume(Effect.succeed(server)))
    return Effect.sync(() => { server.close() })
  }),
  server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }).pipe(
    Effect.timeoutFail({ duration: "5 seconds", onTimeout: () => new InfrastructureFailure({ operation: "port-fault", message: "Could not release the isolated service port" }) }), Effect.orDie,
  ),
).pipe(Effect.map(server => (server.address() as AddressInfo).port))
