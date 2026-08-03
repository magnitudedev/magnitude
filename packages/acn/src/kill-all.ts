import { Console, Effect } from "effect"
import { listAcnInstances } from "./daemon-registration"
import { terminatePublishedAcn } from "./peer-acn"

export const killAllAcns = (dataDir: string) =>
  Effect.gen(function* () {
    const instances = yield* listAcnInstances(dataDir)
    if (instances.length === 0) {
      yield* Console.log("No published ACNs found.")
      return
    }

    const results = yield* Effect.forEach(
      instances,
      (instance) =>
        terminatePublishedAcn(instance).pipe(Effect.either),
      { concurrency: "unbounded" },
    )
    yield* Effect.forEach(
      results,
      (result, index) => {
        const instance = instances[index]!
        const label = `${instance.record.version} pid ${instance.record.pid} (${instance.record.id})`
        return result._tag === "Right"
          ? Console.log(`terminated ${label}`)
          : Console.error(`failed ${label}: ${String(result.left)}`)
      },
      { discard: true },
    )
  })
