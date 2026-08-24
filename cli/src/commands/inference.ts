import type { Command as Commander } from "@commander-js/extra-typings"
import { FetchHttpClient } from "@effect/platform"
import {
  Inference,
  makeInferenceClient,
  type InferenceClient,
  inferenceClientErrorMessage,
} from "@magnitudedev/sdk"
import { Effect } from "effect"

const printJson = (value: unknown) => Effect.sync(() => {
  process.stdout.write(`${JSON.stringify(value, (_key, member) => {
    if (typeof member !== "object" || member === null || member._id !== "Option") return member
    return member._tag === "Some" ? member.value : undefined
  }, 2)}\n`)
})

const explain = inferenceClientErrorMessage

const runInference = <Value>(
  use: (client: InferenceClient) => Effect.Effect<Value, unknown>,
) => Effect.runPromise(Effect.acquireUseRelease(
  makeInferenceClient(),
  (client) => use(client).pipe(Effect.tap(printJson)),
  (client) => client.close,
).pipe(
  Effect.provide(FetchHttpClient.layer),
  Effect.catchAll((error) => Effect.sync(() => {
    process.stderr.write(`${explain(error)}\n`)
    process.exitCode = 1
  })),
  Effect.asVoid,
))

export const registerInferenceCommands = (program: Commander): void => {
  program.command("hardware")
    .description("Show the local inference hardware profile")
    .action(() => runInference((client) => client.query(
      Inference.GetInferenceHardware,
      {},
    )))

  const models = program.command("models").description("Inspect and manage inference models")
  models.command("list")
    .description("List callable models and their installation state")
    .action(() => runInference((client) => client.query(Inference.GetInferenceModels, {})))
  models.command("get")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.query(
      Inference.GetInferenceModel,
      { modelId },
    )))
  models.command("install")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.mutate(
      Inference.InstallInferenceModel,
      { modelId },
    )))
  models.command("uninstall")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.mutate(
      Inference.UninstallInferenceModel,
      { modelId },
    )))
  models.command("load-plan")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.query(
      Inference.PreviewInferenceModelLoad,
      { modelId },
    )))
  models.command("properties")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.query(
      Inference.GetInferenceModelProperties,
      { modelId },
    )))

  const downloads = program.command("downloads").description("Inspect and control model downloads")
  downloads.command("list")
    .action(() => runInference((client) => client.query(Inference.GetInferenceDownloads, {})))
  downloads.command("get")
    .argument("<download-id>", "Exact download occurrence ID")
    .action((downloadId) => runInference((client) => client.query(
      Inference.GetInferenceDownload,
      { downloadId },
    )))
  downloads.command("cancel")
    .argument("<download-id>", "Exact download occurrence ID")
    .action((downloadId) => runInference((client) => client.mutate(
      Inference.CancelInferenceDownload,
      { downloadId },
    )))
  downloads.command("acknowledge-failure")
    .argument("<download-id>", "Exact failed download occurrence ID")
    .action((downloadId) => runInference((client) => client.mutate(
      Inference.AcknowledgeInferenceDownloadFailure,
      { downloadId },
    )))

  const packages = program.command("packages").description("Inspect and remove installed model packages")
  packages.command("list")
    .action(() => runInference((client) => client.query(Inference.GetInferencePackages, {})))
  packages.command("get")
    .argument("<package-id>", "Immutable package ID")
    .action((packageId) => runInference((client) => client.query(
      Inference.GetInferencePackage,
      { packageId },
    )))
  packages.command("remove")
    .argument("<package-id>", "Immutable package ID")
    .action((packageId) => runInference((client) => client.mutate(
      Inference.RemoveInferencePackage,
      { packageId },
    )))

  const instances = program.command("instances").description("Inspect and control loaded model instances")
  instances.command("list")
    .action(() => runInference((client) => client.query(Inference.GetInferenceInstances, {})))
  instances.command("get")
    .argument("<instance-id>", "Exact instance occurrence ID")
    .action((instanceId) => runInference((client) => client.query(
      Inference.GetInferenceInstance,
      { instanceId },
    )))
  instances.command("load")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runInference((client) => client.mutate(
      Inference.EnsureInferenceInstance,
      { modelId },
    )))
  instances.command("stop")
    .argument("<instance-id>", "Exact instance occurrence ID")
    .action((instanceId) => runInference((client) => client.mutate(
      Inference.StopInferenceInstance,
      { instanceId },
    )))
}
