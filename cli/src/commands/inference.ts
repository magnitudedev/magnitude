import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./inference-runtime")

export const registerInferenceCommands = (program: Command): void => {
  program.command("hardware")
    .description("Show the local inference environment")
    .action(() => loadRuntime().then(({ showHardware }) => showHardware()))

  const models = program.command("models").description("Inspect and manage models")
  models.command("list")
    .description("Show the unified model catalog")
    .action(() => loadRuntime().then(({ showModelCatalog }) => showModelCatalog()))
  models.command("install")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => loadRuntime().then(({ installModel }) => installModel(modelId)))
  models.command("remove")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => loadRuntime().then(({ removeModel }) => removeModel(modelId)))

  const downloads = program.command("downloads")
    .description("Inspect and control model downloads")
  downloads.command("list")
    .description("Show download state in the unified model catalog")
    .action(() => loadRuntime().then(({ showModelCatalog }) => showModelCatalog()))
  downloads.command("cancel")
    .argument("<model-id>", "Canonical model ID whose download to cancel")
    .action((modelId) => loadRuntime().then(({ cancelDownload }) => cancelDownload(modelId)))
  downloads.command("acknowledge-failure")
    .argument("<model-id>", "Canonical model ID whose failed download to acknowledge")
    .action((modelId) => loadRuntime().then(({ acknowledgeDownloadFailure }) =>
      acknowledgeDownloadFailure(modelId)))

  const instances = program.command("instances")
    .description("Inspect and control model residency")
  instances.command("list")
    .action(() => loadRuntime().then(({ listInstances }) => listInstances()))
  instances.command("load")
    .argument("<slot-id>", "Configured slot ID")
    .action((slotId) => loadRuntime().then(({ loadInstance }) => loadInstance(slotId)))
  instances.command("stop")
    .argument("<slot-id>", "Configured slot ID")
    .action((slotId) => loadRuntime().then(({ stopInstance }) => stopInstance(slotId)))

  program.command("slots")
    .description("Show resolved model slots")
    .action(() => loadRuntime().then(({ listSlots }) => listSlots()))
}
