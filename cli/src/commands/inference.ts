import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./inference-runtime")

export const registerInferenceCommands = (program: Command): void => {
  const catalog = program.command("catalog").description("Discover and acquire local models")
  catalog.command("list")
    .description("Show the unified model catalog")
    .action(() => loadRuntime().then(({ showModelCatalog }) => showModelCatalog()))
  catalog.command("pull")
    .description("Install a local model or bring it up to date")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => loadRuntime().then(({ pullModel }) => pullModel(modelId)))
  catalog.command("remove")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => loadRuntime().then(({ removeModel }) => removeModel(modelId)))
  catalog.command("cancel")
    .argument("<model-id>", "Canonical model ID whose pull to cancel")
    .action((modelId) => loadRuntime().then(({ cancelDownload }) => cancelDownload(modelId)))

  const models = program.command("models").description("Inspect and control model residency")
  models.command("status")
    .action(() => loadRuntime().then(({ listInstances }) => listInstances()))
  models.command("load")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => loadRuntime().then(({ loadInstance }) => loadInstance(modelId)))
  models.command("stop")
    .description("Stop the active local model")
    .action(() => loadRuntime().then(({ stopInstance }) => stopInstance()))
}
