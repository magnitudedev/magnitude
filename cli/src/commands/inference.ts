import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./inference-runtime")

export const registerInferenceCommands = (program: Command): void => {
  program.command("hardware")
    .description("Show hardware and memory available to local inference")
    .action(() => loadRuntime().then(({ showHardware }) => showHardware()))

  const catalog = program.command("catalog").description("Find, compare, and download local models")
  catalog.command("status")
    .description("Show cached Hugging Face discovery and hardware assessment progress")
    .action(() => loadRuntime().then(({ showCatalogStatus }) => showCatalogStatus()))
  catalog.command("list")
    .description("List catalog models compatible with this computer")
    .action(() => loadRuntime().then(({ showModelCatalog }) => showModelCatalog()))
  catalog.command("show")
    .description("Show details for one catalog model")
    .argument("<model-id>", "Canonical model ID from `magnitude catalog list`")
    .action((modelId) => loadRuntime().then(({ showCatalogModel }) => showCatalogModel(modelId)))
  catalog.command("recommendations")
    .description("Rank compatible models for a speed-to-intelligence preference")
    .option("--preference <value>", "Fastest, Faster, Balanced, Smarter, or Smartest", "balanced")
    .option("--limit <count>", "Maximum recommendations to show", "10")
    .action((options) => loadRuntime().then(({ showRecommendations }) =>
      showRecommendations(options.preference, options.limit)))
  catalog.command("pull")
    .description("Download or update a catalog model")
    .argument("<model-id>", "Canonical model ID from `magnitude catalog list`")
    .action((modelId) => loadRuntime().then(({ pullModel }) => pullModel(modelId)))
  catalog.command("cancel")
    .description("Cancel an active model download or update")
    .argument("<model-id>", "Canonical model ID from `magnitude models status`")
    .action((modelId) => loadRuntime().then(({ cancelDownload }) => cancelDownload(modelId)))
  catalog.command("remove")
    .description("Remove an installed catalog model from this computer")
    .argument("<model-id>", "Canonical model ID from `magnitude models status`")
    .action((modelId) => loadRuntime().then(({ removeModel }) => removeModel(modelId)))

  const models = program.command("models").description("Inspect and control local models on this computer")
  models.command("status")
    .description("Show local model installation and runtime status")
    .argument("[model-id]", "Canonical model ID from `magnitude models status`")
    .action((modelId) => loadRuntime().then(({ showModelsStatus }) =>
      showModelsStatus(modelId)))
  models.command("load")
    .description("Load a local model for inference")
    .argument("<model-id>", "Canonical model ID from `magnitude models status`")
    .action((modelId) => loadRuntime().then(({ loadInstance }) =>
      loadInstance(modelId)))
  models.command("stop")
    .description("Stop the active local model")
    .action(() => loadRuntime().then(({ stopInstance }) => stopInstance()))
}
