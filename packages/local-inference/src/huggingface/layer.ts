import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Layer } from "effect"
import { HuggingFaceDownload, HuggingFaceHub } from "./contracts"
import { HuggingFaceDownloadLive, type HuggingFaceDownloadOptions } from "./download"
import { HuggingFaceHubLive } from "./hub"
import { StorageCapacityLive } from "./storage-capacity"

export const HuggingFaceLive = (options: HuggingFaceDownloadOptions): Layer.Layer<
  HuggingFaceHub | HuggingFaceDownload,
  never,
  FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
> => Layer.merge(
  HuggingFaceHubLive(options.connection),
  HuggingFaceDownloadLive(options).pipe(Layer.provide(StorageCapacityLive)),
)
