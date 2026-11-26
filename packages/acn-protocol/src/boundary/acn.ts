import { rpcGroup } from "../rpc-tree"
import { Connection } from "./connection"
import { Models } from "./models"
import { Changes } from "./changes"
import { Configuration } from "./configuration"
import { Connections } from "./connections"
import { Application } from "./application"
import { HuggingFace } from "./hugging-face"
import { Chat } from "./chat"

export const MagnitudeRpcs = {
  chat: Chat,
  connection: Connection,
  models: Models,
  changes: Changes,
  configuration: Configuration,
  connections: Connections,
  application: Application,
  huggingFace: HuggingFace,
}
export const AcnRpcGroup = rpcGroup(MagnitudeRpcs)
