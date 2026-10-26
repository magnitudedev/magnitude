import { rpcGroup } from "../rpc-tree"
import { Projects } from "./projects"
import { Connection } from "./connection"
import { ProjectFiles } from "./project-files"
import { Skills } from "./skills"
import { Models } from "./models"
import { Git } from "./git"
import { Files } from "./files"
import { Changes } from "./changes"
import { Configuration } from "./configuration"
import { Connections } from "./connections"
import { Application } from "./application"
import { HuggingFace } from "./hugging-face"
import { Chat } from "./chat"

export const MagnitudeRpcs = {
  chat: Chat,
  projects: Projects,
  connection: Connection,
  projectFiles: ProjectFiles,
  skills: Skills,
  models: Models,
  git: Git,
  files: Files,
  changes: Changes,
  configuration: Configuration,
  connections: Connections,
  application: Application,
  huggingFace: HuggingFace,
}
export const AcnRpcGroup = rpcGroup(MagnitudeRpcs)
