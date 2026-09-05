import { rpcGroup } from "../rpc-tree"
import { Projects } from "./projects"
import { Agent } from "./agent"
import { Connection } from "./connection"
import { Display } from "./display"
import { Shell } from "./shell"
import { Onboarding } from "./onboarding"
import { ProjectFiles } from "./project-files"
import { Skills } from "./skills"
import { Models } from "./models"
import { Git } from "./git"
import { Files } from "./files"
import { Changes } from "./changes"
import { Sessions } from "./sessions"
import { Configuration } from "./configuration"

export const MagnitudeRpcs = {
  projects: Projects,
  agent: Agent,
  connection: Connection,
  display: Display,
  shell: Shell,
  onboarding: Onboarding,
  projectFiles: ProjectFiles,
  skills: Skills,
  models: Models,
  git: Git,
  files: Files,
  changes: Changes,
  sessions: Sessions,
  configuration: Configuration,
}
export const AcnRpcGroup = rpcGroup(MagnitudeRpcs)
