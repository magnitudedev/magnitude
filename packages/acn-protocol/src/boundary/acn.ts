import { Group } from "@magnitudedev/effect-query"
import { Agent } from "./agent"
import { Changes } from "./changes"
import { ClientLease } from "./client-lease"
import { Configuration } from "./configuration"
import { Connection } from "./connection"
import { Display } from "./display"
import { Files } from "./files"
import { Git } from "./git"
import { Models } from "./models"
import { Onboarding } from "./onboarding"
import { ProjectFiles } from "./project-files"
import { Projects } from "./projects"
import { Sessions } from "./sessions"
import { Shell } from "./shell"
import { Skills } from "./skills"

/** The complete client↔ACN boundary, composed solely from domain groups. */
export const AcnBoundary = Group.make({
  Agent,
  Sessions,
  Projects,
  ProjectFiles,
  Connection,
  Configuration,
  Files,
  Git,
  Models,
  Skills,
  Shell,
  Display,
  Onboarding,
  ClientLease,
  Changes,
})
