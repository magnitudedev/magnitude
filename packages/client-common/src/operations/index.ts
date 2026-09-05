import { Group } from "@magnitudedev/effect-query";
import { Agent } from "./agent";
import { Changes } from "./changes";
import { Configuration } from "./configuration";
import { Connection } from "./connection";
import { Display } from "./display";
import { Files } from "./files";
import { Git } from "./git";
import { Models } from "./models";
import { Onboarding } from "./onboarding";
import { ProjectFiles } from "./project-files";
import { Projects } from "./projects";
import { Sessions } from "./sessions";
import { Shell } from "./shell";
import { Skills } from "./skills";

export const AcnQueries = Group.make({
  Projects,
  Agent,
  Connection,
  Display,
  Shell,
  Onboarding,
  ProjectFiles,
  Skills,
  Models,
  Git,
  Files,
  Changes,
  Sessions,
  Configuration,
});
export {
  Agent,
  Changes,
  Configuration,
  Connection,
  Display,
  Files,
  Git,
  Models,
  Onboarding,
  ProjectFiles,
  Projects,
  Sessions,
  Shell,
  Skills,
};
