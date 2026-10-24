import { Chat } from "./chat";
import { Group } from "@magnitudedev/effect-query";
import { Agent } from "./agent";
import { Application } from "./application";
import { Changes } from "./changes";
import { Configuration } from "./configuration";
import { Connection } from "./connection";
import { Connections } from "./connections";
import { Display } from "./display";
import { Files } from "./files";
import { Git } from "./git";
import { HuggingFace } from "./hugging-face";
import { Models } from "./models";
import { ProjectFiles } from "./project-files";
import { Projects } from "./projects";
import { Sessions } from "./sessions";
import { Shell } from "./shell";
import { Skills } from "./skills";

export const AcnQueries = Group.make({
  Chat,
  Projects,
  Agent,
  Application,
  Connection,
  Connections,
  Display,
  Shell,
  ProjectFiles,
  Skills,
  Models,
  Git,
  Files,
  Changes,
  Sessions,
  Configuration,
  HuggingFace,
});
export {
  Agent,
  Application,
  Changes,
  Configuration,
  Connection,
  Connections,
  Display,
  Files,
  Git,
  HuggingFace,
  Models,
  ProjectFiles,
  Projects,
  Sessions,
  Shell,
  Skills,
};
