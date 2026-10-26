import { Group } from "@magnitudedev/effect-query";
import { Application } from "./application";
import { Changes } from "./changes";
import { Configuration } from "./configuration";
import { Connection } from "./connection";
import { Connections } from "./connections";
import { Files } from "./files";
import { Git } from "./git";
import { HuggingFace } from "./hugging-face";
import { Models } from "./models";
import { ProjectFiles } from "./project-files";
import { Projects } from "./projects";
import { Chat } from "./chat";
import { Skills } from "./skills";

export const AcnQueries = Group.make({
  Chat,
  Projects,
  Application,
  Connection,
  Connections,
  ProjectFiles,
  Skills,
  Models,
  Git,
  Files,
  Changes,
  Configuration,
  HuggingFace,
});
export {
  Application,
  Changes,
  Configuration,
  Connection,
  Connections,
  Files,
  Git,
  HuggingFace,
  Models,
  ProjectFiles,
  Projects,
  Skills,
};
