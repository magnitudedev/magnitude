import { Group } from "@magnitudedev/effect-query";
import { Application } from "./application";
import { Changes } from "./changes";
import { Configuration } from "./configuration";
import { Connection } from "./connection";
import { Connections } from "./connections";
import { HuggingFace } from "./hugging-face";
import { Models } from "./models";
import { Chat } from "./chat";

export const AcnQueries = Group.make({
  Chat,
  Application,
  Connection,
  Connections,
  Models,
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
  HuggingFace,
  Models,
};
