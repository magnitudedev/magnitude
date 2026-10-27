import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import { SkillListEntry, SkillContent } from "../schemas/skills"
import { OperationFailed } from "../errors"

const ListSkills = Rpc.make("ListSkills", {
  payload: Schema.Struct({ cwd: Schema.String }),
  success: Schema.Array(SkillListEntry),
  error: OperationFailed,
}).pipe(replaySafe)

const GetSkill = Rpc.make("GetSkill", {
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: SkillContent,
  error: OperationFailed,
}).pipe(replaySafe)

export const Skills = {
  listSkills: ListSkills,
  getSkill: GetSkill,
}
