import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import { SkillListEntry, SkillContent } from "../schemas/skills"
import { SessionError } from "../errors"

const ListSkills = Rpc.make("ListSkills", {
  payload: Schema.Struct({ cwd: Schema.String }),
  success: Schema.Array(SkillListEntry),
  error: SessionError,
}).pipe(replaySafe)

const GetSkill = Rpc.make("GetSkill", {
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: SkillContent,
  error: SessionError,
}).pipe(replaySafe)

export const Skills = {
  listSkills: ListSkills,
  getSkill: GetSkill,
}
