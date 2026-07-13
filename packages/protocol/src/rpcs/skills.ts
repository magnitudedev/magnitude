import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SkillListEntry, SkillContent } from "../schemas/skills"
import { SessionError } from "../errors"

export const ListSkills = Rpc.make("ListSkills", {
  payload: Schema.Struct({ cwd: Schema.String }),
  success: Schema.Array(SkillListEntry),
  error: SessionError
})

export const GetSkill = Rpc.make("GetSkill", {
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: SkillContent,
  error: SessionError
})
