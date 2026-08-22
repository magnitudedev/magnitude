import { Schema } from "effect"
import { Group, Query } from "@magnitudedev/effect-query"
import { SkillListEntry, SkillContent } from "../schemas/skills"
import { SessionError } from "../errors"

const ListSkills = Query.make("ListSkills", {
  payload: Schema.Struct({ cwd: Schema.String }),
  success: Schema.Array(SkillListEntry),
  error: SessionError,
})

const GetSkill = Query.make("GetSkill", {
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: SkillContent,
  error: SessionError,
})

export const Skills = Group.make({ ListSkills, GetSkill })
