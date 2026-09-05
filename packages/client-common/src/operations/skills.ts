import { Skills as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { query } from "./bind";

const ListSkills = query(Rpcs.listSkills, (client) => client.skills.listSkills);

const GetSkill = query(Rpcs.getSkill, (client) => client.skills.getSkill);

export const Skills = Group.make({ ListSkills, GetSkill });
