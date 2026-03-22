---
name: feature
description: Plan, approve, build, and review a feature
---

<phase name="explore">
  <submit>
    <file name="findings" type="md" description="Exploration findings covering architecture, patterns, and integration points"/>
  </submit>
</phase>

Use parallelized explorers to understand all code areas relevant to this feature. Map out the architecture, existing patterns, dependencies, and integration points.

<phase name="plan">
  <submit>
    <file name="plan" type="md" description="Detailed implementation plan"/>
  </submit>
  <criteria>
    <user-approval>
      Review the implementation plan. Ready to begin implementation?
    </user-approval>
  </criteria>
</phase>

Based on `{{explore.findings}}`, create a detailed implementation plan. Cover the approach, key decisions, edge cases, testing strategy, and potential risks. Present the plan to the user, discuss key decisions, fill in any missing assumptions, and incorporate feedback. Iterate until the user approves.

<phase name="build">
  <submit>
    <file name="test_script" type="sh" description="Script that runs tests for the new feature"/>
  </submit>
  <criteria>
    <shell-succeed name="tests">bash {{build.test_script}}</shell-succeed>
  </criteria>
</phase>

Implement the plan at `{{plan.plan}}` using parallel builders where possible. Write tests for the new functionality.

<phase name="review">
  <criteria>
    <shell-succeed name="tests">bash {{build.test_script}}</shell-succeed>
    <agent-approval name="review" subagent="reviewer">
      Review the implementation for correctness, code quality, test coverage, and adherence to the plan at {{plan.plan}}.
    </agent-approval>
  </criteria>
</phase>

Address any issues found during review. Ensure the implementation is clean, well-tested, and matches the approved plan.
