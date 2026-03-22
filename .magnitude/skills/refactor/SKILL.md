---
name: refactor
description: Safely refactor code with test-verified behavior parity
---

<phase name="scope">
  <submit>
    <file name="scope_doc" type="md" description="Document describing refactor scope and approach"/>
    <file name="test_script" type="sh" description="Script that runs the relevant tests and captures results"/>
  </submit>
</phase>

Identify the scope of this refactor. Explore the relevant code and determine:
1. What exactly is being refactored and why
2. What tests exist that cover this code
3. Whether additional tests are needed to verify behavior parity

If additional tests are needed, write them now before proceeding.

Write a test script that runs the relevant tests and captures baseline results in whatever format is appropriate.

<phase name="baseline">
  <submit>
    <file name="baseline_results" description="Baseline test results"/>
  </submit>
  <criteria>
    <shell-succeed name="baseline">bash {{scope.test_script}}</shell-succeed>
  </criteria>
</phase>

Run the test script to establish the green baseline. Verify that all expected tests are passing. If tests are failing, fix them before proceeding.

<phase name="refactor">
  <submit>
    <file name="verify_script" type="sh" description="Script that verifies behavior parity against baseline. Must exit 0 if parity holds."/>
  </submit>
  <criteria>
    <shell-succeed name="parity">bash {{refactor.verify_script}}</shell-succeed>
  </criteria>
</phase>

Execute the refactor described in `{{scope.scope_doc}}`.

Keep changes focused and mechanical where possible. Do not change behavior — only structure.

<phase name="verify">
  <criteria>
    <shell-succeed name="parity">bash {{refactor.verify_script}}</shell-succeed>
    <agent-approval name="review" subagent="reviewer">
      Review the refactored code for quality, readability, and adherence to the scope defined in {{scope.scope_doc}}.
    </agent-approval>
  </criteria>
</phase>

Review the refactored code for quality issues. Ensure the refactor is clean, follows existing patterns, and the test suite still passes.
