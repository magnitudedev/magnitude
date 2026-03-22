---
name: bug
description: Systematically diagnose, reproduce, and fix a bug
---

<phase name="investigate">
  <submit>
    <file name="analysis" type="md" description="Analysis of involved systems, potential causes, and debugging strategy"/>
  </submit>
</phase>

Understand the involved systems. Gather evidence — logs, stack traces, error messages, user reports. Map out the code paths involved. Form hypotheses about potential root causes and document a debugging strategy.

<phase name="reproduce">
  <submit>
    <file name="repro_test" type="sh" description="Minimal reproduction that demonstrates the bug — must exit non-zero"/>
  </submit>
  <criteria>
    <shell-succeed name="repro">! bash {{reproduce.repro_test}}</shell-succeed>
  </criteria>
</phase>

Isolate the root cause through progressively smaller reproductions. Start broad, then narrow down until you have a minimal, clean test that reliably triggers the bug. The reproduction script must fail (exit non-zero) to confirm the bug exists.

<phase name="fix">
  <criteria>
    <shell-succeed name="verify-fix">bash {{reproduce.repro_test}}</shell-succeed>
  </criteria>
</phase>

Fix the bug. The reproduction test must now pass (exit 0), confirming the fix.

<phase name="verify">
  <criteria>
    <shell-succeed name="verify-fix">bash {{reproduce.repro_test}}</shell-succeed>
    <agent-approval name="review" subagent="reviewer">
      Review the fix for correctness and ensure it addresses the root cause identified in {{investigate.analysis}} without introducing regressions.
    </agent-approval>
  </criteria>
</phase>

Ensure the fix is clean, addresses the actual root cause, and doesn't introduce regressions. Run any broader test suites that cover the affected area.
