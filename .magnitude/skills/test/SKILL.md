---
name: test
description: Test workflow skill
---

<phase name="test">
  <submit>
    <file name="test_script" type="sh" description="Script in your workspace that echos hello"/>
  </submit>
  <criteria>
    <shell-succeed name="tests">$SHELL {{test.test_script}}</shell-succeed>
  </criteria>
</phase>

This workflow is a simple test of the workflow system itself.
Make a shell script in your workspace ($M) that does sleep 5 and echo "hello!" and submit it.
