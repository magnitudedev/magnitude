// HTML report generation utility for Magnitude test results

interface HtmlReportTest {
  test: any;
  result: any;
}

export function renderHtmlReport(testResults: HtmlReportTest[]): string {
  const summary = {
    passed: testResults.filter(({ result }) => result.passed).length,
    failed: testResults.filter(({ result }) => !result.passed).length,
    total: testResults.length,
  };
  let html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Magnitude Test Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; }
    h1 { color: #2c3e50; }
    .summary { margin-bottom: 2em; }
    .passed { color: green; }
    .failed { color: red; }
    .test { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 1.5em; padding: 1em; }
    .test-title { font-weight: bold; font-size: 1.2em; }
    .test-status { font-weight: bold; }
    .test-error { color: #c0392b; margin-top: 0.5em; }
    .test-section { margin-top: 0.5em; }
    .step { margin-left: 1em; margin-bottom: 0.5em; }
    .check { margin-left: 1em; margin-bottom: 0.5em; }
    .action { margin-left: 2em; color: #555; }
  </style>
</head>
<body>
  <h1>Magnitude Test Report</h1>
  <div class="summary">
    <div><b>Total:</b> ${summary.total}</div>
    <div class="passed"><b>Passed:</b> ${summary.passed}</div>
    <div class="failed"><b>Failed:</b> ${summary.failed}</div>
  </div>
  <div class="tests">
`;
  for (const { test, result } of testResults) {
    html += `<div class="test">
  <div class="test-title">${test.title}</div>
  <div><b>URL:</b> ${test.url}</div>
  <div class="test-status ${result.passed ? 'passed' : 'failed'}">${result.passed ? 'PASSED' : 'FAILED'}</div>
  ${!result.passed && result.failure ? `<div class="test-error">Error: ${result.failure.message}</div>` : ''}
`;
    if (result.state && result.state.stepsAndChecks) {
      html += `<div class="test-section"><b>Steps & Checks:</b></div>`;
      for (const item of result.state.stepsAndChecks) {
        if (item.variant === 'step') {
          html += `<div class="step"><b>Step:</b> ${item.description} <span class="${item.status}">[${item.status}]</span></div>`;
          if (item.thoughts && item.thoughts.length > 0) {
            html += `<div class="action" style="color:#888;"><b>Thoughts:</b><ul>`;
            for (const thought of item.thoughts) {
              html += `<li>${thought}</li>`;
            }
            html += `</ul></div>`;
          }
          if (item.actions && item.actions.length > 0) {
            for (const action of item.actions) {
              html += `<div class="action">- ${action.pretty}</div>`;
            }
          }
        } else if (item.variant === 'check') {
          html += `<div class="check"><b>Check:</b> ${item.description} <span class="${item.status}">[${item.status}]</span></div>`;
        }
      }
    } else {
      html += `<div class="test-section"><i>No step/check data available.</i></div>`;
    }
    html += `</div>`;
  }
  html += `  </div>\n</body>\n</html>`;
  return html;
} 