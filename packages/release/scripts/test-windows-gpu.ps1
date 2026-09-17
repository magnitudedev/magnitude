param(
  [Parameter(Mandatory = $true)][ValidateSet('cuda', 'vulkan')][string]$Backend,
  [Parameter(Mandatory = $true)][string]$InstallationDirectory,
  [Parameter(Mandatory = $true)][string]$ModelId,
  [Parameter(Mandatory = $true)][string]$OutputDirectory,
  [string]$Origin = 'http://127.0.0.1:8080',
  [string]$AuthToken = $env:MAGNITUDE_ICN_AUTH_TOKEN
)
# Exercise an already-running, isolated Windows ICN with a small instruction model.
# The caller owns the process and GPU-machine lifetime; this script never provisions resources.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (!$AuthToken) { throw 'MAGNITUDE_ICN_AUTH_TOKEN or -AuthToken is required' }
$InstallationDirectory = [IO.Path]::GetFullPath($InstallationDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$declaration = Get-Content (Join-Path $InstallationDirectory 'installation.json') -Raw | ConvertFrom-Json
if ($declaration.backend -ne $Backend) { throw 'Installation backend does not match the requested test' }
$headers = @{Authorization="Bearer $AuthToken"}
$results=New-Object System.Collections.Generic.List[object]
function Record($name,$detail) { $results.Add(@{test=$name;detail=$detail}); Write-Output "PASS $name" }
function Request($method,$path,$body) {
 $requestParameters=@{Uri="$origin$path";Headers=$headers;Method=$method;TimeoutSec=600}
 if ($null -ne $body) {$requestParameters.ContentType='application/json';$requestParameters.Body=($body | ConvertTo-Json -Depth 30 -Compress)}
 Invoke-RestMethod @requestParameters
}
function Payload($text,[int]$tokens=48) { @{model=$ModelId;messages=@(@{role='user';content=$text});temperature=0;seed=42;max_tokens=$tokens;stream=$false} }
function AssertResponse($response) {
 if (!$response.choices -or [string]::IsNullOrWhiteSpace($response.choices[0].message.content) -or $response.usage.completion_tokens -lt 1) { throw 'Generation did not produce text and completion usage' }
 if ($response.choices[0].finish_reason -notin @('stop','length')) { throw 'Unexpected generation finish reason' }
}
try {
 $hardware=Request GET '/api/v1/hardware' $null
 if ($hardware.native_build -ne $declaration.nativeBuild) { throw 'Running engine identity differs from the installation' }
 $domains=@($hardware.memory_domains | Where-Object {@($_.devices | Where-Object {$_.backend -eq $Backend -and $_.kind -ne 'cpu'}).Count -gt 0})
 if ($domains.Count -eq 0) { throw "No $Backend GPU hardware domain" }
 Record 'hardware' $hardware
 for ($i=0;$i -lt 60;$i++) {
  $models=Request GET '/v1/models' $null
  if (@($models.data | Where-Object { $_.id -eq $ModelId }).Count -gt 0) { break }
  Start-Sleep -Seconds 2
 }
 if (!(@($models.data) | Where-Object { $_.id -eq $ModelId })) { throw 'Requested model is not servable' }
 $response=Request POST '/v1/chat/completions' (Payload 'What is the capital of France? Answer in one sentence.')
 AssertResponse $response
 if ($response.choices[0].message.content -notmatch 'Paris') { throw 'Basic factual inference produced the wrong answer' }
 Record 'baseline-generation' $response
 $instances=Request GET '/api/v1/instances' $null
 $ready=@($instances.instances | Where-Object {$_.modelId -eq $ModelId -and $_.lifecycle._tag -eq 'Ready'})
 $allocations=@($ready | ForEach-Object {$_.lifecycle.allocation.memoryDomains} | Where-Object {$_.memoryDomainId -in $domains.id -and $_.modelBytes -gt 0})
 if ($allocations.Count -eq 0) { throw 'Model weights were not allocated on the GPU' }
 Record 'gpu-allocation' $instances
 $modules=@(Get-Process magnitude-inference | ForEach-Object {$_.Modules} | Where-Object {$_.ModuleName -match 'ggml|cublas|cudart'} | Select-Object ModuleName,FileName -Unique)
 if (!($modules | Where-Object {$_.ModuleName -eq "ggml-$Backend.dll" -and $_.FileName -like "$InstallationDirectory\backends\*"})) { throw 'Expected owned GPU module was not loaded' }
 Record 'loaded-modules' $modules
 for ($i=0;$i -lt 8;$i++) {
  $response=Request POST '/v1/chat/completions' (Payload "Write one sentence about the number $($i+1).")
  AssertResponse $response
  Record "repeated-generation-$i" $response
 }
 $longText=('The orchard has apples, pears, and peaches. The gardener waters the trees every morning. ' * 80)+'Summarize this description in one sentence.'
 $response=Request POST '/v1/chat/completions' (Payload $longText)
 AssertResponse $response
 if ($response.usage.prompt_tokens -lt 1000) { throw 'Long prefill did not exercise at least 1000 tokens' }
 Record 'long-prefill' $response
 $body=Payload 'Count from one to ten.' 80;$body.stream=$true
 $stream=Invoke-WebRequest "$origin/v1/chat/completions" -Headers $headers -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 20) -UseBasicParsing -TimeoutSec 600
 $streamText = ''
 foreach ($line in ($stream.Content -split "`n")) {
  if ($line.StartsWith('data: {')) {
   $chunk = $line.Substring(6) | ConvertFrom-Json
   foreach ($choice in $chunk.choices) { $streamText += $choice.delta.content }
  }
 }
 if ($stream.Content -notmatch 'data: \[DONE\]' -or [string]::IsNullOrWhiteSpace($streamText)) { throw 'Streaming response is incomplete' }
 Record 'streaming' $stream.Content
 $jobs=@(1..3 | ForEach-Object {
  Start-Job -ArgumentList $origin,$headers,(Payload "Say hello in one sentence, request $_.") -ScriptBlock {
   param($origin,$headers,$body)
   $ErrorActionPreference='Stop'
   Invoke-RestMethod "$origin/v1/chat/completions" -Headers $headers -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 20) -TimeoutSec 600
  }
 })
 try {
  $jobs | Wait-Job -Timeout 600 | Out-Null
  foreach($job in $jobs) {
   if ($job.State -ne 'Completed') { throw "Concurrent request failed: $($job.State)" }
   $response=Receive-Job $job -ErrorAction Stop
   AssertResponse $response
   Record "concurrent-$($job.Id)" $response
  }
 } finally { $jobs | Stop-Job; $jobs | Remove-Job -Force }
 $body=Payload 'Write a long story about a forest.' 512;$body.stream=$true;$body.ignore_eos=$true
 $request=[Net.HttpWebRequest]::Create("$origin/v1/chat/completions")
 $request.Method='POST';$request.ContentType='application/json';$request.Headers['Authorization']=$headers.Authorization;$request.Timeout=600000
 $bytes=[Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 20));$request.ContentLength=$bytes.Length
 $writer=$request.GetRequestStream();$writer.Write($bytes,0,$bytes.Length);$writer.Close()
 $http=$request.GetResponse();$reader=New-Object IO.StreamReader($http.GetResponseStream())
 try {
  $receivedToken = $false
  while ($null -ne ($line = $reader.ReadLine())) {
   if ($line.StartsWith('data: {')) {
    $chunk = $line.Substring(6) | ConvertFrom-Json
    if (@($chunk.choices | Where-Object { ![string]::IsNullOrEmpty($_.delta.content) }).Count -gt 0) {
     $receivedToken = $true
     break
    }
   }
  }
  if (!$receivedToken) { throw 'No streaming token before cancellation' }
 }
 finally {$reader.Dispose();$http.Close();$request.Abort()}
 $response=Request POST '/v1/chat/completions' (Payload 'What is two plus two?')
 AssertResponse $response
 Record 'cancellation-recovery' $response
 $invalid=Payload 'Hello';$invalid.max_tokens=-1
 $rejected=$false
 try { Request POST '/v1/chat/completions' $invalid | Out-Null } catch { $rejected=[int]$_.Exception.Response.StatusCode -in @(400,422) }
 if (!$rejected) { throw 'Malformed inference request was not rejected' }
 Record 'invalid-request-rejected' $true
 $instances=Request GET '/api/v1/instances' $null
 foreach($instance in @($instances.instances | Where-Object {$_.modelId -eq $ModelId -and $_.lifecycle._tag -eq 'Ready'})) {
  Request POST "/api/v1/instances/$($instance.id)/stop" $null | Out-Null
 }
 $stopped=$false
 for($i=0;$i -lt 60;$i++) {
  $instances=Request GET '/api/v1/instances' $null
  if (@($instances.instances | Where-Object {$_.modelId -eq $ModelId -and $_.lifecycle._tag -in @('Ready','Loading','Stopping')}).Count -eq 0) {$stopped=$true;break}
  Start-Sleep -Seconds 1
 }
 if (!$stopped) { throw 'Model unload timed out' }
 Record 'model-unload' $instances
 $response=Request POST '/v1/chat/completions' (Payload 'What is the capital of France? Answer in one sentence.')
 AssertResponse $response
 Record 'model-reload' $response
 Request GET '/health' $null | Out-Null
 Record 'healthy-after-suite' $true
} finally {
 $results | ConvertTo-Json -Depth 60 | Set-Content (Join-Path $OutputDirectory "$Backend-test-results.json") -Encoding utf8
}
