# Resumable Zenodo downloader with MD5 verification.
# Usage: powershell -File zenodo_get.ps1 -Record 10463239 -Dest F:\LandslideData\raw\landslide4sense [-Skip "Future work.zip"]
param(
  [Parameter(Mandatory=$true)][string]$Record,
  [Parameter(Mandatory=$true)][string]$Dest,
  [string[]]$Skip = @()
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$meta = Invoke-RestMethod "https://zenodo.org/api/records/$Record" -TimeoutSec 60
$meta | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $Dest "_zenodo_record_$Record.json") -Encoding utf8
foreach ($f in $meta.files) {
  if ($Skip -contains $f.key) { "skip $($f.key)"; continue }
  $out = Join-Path $Dest $f.key
  $ok  = "$out.md5ok"
  if (Test-Path $ok) { "already verified $($f.key)"; continue }
  $url = "https://zenodo.org/api/records/$Record/files/$([uri]::EscapeDataString($f.key))/content"
  for ($try = 1; $try -le 5; $try++) {
    curl.exe -L -C - --fail --retry 5 --retry-delay 5 -o $out $url
    if ((Get-Item $out).Length -ge $f.size) { break }
  }
  $expected = $f.checksum -replace '^md5:', ''
  $actual = (Get-FileHash $out -Algorithm MD5).Hash.ToLower()
  if ($actual -eq $expected) { "md5 OK $($f.key)"; Set-Content $ok $actual }
  else { "MD5 MISMATCH $($f.key): expected $expected got $actual" }
}
"FINISHED record $Record"
