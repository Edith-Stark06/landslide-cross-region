# Finish a Zenodo file using N parallel resumable HTTP range downloads, then join and verify MD5.
# The existing $Dest\$Key is kept as the verified-by-length prefix (bytes 0..len-1).
param(
  [Parameter(Mandatory=$true)][string]$Record,
  [Parameter(Mandatory=$true)][string]$Key,
  [Parameter(Mandatory=$true)][string]$Dest,
  [int]$Parts = 4
)
$ErrorActionPreference = 'Stop'
$meta = Invoke-RestMethod "https://zenodo.org/api/records/$Record" -TimeoutSec 60
$f = $meta.files | Where-Object key -eq $Key
$total = [int64]$f.size
$out = Join-Path $Dest $Key
$prefixFile = "$out.prefix"
if (-not (Test-Path $prefixFile)) { Move-Item $out $prefixFile }   # freeze the prefix once
$s0 = (Get-Item $prefixFile).Length
"prefix bytes: $s0 of $total"
$chunk = [int64][math]::Ceiling(($total - $s0) / $Parts)
$url = "https://zenodo.org/api/records/$Record/files/$([uri]::EscapeDataString($Key))/content"
$procs = @(); $partFiles = @(); $ps = "F:\LandslideData\scripts\part_dl.ps1"
for ($i = 0; $i -lt $Parts; $i++) {
  $a = $s0 + $i * $chunk; $b = [math]::Min($a + $chunk - 1, $total - 1)
  $pf = "$out.part$i"; $partFiles += $pf
  $procs += Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$ps,"-Url",$url,"-A",$a,"-B",$b,"-Out",$pf
}
$procs | ForEach-Object { $_.WaitForExit() }
for ($i = 0; $i -lt $Parts; $i++) {
  $a = $s0 + $i * $chunk; $b = [math]::Min($a + $chunk - 1, $total - 1)
  if ((Get-Item $partFiles[$i]).Length -ne ($b - $a + 1)) { throw "part $i incomplete: rerun this script" }
}
$joined = "$out.joined"
$fs = [IO.File]::Open($joined, 'Create')
foreach ($src in @($prefixFile) + $partFiles) { $in = [IO.File]::OpenRead($src); $in.CopyTo($fs, 4MB); $in.Close() }
$fs.Close()
if ((Get-Item $joined).Length -ne $total) { throw "joined size mismatch" }
$actual = (Get-FileHash $joined -Algorithm MD5).Hash.ToLower()
$expectedMd5 = $f.checksum -replace '^md5:', ''
if ($actual -eq $expectedMd5) {
  Move-Item $joined $out -Force
  Remove-Item $prefixFile; foreach ($pf in $partFiles) { Remove-Item $pf }
  Set-Content "$out.md5ok" $actual
  "md5 OK $Key"
} else { "MD5 MISMATCH $Key expected $expectedMd5 got $actual (kept $joined)" }
