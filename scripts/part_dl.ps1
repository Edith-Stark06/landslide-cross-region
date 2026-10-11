# Download bytes [A,B] of a URL into $Out, resuming across connection resets (appends whatever curl received).
param([string]$Url,[int64]$A,[int64]$B,[string]$Out)
$expected = $B - $A + 1
for ($try = 1; $try -le 200; $try++) {
  $len = if (Test-Path $Out) { (Get-Item $Out).Length } else { 0 }
  if ($len -ge $expected) { break }
  $tmp = "$Out.tmp"
  curl.exe -L --fail --silent --show-error --range "$($A + $len)-$B" -o $tmp $Url
  if ((Test-Path $tmp) -and ((Get-Item $tmp).Length -gt 0)) {
    $in = [IO.File]::OpenRead($tmp); $fs = [IO.File]::Open($Out, 'Append'); $in.CopyTo($fs, 4MB); $fs.Close(); $in.Close()
  }
  if (Test-Path $tmp) { Remove-Item $tmp }
  Start-Sleep 3
}
