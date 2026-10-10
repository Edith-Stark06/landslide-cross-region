# Stage the code dataset for Kaggle (src, splits, configs, jobs.json, GIT_HASH) in .cache/code_pkg.
# Usage: powershell -File scripts/pack_code.ps1 -Jobs '[{"config":"configs/smoke_l4s_unet.yaml","seeds":[0]}]'
param([Parameter(Mandatory=$true)][string]$Jobs)
$root = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $root ".cache\code_pkg"
if (Test-Path $pkg) { Remove-Item $pkg -Recurse -Force }
New-Item -ItemType Directory -Force -Path $pkg | Out-Null
foreach ($d in 'src','splits','configs') { Copy-Item (Join-Path $root $d) (Join-Path $pkg $d) -Recurse }
Get-ChildItem $pkg -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Set-Content (Join-Path $pkg "jobs.json") $Jobs -Encoding ascii
$hash = (git -C $root rev-parse HEAD 2>$null); $dirty = (git -C $root status --porcelain 2>$null)
if (-not $hash) { $hash = "unknown" } elseif ($dirty) { $hash = "$hash-dirty" }
Set-Content (Join-Path $pkg "GIT_HASH") $hash -Encoding ascii
$meta = [ordered]@{ title="Landslide XR code"; id="edithstark/landslide-xr-code"; licenses=@(@{name="other"}) }
($meta | ConvertTo-Json -Depth 4) | Set-Content (Join-Path $pkg "dataset-metadata.json") -Encoding ascii
"staged $pkg  (revision $hash)"
