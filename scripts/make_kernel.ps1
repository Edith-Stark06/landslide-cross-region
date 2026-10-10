# Create (and optionally push) a private GPU kernel that runs jobs.json from the landslide-xr-code dataset.
# Usage: powershell -File scripts/make_kernel.ps1 -Name landslide-xr-smoke -Datasets l4s [-Push]
param([Parameter(Mandatory=$true)][string]$Name, [Parameter(Mandatory=$true)][string[]]$Datasets, [switch]$Push)
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root ".cache\kernels\$Name"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Copy-Item (Join-Path $root "scripts\kaggle_entry.py") (Join-Path $dir "entry.py") -Force
$sources = @("edithstark/landslide-xr-code") + ($Datasets | ForEach-Object { "edithstark/landslide-xr-$_" })
$meta = [ordered]@{ id="edithstark/$Name"; title=$Name; code_file="entry.py"; language="python"; kernel_type="script";
  is_private=$true; enable_gpu=$true; enable_internet=$true; dataset_sources=$sources; competition_sources=@(); kernel_sources=@() }
($meta | ConvertTo-Json -Depth 4) | Set-Content (Join-Path $dir "kernel-metadata.json") -Encoding ascii
"kernel folder: $dir"
if ($Push) { & (Join-Path $root ".venv\Scripts\kaggle.exe") kernels push -p $dir }
