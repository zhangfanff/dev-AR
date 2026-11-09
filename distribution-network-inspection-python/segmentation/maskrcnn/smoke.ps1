param([string]$RunDir = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $RunDir) { $RunDir = 'runs/smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$taskSmokeOutput = 'results/' + (Split-Path -Leaf $RunDir)
& 'G:/anaconda/envs/yolo12-common/python.exe' train.py --smoke --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Smoke training failed with exit code $LASTEXITCODE" }
& 'G:/anaconda/envs/yolo12-common/python.exe' eval.py --split test --trained-model "$RunDir/checkpoint_best_mask.pth" --max-images 3 --output-dir $taskSmokeOutput
if ($LASTEXITCODE -ne 0) { throw "Smoke evaluation failed with exit code $LASTEXITCODE" }
