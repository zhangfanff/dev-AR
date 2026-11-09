param(
    [string]$Checkpoint = 'runs/wire700/checkpoint_best_mask.pth',
    [switch]$ReusePredictions
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskEvalArgs = @('eval.py', '--split', 'test', '--trained-model', $Checkpoint)
if ($ReusePredictions) { $taskEvalArgs += '--reuse-predictions' }
& 'G:/anaconda/envs/yolo12-common/python.exe' @taskEvalArgs
if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed with exit code $LASTEXITCODE" }
