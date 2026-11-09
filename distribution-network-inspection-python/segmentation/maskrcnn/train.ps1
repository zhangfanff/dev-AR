param(
    [int]$Epochs = 30,
    [string]$RunDir = 'runs/wire700',
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskTrainArgs = @('train.py', '--epochs', $Epochs, '--run-dir', $RunDir)
if ($Resume) { $taskTrainArgs += @('--resume', $Resume) }
& 'G:/anaconda/envs/yolo12-common/python.exe' @taskTrainArgs
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
