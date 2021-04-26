#conda activate pytorch_p36
conda activate pytorch_p38

$work_dir = "D:\VDisk\Network\OneDrive"
if ( Test-Path "D:\Flash\OneDrive" ) {
    $work_dir = "D:\Flash\OneDrive"
}

$gsoft_dir = "$work_dir\GSoft"
add-env-path $gsoft_dir

$env:PYTHONPATH = "$($work_dir)\Tensorflow\pcore"


$trainingConfig = @{
    "workDir" = "d:/demo/training-yolo";
    "demoDir" = "d:/demo";
}

$env:_trainingConfig = ConvertTo-Json $trainingConfig

Write-Host $env:PYTHONPATH