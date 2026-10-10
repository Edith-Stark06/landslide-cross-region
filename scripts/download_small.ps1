$ErrorActionPreference='Stop'
function Get-Z($rec,$name,$dir){ curl.exe -L -C - --fail --retry 5 -o "$dir\$name" "https://zenodo.org/api/records/$rec/files/$name/content" }
Get-Z 10463239 'ValidData.zip' 'D:\Documents\Research Work\Landslide Cross Region\data\raw\landslide4sense'
Get-Z 10463239 'TestData.zip' 'D:\Documents\Research Work\Landslide Cross Region\data\raw\landslide4sense'
Get-Z 10463239 'TrainData.zip' 'D:\Documents\Research Work\Landslide Cross Region\data\raw\landslide4sense'
foreach ($n in 'trainX.npy','trainY.npy','valX.npy','valY.npy','testX.npy','testY.npy') { Get-Z 7189381 $n 'D:\Documents\Research Work\Landslide Cross Region\data\raw\hr_gldd' }
'DONE' | Out-File 'D:\Documents\Research Work\Landslide Cross Region\data\raw\_download_done.txt'
