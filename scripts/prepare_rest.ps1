$py = 'D:\Documents\Research Work\Landslide Cross Region\.venv\Scripts\python.exe'
foreach ($d in 'cas','lmhld') {
  & $py -I -c "import sys; sys.path.insert(0, r'D:\Documents\Research Work\Landslide Cross Region\src'); import landslide.prepare as p; p.prepare_$d()" *>> 'D:\Documents\Research Work\Landslide Cross Region\results\prepare_rest.log'
}
'ALL DONE' | Out-File -Append 'D:\Documents\Research Work\Landslide Cross Region\results\prepare_rest.log'
