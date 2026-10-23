# Start weekend_act.cmd outside this SSH session (a WMI-created process outlives the session's job)
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd.exe /c C:\Users\Hercules\roboarm-lerobot\weekend_act.cmd'; CurrentDirectory = 'C:\Users\Hercules\roboarm-lerobot' }
"started: return $($r.ReturnValue), pid $($r.ProcessId)"
