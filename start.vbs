' Starts Data Annotation Studio via start.bat in a minimised console window.
' The UI opens as configured in config.ini -> [App] LAUNCH_MODE (default: browser).
' Close the minimised "Data Annotation Studio" console window to stop the app.
Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ScriptDir
WshShell.Run "cmd /c """ & ScriptDir & "\start.bat""", 7, False
