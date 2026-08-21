Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)

ExePath = ScriptDir & "\dist\DataAnnotationStudio\DataAnnotationStudio.exe"
PywPath = ScriptDir & "\venv\Scripts\pythonw.exe"
AppPath = ScriptDir & "\app.py"

If FSO.FileExists(ExePath) Then
    WshShell.Run """" & ExePath & """", 0, False
ElseIf FSO.FileExists(PywPath) Then
    WshShell.Run """" & PywPath & """ """ & AppPath & """", 0, False
Else
    WshShell.Run "pythonw """ & AppPath & """", 0, False
End If
