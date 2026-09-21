# Hop this Windows laptop onto the robot's WiFi, run one command, hop back.
#
#   powershell -ExecutionPolicy Bypass -File tools\wifi_hop.ps1 -Command "python tools\sync_robot.py"
#
# Why: the robot's network (IISLab-AMEC, 192.168.73.x) has no internet, and the
# laptop's usual one (uvptechnicom) cannot see the robot. Anything driven from
# a tool that itself needs the internet (Claude Code, for one) has to do the
# whole robot job inside a single call that ends back on the home network.
#
# How: WlanConnect through wlanapi.dll, not `netsh wlan connect`, which on this
# box is gated behind the Windows Location permission. A background keeper
# re-issues the connect while the command runs, because Windows drifts back to
# the network that has internet. Everything is also appended to wifi_hop.log
# next to this script, in case the caller's output is cut.
param(
    [string]$Command = "",
    [string]$RobotWifi = "IISLab-AMEC",
    [string]$HomeWifi = "uvptechnicom",
    [string]$Robot = "192.168.73.210",
    [string]$Adapter = "WiFi",
    [int]$TimeoutSec = 45,
    [switch]$Keep   # internal: run as the background keeper loop
)

$src = @"
using System;
using System.Runtime.InteropServices;
public class Wlan {
    [DllImport("wlanapi.dll")] public static extern uint WlanOpenHandle(uint v, IntPtr r, out uint nv, out IntPtr h);
    [DllImport("wlanapi.dll")] public static extern uint WlanCloseHandle(IntPtr h, IntPtr r);
    [DllImport("wlanapi.dll")] public static extern uint WlanConnect(IntPtr h, ref Guid g, ref P p, IntPtr r);
    [StructLayout(LayoutKind.Sequential)]
    public struct P {
        public int mode;                                   // 0 = by profile name
        [MarshalAs(UnmanagedType.LPWStr)] public string profile;
        public IntPtr ssid;
        public IntPtr bssidList;
        public int bssType;                                // 1 = infrastructure
        public uint flags;
    }
    public static uint Connect(Guid iface, string profile) {
        IntPtr h; uint nv;
        uint rc = WlanOpenHandle(2, IntPtr.Zero, out nv, out h);
        if (rc != 0) return rc;
        P p = new P(); p.mode = 0; p.profile = profile; p.ssid = IntPtr.Zero; p.bssidList = IntPtr.Zero; p.bssType = 1; p.flags = 0;
        rc = WlanConnect(h, ref iface, ref p, IntPtr.Zero);
        WlanCloseHandle(h, IntPtr.Zero);
        return rc;
    }
}
"@
if (-not ([System.Management.Automation.PSTypeName]"Wlan").Type) { Add-Type -TypeDefinition $src }

function Connect-Wlan([string]$ProfileName) {
    $guid = [Guid](Get-NetAdapter -Name $Adapter).InterfaceGuid
    return [Wlan]::Connect($guid, $ProfileName)
}
function Get-WlanName {
    return ((Get-NetConnectionProfile -InterfaceAlias $Adapter -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name) -join ",")
}
function Hop([string]$ProfileName, [string]$WaitHost) {
    $rc = Connect-Wlan $ProfileName
    "WlanConnect('$ProfileName') returned $rc"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $ok = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $ip = (Get-NetIPAddress -InterfaceAlias $Adapter -AddressFamily IPv4 -ErrorAction SilentlyContinue |
               Where-Object { $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
        if (-not $ip) { continue }
        if ($WaitHost -eq "") { $ok = $true; break }
        if (Test-Connection $WaitHost -Count 1 -Quiet -ErrorAction SilentlyContinue) { $ok = $true; break }
    }
    "network: $(Get-WlanName)  laptop ip: $ip  reached: $ok"
    return $ok
}

if ($Keep) {
    # keeper loop: re-connect whenever Windows wanders off to a network with internet
    while ($true) {
        if ((Get-WlanName) -ne $RobotWifi) { $null = Connect-Wlan $RobotWifi; Start-Sleep -Seconds 6 }
        Start-Sleep -Seconds 3
    }
}
if ($Command -eq "") { "usage: wifi_hop.ps1 -Command '<what to run while on the robot WiFi>'"; exit 1 }

$log = Join-Path $PSScriptRoot "wifi_hop.log"
Start-Transcript -Path $log -Append | Out-Null
$keeper = $null
try {
    if (Hop $RobotWifi $Robot) {
        $keeper = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList `
            "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Keep -RobotWifi `"$RobotWifi`" -Adapter `"$Adapter`""
        "---- running: $Command"
        Invoke-Expression $Command
        "---- command exit code: $LASTEXITCODE"
    } else {
        "robot $Robot did not answer on $RobotWifi; nothing run"
    }
} finally {
    if ($keeper) { Stop-Process -Id $keeper.Id -Force -ErrorAction SilentlyContinue }
    "---- returning to $HomeWifi"
    $null = Hop $HomeWifi ""
    $back = $false
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-NetConnection api.anthropic.com -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue) { $back = $true; break }
        Start-Sleep -Seconds 2
    }
    "internet back: $back"
    Stop-Transcript | Out-Null
}
