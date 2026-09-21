#!/usr/bin/env bash
# The robot's desktop (:0) with or without a monitor on the DisplayPort.
#
#   sudo bash tools/display_setup.sh install [PORT] [--restart]
#   bash tools/display_setup.sh check
#   sudo bash tools/display_setup.sh uninstall [--restart]
#
# install: puts tools/20-headless-virtual.conf in /etc/X11/xorg.conf.d and
#   tools/monitors.xml in /etc/xdg (Mutter's system-wide fallback), restores
#   the stock /etc/X11/xorg.conf if an earlier version edited it, and drops the
#   earlier DP-0 headless entry from the per-user monitors.xml files. PORT is
#   the X name of the physical DisplayPort head, DP-1 on this robot (see the
#   .conf). Nothing takes effect until the display manager restarts: pass
#   --restart, or `sudo systemctl restart gdm3`, or reboot. That ends the
#   current desktop session; the docker containers are not affected.
# check: prints what the kernel, Xorg, Mutter and the VNC ports see. Run it
#   after the restart, once with nothing plugged in and once with the monitor.
# uninstall: removes both files, restores the stock xorg.conf.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XORG_CONF=/etc/X11/xorg.conf
BACKUP=/etc/X11/xorg.conf.backup-roboarm
SNIPPET=/etc/X11/xorg.conf.d/20-headless-virtual.conf
SYSTEM_MONITORS=/etc/xdg/monitors.xml
MARK="RoboARM"
LOGIN_USER=${SUDO_USER:-$(id -un)}
LOGIN_UID=$(id -u "$LOGIN_USER")
LOGIN_HOME=$(eval echo "~$LOGIN_USER")

need_root() {
    [ "$(id -u)" -eq 0 ] || { echo "run this with sudo" >&2; exit 1; }
}

# Run a command on the logged-in user's session bus, also when we are root.
as_login_user() {
    local env=(XDG_RUNTIME_DIR=/run/user/$LOGIN_UID DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$LOGIN_UID/bus)
    if [ "$(id -u)" -eq 0 ]; then
        runuser -u "$LOGIN_USER" -- env "${env[@]}" "$@"
    else
        env "${env[@]}" "$@"
    fi
}

restore_vendor_xorg_conf() {
    grep -q "$MARK" "$XORG_CONF" 2>/dev/null || return 0
    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$XORG_CONF"
        echo "restored the stock $XORG_CONF from $BACKUP"
    else
        sed -i "/^# --- headless support (added for the $MARK project)/,/\"ModeValidation\"/d" "$XORG_CONF"
        echo "removed the $MARK block from $XORG_CONF"
    fi
}

drop_headless_user_entries() {
    # Headless configurations now come from $SYSTEM_MONITORS. An entry whose
    # monitors are all vendor "unknown" in a per-user file is the earlier DP-0
    # attempt; it never matches again and only confuses the next reader.
    local f
    for f in "$LOGIN_HOME/.config/monitors.xml" /var/lib/gdm3/.config/monitors.xml; do
        [ -f "$f" ] || continue
        python3 - "$f" <<'EOF'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
tree = ET.parse(path)
root = tree.getroot()
dropped = 0
for conf in list(root.findall("configuration")):
    specs = conf.findall(".//monitorspec")
    if specs and all(s.findtext("vendor") == "unknown" for s in specs):
        root.remove(conf)
        dropped += 1
if dropped:
    tree.write(path)  # same inode, so the owner stays the same
    print(f"dropped {dropped} headless configuration(s) from {path}")
EOF
    done
}

install() {
    need_root
    local port=$1
    restore_vendor_xorg_conf
    mkdir -p "$(dirname "$SNIPPET")"
    sed -E "s/^(\s*Option\s+\"ConnectedMonitor\"\s+\")DP-1\"/\1$port\"/" \
        "$HERE/20-headless-virtual.conf" > "$SNIPPET"
    if [ -f "$SYSTEM_MONITORS" ] && ! grep -q "$MARK" "$SYSTEM_MONITORS"; then
        cp "$SYSTEM_MONITORS" "$SYSTEM_MONITORS.backup-roboarm"
        echo "kept the existing $SYSTEM_MONITORS as $SYSTEM_MONITORS.backup-roboarm"
    fi
    sed "s#<connector>DP-1</connector>#<connector>$port</connector>#" \
        "$HERE/monitors.xml" > "$SYSTEM_MONITORS"
    drop_headless_user_entries
    echo "installed $SNIPPET and $SYSTEM_MONITORS, forcing $port"
}

uninstall() {
    need_root
    rm -f "$SNIPPET"
    if [ -f "$SYSTEM_MONITORS" ] && grep -q "$MARK" "$SYSTEM_MONITORS"; then
        rm -f "$SYSTEM_MONITORS"
    fi
    if [ -f "$SYSTEM_MONITORS.backup-roboarm" ]; then
        mv "$SYSTEM_MONITORS.backup-roboarm" "$SYSTEM_MONITORS"
    fi
    restore_vendor_xorg_conf
    echo "removed; the desktop is stock again once the display manager restarts"
}

check() {
    set +e
    local c f log xauth bus
    echo "== kernel: DRM connectors"
    for c in /sys/class/drm/card*-*; do
        [ -e "$c/status" ] || continue
        printf '%s: %s, EDID %s bytes\n' "$(basename "$c")" "$(cat "$c/status")" "$(wc -c < "$c/edid")"
    done

    echo "== config files"
    for f in "$SNIPPET" "$SYSTEM_MONITORS"; do
        if [ -f "$f" ]; then echo "present: $f"; else echo "absent:  $f"; fi
    done
    if [ ! -f "$XORG_CONF" ]; then
        echo "absent:  $XORG_CONF"
    elif grep -q "$MARK" "$XORG_CONF"; then
        echo "$XORG_CONF still carries the old $MARK block (install restores it)"
    else
        echo "$XORG_CONF is stock"
    fi
    grep -E 'Option\s+"ConnectedMonitor"' "$SNIPPET" 2>/dev/null

    echo "== Xorg log (what the driver decided)"
    log=/var/log/Xorg.0.log
    if [ -r "$log" ]; then
        grep -E 'NVIDIA.*(ConnectedMonitor|ModeValidation|Virtual screen|connected|EDID|Setting mode|\(EE\)|\(WW\))' "$log" | tail -n 25
    else
        echo "cannot read $log"
    fi

    echo "== X on :0 (xrandr)"
    xauth=""
    for f in "/run/user/$LOGIN_UID/gdm/Xauthority" "$LOGIN_HOME/.Xauthority" /run/user/*/gdm/Xauthority; do
        if [ -r "$f" ]; then xauth=$f; break; fi
    done
    if [ -n "$xauth" ] && DISPLAY=:0 XAUTHORITY=$xauth xrandr --query >/dev/null 2>&1; then
        DISPLAY=:0 XAUTHORITY=$xauth xrandr --query | grep -vE '^\s'
        DISPLAY=:0 XAUTHORITY=$xauth xrandr --prop \
            | awk '/^[A-Za-z]/ {out=$1} /^\s+EDID:/ {print out ": has an EDID, so a real monitor is attached"}'
    else
        echo "cannot reach X on :0 (no readable Xauthority, or X is not running)"
    fi

    echo "== Mutter (gnome-shell) monitors and logical monitors"
    bus=/run/user/$LOGIN_UID/bus
    if [ -S "$bus" ]; then
        as_login_user gdbus call --session \
            --dest org.gnome.Mutter.DisplayConfig --object-path /org/gnome/Mutter/DisplayConfig \
            --method org.gnome.Mutter.DisplayConfig.GetCurrentState 2>/dev/null > /tmp/roboarm-displayconfig.txt
        if [ -s /tmp/roboarm-displayconfig.txt ]; then
            grep -oE "\('(DP|HDMI|eDP)-[0-9]+', '[^']*', '[^']*', '[^']*'\)" /tmp/roboarm-displayconfig.txt \
                | sort -u | sed 's/^/monitor /'
            grep -oE "\([0-9]+, [0-9]+, [0-9.]+, uint32 [0-9]+, (true|false), \[\('[^']+'" /tmp/roboarm-displayconfig.txt \
                | sed 's/^/logical /'
        else
            echo "gnome-shell did not answer on the session bus"
        fi
        rm -f /tmp/roboarm-displayconfig.txt
    else
        echo "no session bus at $bus (nobody logged in?)"
    fi

    echo "== services"
    printf 'gdm3: %s\n' "$(systemctl is-active gdm3 2>/dev/null)"
    if [ -S "$bus" ]; then
        printf 'gnome-remote-desktop (user service): %s\n' \
            "$(as_login_user systemctl --user is-active gnome-remote-desktop 2>/dev/null)"
    fi
    echo "listening (5900 = VNC, 3389 = RDP):"
    ss -ltn '( sport = :5900 or sport = :3389 )' 2>/dev/null | tail -n +2
}

cmd=${1:-}
shift || true
port=DP-1
restart=0
for arg in "$@"; do
    case $arg in
        --restart) restart=1 ;;
        *) port=$arg ;;
    esac
done
case $cmd in
    install) install "$port" ;;
    uninstall) uninstall ;;
    check) check; exit 0 ;;
    *) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
if [ "$restart" -eq 1 ]; then
    echo "restarting gdm3 (this ends the current desktop session)"
    systemctl restart gdm3
else
    echo "now: sudo systemctl restart gdm3   (or reboot), then: bash tools/display_setup.sh check"
fi
