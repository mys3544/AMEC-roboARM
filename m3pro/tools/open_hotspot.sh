#!/bin/bash
# The M3 Pro's hotspot (SSID ROSMASTER, 192.168.8.88) is the NetworkManager
# connection "Hotspot": wifi in AP mode, WPA2-PSK, ipv4 shared. Run as root:
#
#   bash open_hotspot.sh check   # show it; exit 0 when it has no password
#   bash open_hotspot.sh open    # take the password off
#
# `open` restarts the hotspot, which drops every client -- the SSH session that
# started it too -- so m3pro/tools/open_hotspot.py runs it as a detached systemd
# unit. It then waits for a client to come back on the open network and create
# $CONFIRM, and puts the password back if none does within $WAIT s. It logs to
# $LOG and keeps the keyfile from before the change in /var/backups.
set -u
CONFIRM=/tmp/m3pro-open-hotspot.confirm
WAIT=120
LOG=/var/log/m3pro-open-hotspot.log

case "${1:-}" in
    check) ;;
    open) exec >>"$LOG" 2>&1; echo "== $(date '+%F %T') taking the password off the hotspot" ;;
    *) echo "usage: $0 check|open"; exit 2 ;;
esac

uuid=$(nmcli -g UUID,TYPE connection show | while IFS=: read -r id type; do
    [ "$type" = 802-11-wireless ] && [ "$(nmcli -g 802-11-wireless.mode connection show "$id")" = ap ] && echo "$id"
done | head -1)
[ -n "$uuid" ] || { echo "no NetworkManager wifi connection in AP mode"; exit 2; }

check() {
    nmcli connection show "$uuid" |
        grep -E '^(connection\.id|802-11-wireless\.(ssid|mode)|802-11-wireless-security\.key-mgmt|ipv4\.addresses|GENERAL\.STATE):'
    if [ -z "$(nmcli -g 802-11-wireless-security.key-mgmt connection show "$uuid" 2>/dev/null)" ]; then
        echo "=> the hotspot is OPEN, no password"
        return 0
    fi
    echo "=> the hotspot has a password"
    return 1
}

if [ "$1" = check ]; then check; exit; fi

check >/dev/null && { echo "already open, nothing to do"; exit 0; }
file=$(nmcli -g UUID,FILENAME connection show | grep "^$uuid:" | cut -d: -f2-)
backup=/var/backups/$(basename "$file").$(date +%Y%m%d-%H%M%S)
cp -p "$file" "$backup" || exit 1
echo "connection $uuid, keyfile saved as $backup"

restore() {
    echo "putting the password back"
    cp -p "$backup" "$file" && nmcli connection load "$file" && nmcli connection up "$uuid"
    check
    exit 1
}

sleep 5   # let the SSH command that started this unit return first
nmcli connection modify "$uuid" remove 802-11-wireless-security || exit 1
nmcli connection up "$uuid" || restore
rm -f "$CONFIRM"   # only a client of the open hotspot counts
for _ in $(seq "$WAIT"); do
    if [ -e "$CONFIRM" ]; then
        rm -f "$CONFIRM"
        echo "a client came back without a password"
        check
        exit 0
    fi
    sleep 1
done
echo "no client came back within $WAIT s"
restore
