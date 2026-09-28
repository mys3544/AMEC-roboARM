#!/bin/bash
# Make the M3 Pro join the lab WiFi (IISLab-AMEC) at boot instead of starting its own
# hotspot (the NetworkManager connection "Hotspot", SSID ROSMASTER). Run as root:
#
#   bash lab_wifi.sh check    # what is up now and what starts at boot
#   bash lab_wifi.sh probe    # join the lab WiFi for a moment, log the address, back to the hotspot
#   bash lab_wifi.sh switch   # lab WiFi at boot, hotspot off; undone unless confirmed
#
# m3pro/tools/lab_wifi.py leaves the lab connection as a keyfile at $NEW (passphrase from
# the laptop's saved profile) and runs probe and switch as detached systemd units: leaving
# the hotspot drops every client, the SSH session that started them too. switch waits
# $WAIT s for a client on the lab network to create $CONFIRM, and otherwise puts the
# hotspot back the way it was. Both log to $LOG.
set -u
LAB=IISLab-AMEC
HOTSPOT=Hotspot
DEV=wlP1p1s0
KEYFILE=/etc/NetworkManager/system-connections/$LAB.nmconnection
NEW=/tmp/m3pro-lab.nmconnection
CONFIRM=/tmp/m3pro-lab-wifi.confirm
WAIT=180
LOG=/var/log/m3pro-lab-wifi.log

check() {
    nmcli -f NAME,DEVICE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show | grep -E "^(NAME|$HOTSPOT|$LAB) "
    echo "$DEV address: $(nmcli -g IP4.ADDRESS device show "$DEV")"
}

lab() {  # join the lab WiFi and log the address it gave us
    nmcli connection up "$LAB" || return 1
    echo "$1: lab address $(nmcli -g IP4.ADDRESS device show "$DEV" | cut -d/ -f1)," \
         "gateway $(nmcli -g IP4.GATEWAY device show "$DEV")"
    if ping -c 1 -W 3 1.1.1.1 >/dev/null; then echo "internet: yes"; else echo "internet: no"; fi
}

case "${1:-}" in
    check) check; exit 0 ;;
    probe|switch) exec >>"$LOG" 2>&1; echo "== $(date '+%F %T') $1" ;;
    *) echo "usage: $0 check|probe|switch"; exit 2 ;;
esac

if [ -f "$NEW" ]; then
    install -o root -g root -m 600 "$NEW" "$KEYFILE" && rm -f "$NEW" && nmcli connection load "$KEYFILE" || exit 1
fi
sleep 5   # let the SSH command that started this unit return first

if [ "$1" = probe ]; then
    lab probe
    ok=$?
    [ "$ok" = 0 ] || echo "probe: could not join $LAB"
    nmcli connection up "$HOTSPOT"
    exit "$ok"
fi

nmcli connection modify "$LAB" connection.autoconnect yes connection.autoconnect-priority 10 &&
    nmcli connection modify "$HOTSPOT" connection.autoconnect no || exit 1
if lab switch; then
    rm -f "$CONFIRM"   # only a client on the lab network counts
    for _ in $(seq "$WAIT"); do
        if [ -e "$CONFIRM" ]; then
            rm -f "$CONFIRM"
            echo "confirmed: $LAB at boot, the hotspot stays off"
            check
            exit 0
        fi
        sleep 1
    done
    echo "nobody confirmed within $WAIT s"
else
    echo "could not join $LAB"
fi
echo "putting the hotspot back"
nmcli connection modify "$LAB" connection.autoconnect no
nmcli connection modify "$HOTSPOT" connection.autoconnect yes
nmcli connection up "$HOTSPOT"
check
exit 1
