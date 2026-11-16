#!/bin/zsh
# Copy base64 parts from the Great Lakes screen session to this machine, one part at a time:
# the remote shell prints the part between two marker lines, `screen hardcopy -h` captures the
# scrollback, and the text between the markers is kept (whitespace stripped, since screen wraps
# long lines at the window width). Each part is checked against its md5 from the cluster.
# Usage: gl_fetch_parts.sh <md5 list "md5  part_NNN"> <remote dir> <local dir>
# Prepare on the cluster:  base64 -w 76 <file> | split -l 40000 -d -a 3 - part_ ; md5sum part_*
# Raise the scrollback first: screen -S gl -X scrollback 100000 (a part wraps to ~80,000 rows).
set -u
list="$1"; rdir="$2"; ldir="$3"
mkdir -p "$ldir"
hc=/tmp/claude-505/gl/hc_fetch.txt
while read -r md5 part; do
  if [[ -f "$ldir/$part" ]] && [[ "$(md5 -q "$ldir/$part")" == "$md5" ]]; then
    echo "$part already here"; continue
  fi
  for attempt in 1 2 3; do
    screen -S gl -X scrollback 0; screen -S gl -X scrollback 100000     # empty the history
    tag="F$RANDOM$RANDOM"
    printf 'printf "%s_B\\n"; cat %s/%s; printf "%s_E\\n"\n' "$tag" "$rdir" "$part" "$tag" \
      > /tmp/claude-505/gl/fetch_cmd.txt
    screen -S gl -X readbuf /tmp/claude-505/gl/fetch_cmd.txt && screen -S gl -X paste .
    start=$SECONDS
    # `screen -X hardcopy` returns before screen has written the file, so wait before reading
    # it; otherwise grep often sees a half-written copy (~3 MB per part) and misses the marker.
    until screen -S gl -X hardcopy -h "$hc" && sleep 1 && grep -qx "${tag}_E" "$hc"; do
      (( SECONDS - start > 300 )) && break
    done
    # Join everything and re-wrap to the cluster's 76-column lines, so the md5 matches.
    { awk -v b="${tag}_B" -v e="${tag}_E" '$0==b{f=1;next} $0==e{f=0} f' "$hc" \
        | tr -d ' \t\r\n' | fold -w 76; echo; } > "$ldir/$part.tmp"
    if [[ "$(md5 -q "$ldir/$part.tmp")" == "$md5" ]]; then
      mv "$ldir/$part.tmp" "$ldir/$part"; echo "$part ok ($((SECONDS - start)) s)"; break
    fi
    echo "$part attempt $attempt: md5 mismatch"
  done
done < "$list"
