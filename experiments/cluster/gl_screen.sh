#!/bin/zsh
# Run one command in the logged-in Great Lakes session held by `screen -S gl`
# and print its output. Usage: gl.sh 'command' [timeout_seconds]
# The command is pasted verbatim (screen readbuf + paste), so $, quotes and
# backslashes reach the remote shell unchanged. Output is taken between two
# marker lines that the remote shell prints before and after the command.
set -u
cmd="$1"; timeout="${2:-60}"
tag="GL$RANDOM$RANDOM"
dir=/tmp/claude-505/gl; mkdir -p "$dir"  # absolute: screen resolves relative paths from its own cwd
buf="$dir/buf_$tag.txt"; hc="$dir/hc_$tag.txt"
printf 'echo %s_S; %s\necho %s_E\n' "$tag" "$cmd" "$tag" > "$buf"
screen -S gl -X readbuf "$buf" && screen -S gl -X paste . || { echo "screen session gl not reachable"; exit 2; }
end=$((SECONDS + timeout))
while (( SECONDS < end )); do
  sleep 0.5
  screen -S gl -X hardcopy -h "$hc"
  if grep -qx "${tag}_E" "$hc" 2>/dev/null; then
    awk -v s="${tag}_S" -v e="${tag}_E" '$0==s{f=1;next} $0==e{f=0} f && index($0, "echo " e)==0' "$hc" | sed -e 's/[[:space:]]*$//'
    rm -f "$buf" "$hc"; exit 0
  fi
done
echo "[timeout after ${timeout}s; last screen lines:]"; tail -15 "$hc"; rm -f "$buf"; exit 1
