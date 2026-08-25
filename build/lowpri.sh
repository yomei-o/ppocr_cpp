# Run the build below the priority of everything else on the machine.
#
# WHY. A parallel build takes every core it can get, and on Windows that makes the rest of the
# desktop stutter — the editor, the browser, anything that wants a slice while 12 g++ processes are
# running. Lowering the priority class does not make the build much slower in practice, because the
# cores are idle whenever nothing else wants them; it only changes who wins when something does.
#
# HOW. Priority class is inherited by child processes at creation, so setting it on THIS shell is
# enough — make, every g++ it spawns, and the final link all come out below normal. That is why
# there is no wrapper around each command here: a wrapper would have to solve quoting for arbitrary
# argument lists, and inheritance solves it for free.
#
#   PRIORITY=Idle sh build/gcc.sh ...      lowest; the build yields to literally everything
#   PRIORITY=Normal sh build/gcc.sh ...    off
#
# Idle also lowers the process's I/O priority on Windows, which matters here: a from-scratch build
# writes a few hundred object files, and on a machine with one disk that is often what is actually
# making the desktop feel slow rather than the CPU.
#
# Failing to set it is never fatal. This runs under mingw/MSYS on Windows, WSL, Kaggle's Linux
# containers and Emscripten, and a build that stops because it could not find powershell would be a
# worse outcome than a build that runs at normal priority.

PRIORITY="${PRIORITY:-BelowNormal}"

# $$ under MSYS/Git Bash is the MSYS process id, NOT the Windows one — they are different numbers
# for the same shell (14533 vs 4764 on the machine this was written on). Handing $$ to Get-Process
# silently targets some unrelated process or none at all, and the whole thing then looks like it
# worked while changing nothing. /proc/$$/winpid is the translation.
_LOWPRI_PID="$$"
[ -r "/proc/$$/winpid" ] && _LOWPRI_PID=$(cat "/proc/$$/winpid")

if [ "$PRIORITY" != "Normal" ]; then
  if command -v powershell >/dev/null 2>&1; then
    powershell -NoProfile -NonInteractive -Command \
      "try { (Get-Process -Id $_LOWPRI_PID).PriorityClass = '$PRIORITY' } catch {}" \
      >/dev/null 2>&1 || true
  elif command -v renice >/dev/null 2>&1; then
    # POSIX has no BelowNormal/Idle; 10 and 19 are the usual stand-ins.
    case "$PRIORITY" in
      Idle) renice -n 19 -p $$ >/dev/null 2>&1 || true ;;
      *)    renice -n 10 -p $$ >/dev/null 2>&1 || true ;;
    esac
  fi
fi

# If the desktop still stutters, the other lever is how many compilers run at once: the priority
# class decides who wins a contended core, JOBS decides how many cores are contended at all.
#
#   JOBS=$(( $(nproc) - 2 )) sh build/gcc.sh ...
