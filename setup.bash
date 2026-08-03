# Source this to get a working shell:  source setup.bash
#
# Creates .venv if it is missing, installs requirements.txt when it has changed
# since the last run, and activates the environment. Safe to source repeatedly --
# the common case does no work and returns immediately.
#
# Deliberately avoids `set -e` and `exit`: this runs inside your interactive
# shell, so either one would close your terminal on a bad day.

if [ -z "${BASH_SOURCE[0]}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "setup.bash must be sourced, not executed:" >&2
    echo "    source setup.bash" >&2
    exit 1
fi

_mlx_setup() {
    local root venv python stamp want have created=0

    root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    venv="$root/.venv"
    stamp="$venv/.requirements-sha"

    if [ ! -f "$root/requirements.txt" ]; then
        echo "setup.bash: requirements.txt not found in $root" >&2
        return 1
    fi

    # Leave any unrelated environment alone rather than silently stacking venvs.
    if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" != "$venv" ]; then
        echo "setup.bash: another virtualenv is active ($VIRTUAL_ENV)." >&2
        echo "            run 'deactivate' first." >&2
        return 1
    fi

    if [ ! -x "$venv/bin/python" ]; then
        python="$(command -v python3 || command -v python)"
        if [ -z "$python" ]; then
            echo "setup.bash: no python3 on PATH" >&2
            return 1
        fi
        echo "creating venv at .venv ($("$python" --version 2>&1))"
        "$python" -m venv "$venv" || {
            echo "setup.bash: could not create the venv" >&2
            return 1
        }
        created=1
    fi

    # Reinstall only when requirements.txt actually changed. pip is fast when
    # everything is satisfied, but not free, and this gets sourced a lot.
    want="$(sha256sum "$root/requirements.txt" 2>/dev/null | cut -d' ' -f1)"
    have="$(cat "$stamp" 2>/dev/null)"
    if [ "$created" = 1 ] || [ -z "$want" ] || [ "$want" != "$have" ]; then
        echo "installing requirements…"
        "$venv/bin/python" -m pip install --quiet --upgrade pip || return 1
        "$venv/bin/python" -m pip install --quiet -r "$root/requirements.txt" || {
            echo "setup.bash: pip install failed" >&2
            return 1
        }
        [ -n "$want" ] && printf '%s\n' "$want" > "$stamp"
    fi

    # shellcheck source=/dev/null
    . "$venv/bin/activate" || return 1

    # Let `python -m train.ga …` work from any directory, not just the repo root.
    case ":$PYTHONPATH:" in
        *":$root:"*) ;;
        *) export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$root" ;;
    esac

    echo "ml-experiments ready · $(python --version 2>&1) · $venv"
    echo "  python -m unittest discover -s tests -t .   run the tests"
    echo "  python -m core.arena snake ga dqn           compare models"
    echo "  python serve.py                             match viewer on :8000"
    echo "  deactivate                                  leave the environment"
}

_mlx_setup
_mlx_status=$?
unset -f _mlx_setup
if [ "$_mlx_status" -ne 0 ]; then
    echo "setup.bash: setup did not complete" >&2
fi
# `return` from a sourced file hands the status back to the caller intact, so
# scripts can do: source setup.bash || exit 1
return "$_mlx_status"
