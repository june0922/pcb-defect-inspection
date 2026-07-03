# Windows Batch (.bat) File Guidelines
> **Trigger:** Apply these rules ONLY when creating or modifying a `.bat` file. No need to consult this section for other file types.

## Encoding
- **Always declare UTF-8 first:** Every `.bat` file MUST begin with `chcp 65001 >nul` as the very first executable line, before any `echo`, `set`, or logic. Without this, any non-ASCII output (even from called Python scripts) may render as mojibake on Korean Windows (CP949 default).
- **Suppress the chcp output:** Use `>nul` suffix — `chcp 65001 >nul` — so the "Active code page: 65001" message is never shown to users.
- **English-only output:** Write ALL `echo` statements, `set /p` prompts, and `::` comments strictly in English. Do NOT write Korean in `.bat` files — even with `chcp 65001`, Korean characters can corrupt on certain terminal configurations (e.g., ConHost vs. Windows Terminal). Python scripts handle localized output; `.bat` is the launcher only.

## Structure
- **Working directory anchor:** Always navigate to the project root at the top of every script using `cd /d "%~dp0.."`. This makes all subsequent relative paths reliable regardless of where the script is double-clicked or called from.
- **Title line:** Set a descriptive window title with `title <Script Purpose>` for easy identification when multiple terminals are open.
- **Section separation:** Use `echo ========...========` banners to visually separate logical stages (preprocessing, training, results, etc.).

## User Interaction
- **Default to safe:** For any destructive or long-running action, default the prompt to `N` (No). Never default to `Y`.
- **Validate input case-insensitively:** Use `if /I "%var%"=="Y"` (`/I` flag) so both `y` and `Y` are accepted.
- **Always pause before exit:** End scripts with `pause` so users can read the final output before the window closes.
- **Provide cancellation path:** If the user declines an action, print a clear cancellation message and `exit /b` cleanly.

## Error Handling
- **Check prerequisites:** Before running Python commands, verify that critical files exist (e.g., `if not exist "runs\train\weights\last.pt" goto ...`).
- **Use `exit /b <code>`:** Exit with a non-zero code on failure so calling processes or CI systems can detect errors. Use `exit /b 1` for failures and `exit /b 0` for success.
- **Never silently swallow errors:** Avoid constructs that hide failures. Only suppress output with `>nul` when the command is expected to fail gracefully (e.g., `rmdir ... 2>nul`).

# Script Synchronization Guidelines
> **Trigger:** Apply these rules ONLY when creating or modifying a `.bat` or `.sh` file. No need to consult this section for other file types.

- **Keep .bat and .sh Consistent:** Any logic change made to a `.bat` file MUST be mirrored in the corresponding `.sh` file, and vice versa. This project maintains one `.bat` (Windows) and one `.sh` (Linux/macOS/GPU server) for every operational script.
- **Sync Scope:** Synchronize ALL of the following when modifying either file:
  - Control flow: conditions (`if`/`else`), loops, early exits
  - Command-line arguments and their default values
  - Environment variable names and their usage
  - Path construction logic (e.g., `%~dp0..` in `.bat` ↔ `$(dirname "$0")/..` in `.sh`)
  - User-facing prompts and confirmation messages (in English)
  - Error handling and exit codes
- **Idiom Mapping:** Use the correct shell idiom for each platform rather than a literal translation:
  - Working directory: `cd /d "%~dp0.."` (.bat) ↔ `cd "$(dirname "$0")/.."` (.sh)
  - Existence check: `if exist "path"` (.bat) ↔ `[ -f "path" ]` (.sh)
  - User prompt: `set /p var="msg: "` (.bat) ↔ `read -p "msg: " var` (.sh)
  - Conditional: `if /I "%var%"=="Y"` (.bat) ↔ `[[ "$var" =~ ^[Yy]$ ]]` (.sh)
  - Exit: `exit /b 0` (.bat) ↔ `exit 0` (.sh)
- **Commit Rule:** NEVER commit a `.bat` or `.sh` change without updating its counterpart in the same commit. A single commit must contain both files if either is changed.
- **Verification:** After editing, mentally trace both scripts side-by-side to confirm they produce identical behavior for all user inputs (Y, N, empty input, edge cases).


# Path Management Guidelines
- **No Absolute Paths:** NEVER hardcode absolute paths (e.g., `C:\Users\username\...`, `/home/user/...`) in any project file. Always use relative paths from the project root.
- **Relative Paths Only:** All file references in `.yaml`, `.py`, `.bat`, `.sh`, `.json`, `.txt`, and other config files must use relative paths (e.g., `runs/train/weights/last.pt`, `./data.yaml`).
- **No Temp Paths:** NEVER commit file paths pointing to system temporary directories (e.g., `C:\Users\...\AppData\Local\Temp\`, `/tmp/`). These are runtime-only and not reproducible.
- **Runtime Injection Pattern:** If a tool (e.g., ultralytics YOLO) requires an absolute path at runtime, inject it dynamically in code (e.g., `Path(...).resolve()`) rather than hardcoding it in config files. Use a placeholder like `PLACEHOLDER_SET_BY_SCRIPT` in static config files to make this intent clear.
- **Auto-generated Files:** Files auto-generated by training tools (e.g., `runs/train/args.yaml`) often contain absolute paths. Before committing such files, ALWAYS sanitize them by replacing absolute paths with relative equivalents.
- **Verification:** After any training run or tool execution, run a full-project grep for personal path patterns (e.g., `C:\Users`, `/home/`, `AppData`, `Desktop`, `OneDrive`) before committing.


# Agent Instructions Synchronization Guidelines
> **Trigger:** Apply these rules ONLY when modifying `AGENTS.md` or `CLAUDE.md`. No need to consult this section for other file types.

- **Keep AGENTS.md and CLAUDE.md Consistent:** Any addition, modification, or deletion made to `AGENTS.md` MUST be exactly mirrored in `CLAUDE.md`, and vice versa. The purpose is to maintain identical instruction sets across different agent environments.
- **Sync Scope:** Synchronize ALL of the following when modifying either file:
  - New rules or guidelines added to any section.
  - Modifications to existing rules, triggers, or path management logic.
  - Deletions of outdated instructions.
  - Formatting, headers, and bullet points.
- **Commit Rule:** NEVER commit a change to `AGENTS.md` or `CLAUDE.md` without updating its counterpart in the same commit. A single commit must contain both files if either is changed.
- **Verification:** After editing, compare both files side-by-side to confirm their contents remain 100% identical.