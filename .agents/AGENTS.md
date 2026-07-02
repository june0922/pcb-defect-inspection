# Windows Batch (.bat) File Guidelines
- **Encoding & Language:** When creating or modifying `.bat` files, ALWAYS include `chcp 65001 >nul` at the very beginning of the script.
- **English Only:** ALWAYS write any comments (`::`) and console output (`echo`, `set /p`) strictly in English to prevent CP949 encoding corruption on Korean Windows command prompts.
