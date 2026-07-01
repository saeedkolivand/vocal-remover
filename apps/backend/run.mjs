// ponytail: launches the repo-root Python (in .venv) for whatever script is passed,
// with cwd = repo root so app.py's ROOT-relative paths (index.html, output, uploads)
// resolve exactly as they do when run directly. Cross-OS venv path, no shell quoting.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const venv = path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const py = existsSync(venv) ? venv : "python"; // ponytail: fall back to PATH python if no venv
const args = process.argv.slice(2);
if (args.length === 0) args.push("app.py");
spawn(py, args, { cwd: root, stdio: "inherit" }).on("exit", (c) => process.exit(c ?? 0));
