// Sync a release version into the files Tauri/Cargo read. Called by semantic-release
// (@semantic-release/exec) as: node scripts/bump-version.mjs <version>
// The tauri.conf.json version is what the updater compares against the release feed.
import { readFileSync, writeFileSync } from "node:fs";

const v = process.argv[2];
if (!v) throw new Error("usage: bump-version.mjs <version>");

const sub = (path, re, repl) => {
  const before = readFileSync(path, "utf8");
  const after = before.replace(re, repl);
  if (after === before) throw new Error(`${path}: ${re} matched nothing`);
  writeFileSync(path, after);
};

// tauri.conf.json: "version": "x.y.z" (first match = the app version)
sub("src-tauri/tauri.conf.json", /("version":\s*")[^"]+(")/, `$1${v}$2`);

// Cargo.toml: first `version = "x.y.z"` (the [package] version)
sub("src-tauri/Cargo.toml", /^version = "[^"]+"/m, `version = "${v}"`);

console.log(`bumped to ${v}`);
