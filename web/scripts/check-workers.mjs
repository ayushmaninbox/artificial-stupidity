/**
 * Parse-check everything in public/ that the bundler never sees.
 *
 * `next build` compiles app/, but public/ is copied verbatim — so a worker can
 * be syntactically dead and still deploy green. That happened: a stray literal
 * newline inside a string meant worker-image.js never parsed, the browser
 * reported only "the image worker could not start", and the build had been
 * perfectly happy about it.
 *
 * node --check reads .js as CommonJS, which rejects `import` outright, so each
 * file is checked through a .mjs copy instead.
 */
import { copyFileSync, mkdtempSync, readdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = new URL("../public/", import.meta.url).pathname;
const tmp = mkdtempSync(join(tmpdir(), "wcheck-"));
const files = readdirSync(dir).filter((f) => f.endsWith(".js"));

let bad = 0;
for (const f of files) {
  const copy = join(tmp, `${f}.mjs`);
  copyFileSync(join(dir, f), copy);
  try {
    execFileSync(process.execPath, ["--check", copy], { stdio: "pipe" });
  } catch (err) {
    bad++;
    console.error(`\n  ${f} does not parse:\n`);
    // macOS resolves the temp dir through /private, so match the copy by shape
    // rather than by the path we handed out, or the file name comes back mangled.
    console.error(String(err.stderr).replace(/\S*wcheck-\S*?\/([\w.-]+)\.mjs/g, "public/$1"));
  }
}

if (bad) {
  console.error(`${bad} of ${files.length} files in public/ would ship broken.\n`);
  process.exit(1);
}
console.log(`public/: ${files.length} files parse`);
