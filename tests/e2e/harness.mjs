// Shared harness for the headless e2e tests: starts the COOP/COEP server,
// launches Chromium with a WAV as the fake microphone, and opens the app.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
export const repoRoot = path.resolve(here, '../..');
export const toneWav = path.join(here, 'tone.wav');

// Resolve a Chromium binary: $PW_CHROME, the cached Playwright browser, or a
// system Chrome. (The bundled headless_shell version may not match, so we use
// the full chrome binary directly.)
export function findChrome() {
  const cands = [
    process.env.PW_CHROME,
    `${process.env.HOME}/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome`,
    '/usr/bin/google-chrome',
    '/opt/google/chrome/chrome',
  ].filter(Boolean);
  const found = cands.find((p) => {
    try {
      return fs.existsSync(p);
    } catch {
      return false;
    }
  });
  if (!found) throw new Error('No Chromium found; set PW_CHROME=/path/to/chrome');
  return found;
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Start server + browser + page. Returns { page, ctx, errors, close }.
 * `errors` collects console errors and uncaught page exceptions.
 */
export async function launch({ port, wav = toneWav }) {
  const server = spawn('python3', ['serve.py', String(port)], { cwd: repoRoot, stdio: 'ignore' });
  await sleep(800);

  const browser = await chromium.launch({
    headless: true,
    executablePath: findChrome(),
    args: [
      '--headless=new',
      '--use-fake-ui-for-media-stream',
      // Required for --use-file-for-fake-audio-capture to take effect;
      // without it Chromium ignores the WAV and the mic delivers silence.
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${wav}`,
      '--autoplay-policy=no-user-gesture-required',
    ],
  });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1500 } });
  await ctx.grantPermissions(['microphone'], { origin: `http://localhost:${port}` });
  const page = await ctx.newPage();

  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));

  return {
    page,
    ctx,
    errors,
    close: async () => {
      await browser.close();
      server.kill();
    },
  };
}

/** Simple PASS/FAIL check collector shared by the test scripts. */
export function makeChecker() {
  const fails = [];
  const check = (cond, msg) => {
    console.log(`${cond ? 'PASS' : 'FAIL'}: ${msg}`);
    if (!cond) fails.push(msg);
  };
  const finish = () => {
    console.log(fails.length ? `\nRESULT: FAIL (${fails.length})` : '\nRESULT: PASS');
    process.exit(fails.length ? 1 : 0);
  };
  return { check, finish, fails };
}
