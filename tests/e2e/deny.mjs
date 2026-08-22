// Mic-permission deny path: clicking Record without granting the microphone
// must never enter the recording state — the pipeline may only start once
// the permission prompt is actually accepted.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { findChrome, makeChecker, repoRoot, sleep } from './harness.mjs';

const { check, finish } = makeChecker();

const PORT = 8901;
const server = spawn('python3', ['serve.py', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await sleep(800);
let browser;
try {
  // No --use-fake-ui-for-media-stream and no grantPermissions: headless
  // Chromium rejects getUserMedia (NotAllowedError) — the deny path.
  browser = await chromium.launch({
    headless: true,
    executablePath: findChrome(),
    args: ['--headless=new', '--use-fake-device-for-media-stream'],
  });
  const page = await (await browser.newContext()).newPage();
  await page.goto(`http://localhost:${PORT}/?e2e`, { waitUntil: 'load' });
  await page.waitForFunction(() => !document.getElementById('record-btn').disabled, { timeout: 20000 });
  await page.click('#record-btn');
  await sleep(2500);
  const status = await page.evaluate(() => document.getElementById('status').textContent);
  const recordEnabled = await page.evaluate(() => !document.getElementById('record-btn').disabled);
  check(!status.includes('Recording…'), 'denied mic never enters the recording state');
  check(status.includes('denied'), `status explains the denial ("${status}")`);
  check(recordEnabled, 'Record is re-enabled for a retry');
} finally {
  if (browser) await browser.close();
  server.kill();
}
finish();
