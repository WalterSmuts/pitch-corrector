// Headless UX test for the AudioWorklet build.
// - serves the app with COOP/COEP (serve.py) so crossOriginIsolated=true
// - launches Chromium feeding tone.wav as the fake microphone
// - drives Record, waits for the pipeline to go live, checks output + jank
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

// Resolve a Chromium binary: $PW_CHROME, the cached Playwright browser, or a
// system Chrome. (The bundled headless_shell version may not match, so we use
// the full chrome binary directly.)
function findChrome() {
  const cands = [
    process.env.PW_CHROME,
    `${process.env.HOME}/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome`,
    '/usr/bin/google-chrome',
    '/opt/google/chrome/chrome',
  ].filter(Boolean);
  const found = cands.find(p => { try { return fs.existsSync(p); } catch { return false; } });
  if (!found) throw new Error('No Chromium found; set PW_CHROME=/path/to/chrome');
  return found;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
const PORT = 8891;
const WAV = path.join(here, 'tone.wav');

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
const fails = [];
const check = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}: ${msg}`); if (!cond) fails.push(msg); };

// Capture a Chrome DevTools timeline trace via CDP. The resulting JSON loads
// directly in the Firefox Profiler (profiler.firefox.com), Chrome DevTools
// ("Load profile"), or perfetto. Off by default; PROFILE=1 for a timeline
// trace (frames + long tasks), PROFILE=cpu to also include V8 CPU sampling.
const PROFILE = process.env.PROFILE === '1' || process.env.PROFILE === 'cpu';
async function startTrace(client) {
  const cats = [
    'toplevel', 'devtools.timeline', 'disabled-by-default-devtools.timeline',
    'disabled-by-default-devtools.timeline.frame', 'blink.user_timing',
  ];
  if (process.env.PROFILE === 'cpu') cats.push('v8', 'v8.execute', 'disabled-by-default-v8.cpu_profiler');
  await client.send('Tracing.start', {
    transferMode: 'ReturnAsStream',
    traceConfig: { recordMode: 'recordUntilFull', includedCategories: cats },
  });
}
async function stopTrace(client, outPath) {
  const done = new Promise(r => client.once('Tracing.tracingComplete', r));
  await client.send('Tracing.end');
  const { stream } = await done;
  const fh = fs.openSync(outPath, 'w');
  for (;;) {
    const { data, base64Encoded, eof } = await client.send('IO.read', { handle: stream, size: 1 << 20 });
    fs.writeSync(fh, Buffer.from(data, base64Encoded ? 'base64' : 'utf8'));
    if (eof) break;
  }
  fs.closeSync(fh);
  await client.send('IO.close', { handle: stream });
}

// 1) start the COOP/COEP server from the repo root
const server = spawn('python3', ['serve.py', String(PORT)], { cwd: repo, stdio: 'ignore' });
await sleep(800);

let browser;
try {
  // 2) launch chromium feeding the WAV as the mic
  browser = await chromium.launch({
    headless: true,
    executablePath: findChrome(),
    args: [
      '--headless=new',
      '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-audio-capture=${WAV}`,
      '--autoplay-policy=no-user-gesture-required',
    ],
  });
  const ctx = await browser.newContext();
  await ctx.grantPermissions(['microphone'], { origin: `http://localhost:${PORT}` });
  const page = await ctx.newPage();

  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));

  // collect long tasks (jank) with timestamps
  await page.addInitScript(() => {
    window.__longtasks = [];
    try {
      new PerformanceObserver(l => { for (const e of l.getEntries()) window.__longtasks.push({ start: e.startTime, dur: e.duration }); })
        .observe({ entryTypes: ['longtask'] });
    } catch (_) {}
  });

  await page.goto(`http://localhost:${PORT}/?e2e`, { waitUntil: 'load' });

  // 3) cross-origin isolation
  check(await page.evaluate(() => self.crossOriginIsolated === true), 'crossOriginIsolated === true');

  // 4) wait for prewarm to finish (Record enabled)
  await page.waitForFunction(() => !document.getElementById('record-btn').disabled, { timeout: 20000 });

  // 5) click Record, mark the time, wait for the pipeline to go live
  const cdp = PROFILE ? await ctx.newCDPSession(page) : null;
  if (cdp) await startTrace(cdp);
  const recAt = await page.evaluate(() => performance.now());
  await page.click('#record-btn');
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('Recording'),
    { timeout: 15000 });
  const liveAt = await page.evaluate(() => performance.now());
  check(true, `pipeline went live ${Math.round(liveAt - recAt)}ms after Record click`);
  check(await page.evaluate(() => !!(window.__pc && window.__pc.is_audio_ready())), 'is_audio_ready() true');

  // 6) let it process the WAV
  await sleep(1500);

  // 7) read captured input + output and compute RMS
  const rms = await page.evaluate(() => {
    const r = (a) => { if (!a || !a.length) return 0; let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); };
    return { inN: window.__pc.recording_len(), in: r(window.__pc.get_recording()), out: r(window.__pc.get_output_recording()) };
  });
  console.log('  input samples:', rms.inN, ' input RMS:', rms.in.toFixed(4), ' output RMS:', rms.out.toFixed(4));
  check(rms.inN > 40000, 'input captured (mic file fed through)');
  check(rms.in > 0.01, 'input non-silent');
  check(rms.out > 0.01, 'output non-silent (pipeline produced audio)');

  await page.click('#stop-btn');

  if (cdp) {
    const out = path.join(here, 'trace.json');
    await stopTrace(cdp, out);
    console.log(`  trace written: ${out} (load in profiler.firefox.com, Chrome DevTools, or perfetto)`);
  }

  // 8) jank during steady-state recording (exclude the ~init window right after click)
  const lts = await page.evaluate(() => window.__longtasks);
  const steady = lts.filter(t => t.start > liveAt + 300);
  const maxSteady = steady.reduce((m, t) => Math.max(m, t.dur), 0);
  console.log(`  long tasks total=${lts.length}  during steady recording=${steady.length}  maxSteady=${maxSteady.toFixed(0)}ms`);
  check(maxSteady < 50, `no long (>50ms) main-thread task during steady recording (max ${maxSteady.toFixed(0)}ms)`);

  // 9) no fatal console errors
  const fatal = errors.filter(e => /TextDecoder|invalid array type|__wasm_init_tls|Uncaught|SharedArrayBuffer/.test(e));
  if (fatal.length) console.log('  fatal console errors:', fatal);
  check(fatal.length === 0, 'no fatal console errors (TextDecoder/Atomics/etc.)');

} finally {
  if (browser) await browser.close();
  server.kill();
}

console.log(fails.length ? `\nRESULT: FAIL (${fails.length})` : '\nRESULT: PASS');
process.exit(fails.length ? 1 : 0);
