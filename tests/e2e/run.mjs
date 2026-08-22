// Headless pipeline-health test for the AudioWorklet build.
// - serves the app with COOP/COEP (serve.py) so crossOriginIsolated=true
// - launches Chromium feeding tone.wav as the fake microphone
// - drives Record, waits for the pipeline to go live, checks output + jank
// See ui.mjs for the timeline/interaction coverage.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch, makeChecker, sleep } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = 8891;
const { check, finish } = makeChecker();

// Capture a Chrome DevTools timeline trace via CDP. The resulting JSON loads
// directly in the Firefox Profiler (profiler.firefox.com), Chrome DevTools
// ("Load profile"), or perfetto. Off by default; PROFILE=1 for a timeline
// trace (frames + long tasks), PROFILE=cpu to also include V8 CPU sampling.
const PROFILE = process.env.PROFILE === '1' || process.env.PROFILE === 'cpu';
async function startTrace(client) {
  const cats = [
    'toplevel',
    'devtools.timeline',
    'disabled-by-default-devtools.timeline',
    'disabled-by-default-devtools.timeline.frame',
    'blink.user_timing',
  ];
  if (process.env.PROFILE === 'cpu')
    cats.push('v8', 'v8.execute', 'disabled-by-default-v8.cpu_profiler');
  await client.send('Tracing.start', {
    transferMode: 'ReturnAsStream',
    traceConfig: { recordMode: 'recordUntilFull', includedCategories: cats },
  });
}
async function stopTrace(client, outPath) {
  const done = new Promise((r) => client.once('Tracing.tracingComplete', r));
  await client.send('Tracing.end');
  const { stream } = await done;
  const fh = fs.openSync(outPath, 'w');
  for (;;) {
    const { data, base64Encoded, eof } = await client.send('IO.read', {
      handle: stream,
      size: 1 << 20,
    });
    fs.writeSync(fh, Buffer.from(data, base64Encoded ? 'base64' : 'utf8'));
    if (eof) break;
  }
  fs.closeSync(fh);
  await client.send('IO.close', { handle: stream });
}

const { page, ctx, errors, close } = await launch({ port: PORT });
try {
  // collect long tasks (jank) with timestamps
  await page.addInitScript(() => {
    window.__longtasks = [];
    try {
      new PerformanceObserver((l) => {
        for (const e of l.getEntries())
          window.__longtasks.push({ start: e.startTime, dur: e.duration });
      }).observe({ entryTypes: ['longtask'] });
    } catch (_) {}
  });

  await page.goto(`http://localhost:${PORT}/?e2e`, { waitUntil: 'load' });

  // cross-origin isolation
  check(
    await page.evaluate(() => self.crossOriginIsolated === true),
    'crossOriginIsolated === true',
  );

  // wait for prewarm to finish (Record enabled)
  await page.waitForFunction(() => !document.getElementById('record-btn').disabled, {
    timeout: 20000,
  });

  // click Record, mark the time, wait for the pipeline to go live
  const cdp = PROFILE ? await ctx.newCDPSession(page) : null;
  if (cdp) await startTrace(cdp);
  const recAt = await page.evaluate(() => performance.now());
  await page.click('#record-btn');
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('Recording'),
    { timeout: 15000 },
  );
  const liveAt = await page.evaluate(() => performance.now());
  check(true, `pipeline went live ${Math.round(liveAt - recAt)}ms after Record click`);
  check(
    await page.evaluate(() => !!(window.__pc && window.__pc.is_audio_ready())),
    'is_audio_ready() true',
  );

  // let it process the WAV
  await sleep(1500);

  // read captured input + output and compute RMS
  const rms = await page.evaluate(() => {
    const r = (a) => {
      if (!a || !a.length) return 0;
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * a[i];
      return Math.sqrt(s / a.length);
    };
    return {
      inN: window.__pc.recording_len(),
      in: r(window.__pc.get_recording()),
      out: r(window.__pc.get_output_recording()),
    };
  });
  console.log(
    '  input samples:',
    rms.inN,
    ' input RMS:',
    rms.in.toFixed(4),
    ' output RMS:',
    rms.out.toFixed(4),
  );
  check(rms.inN > 40000, 'input captured (mic file fed through)');
  check(rms.in > 0.01, 'input non-silent');
  check(rms.out > 0.01, 'output non-silent (pipeline produced audio)');

  // jank during steady-state recording (exclude the ~init window right after
  // click) — measured BEFORE Stop so the stop-time fit render is separate
  const lts = await page.evaluate(() => window.__longtasks);
  const steady = lts.filter((t) => t.start > liveAt + 300);
  const maxSteady = steady.reduce((m, t) => Math.max(m, t.dur), 0);
  console.log(
    `  long tasks total=${lts.length}  during steady recording=${steady.length}  maxSteady=${maxSteady.toFixed(0)}ms`,
  );
  check(
    maxSteady < 50,
    `no long (>50ms) main-thread task during steady recording (max ${maxSteady.toFixed(0)}ms)`,
  );

  // stopping fits the whole recording; progressive rendering must keep the
  // repaint off the long-task radar too
  const stopAt = await page.evaluate(() => performance.now());
  await page.click('#stop-btn');
  await sleep(700);
  const stopTasks = await page.evaluate(() => window.__longtasks);
  const maxStop = stopTasks.filter((t) => t.start > stopAt).reduce((m, t) => Math.max(m, t.dur), 0);
  check(maxStop < 60, `stop-time fit render does not jank (max ${maxStop.toFixed(0)}ms)`);

  if (cdp) {
    const out = path.join(here, 'trace.json');
    await stopTrace(cdp, out);
    console.log(
      `  trace written: ${out} (load in profiler.firefox.com, Chrome DevTools, or perfetto)`,
    );
  }

  // no fatal console errors
  const fatal = errors.filter((e) =>
    /TextDecoder|invalid array type|__wasm_init_tls|Uncaught|SharedArrayBuffer|pageerror/.test(e),
  );
  if (fatal.length) console.log('  fatal console errors:', fatal);
  check(fatal.length === 0, 'no fatal console errors (TextDecoder/Atomics/etc.)');
} finally {
  await close();
}
finish();
