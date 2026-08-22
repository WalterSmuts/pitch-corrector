// Headless UX test for the timeline UI: record → follow-scroll, stop → fit,
// wheel zoom, seek + playhead alignment, view switching (pixel-probed),
// play/pause transport, and WAV upload. Complements run.mjs (audio pipeline
// health); this file owns the viewport/interaction behavior.
import { launch, makeChecker, sleep, toneWav } from './harness.mjs';

const PORT = 8892;
const { check, finish } = makeChecker();

const { page, errors, close } = await launch({ port: PORT });
try {
  await page.goto(`http://localhost:${PORT}/?e2e`, { waitUntil: 'load' });
  await page.waitForFunction(() => !document.getElementById('record-btn').disabled, { timeout: 20000 });

  // --- Record with a small follow window so it actually scrolls ---
  await page.click('#record-btn');
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('Recording'),
    { timeout: 15000 });
  await page.evaluate(() => window.__tl.follow(true, 1)); // 1s window
  await sleep(2500);

  const followState = await page.evaluate(() => {
    const tl = window.__tl;
    const total = window.__pc.recording_len();
    return { s0: tl.s0, spp: tl.spp, w: tl.w, total, follow: tl.followMode };
  });
  check(followState.follow, 'follow mode active during recording');
  check(followState.s0 > 0, `follow scrolled past the start (s0=${Math.round(followState.s0)})`);
  // The recording grows between the render and this readback, so allow
  // ~100ms of audio of slack on the pin check.
  const rightEdge = followState.s0 + followState.w * followState.spp;
  check(Math.abs(rightEdge - followState.total) < 4800,
    `right edge pinned to the recording head (off by ${Math.round(Math.abs(rightEdge - followState.total))} samples)`);

  // --- Stop → fit: whole recording visible ---
  await page.click('#stop-btn');
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('saved'));
  const fitState = await page.evaluate(() => {
    const tl = window.__tl;
    return { s0: tl.s0, xEnd: tl.xAtSample(window.__pc.recording_len()), cssW: tl.tracks[0].canvas.clientWidth };
  });
  check(fitState.s0 === 0, 'fit starts at 0');
  check(Math.abs(fitState.xEnd - fitState.cssW) < 3,
    `fit maps recording end to right edge (xEnd=${fitState.xEnd.toFixed(1)}, cssW=${fitState.cssW})`);

  // --- Wheel zoom in at the track center halves samples-per-px ---
  const track = page.locator('.tl-track-canvas').first();
  const box = await track.boundingBox();
  const sppBefore = await page.evaluate(() => {
    const tl = window.__tl;
    return tl.spp ?? window.__pc.recording_len() / tl.w;
  });
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -400);
  const zoomed = await page.evaluate(() => window.__tl.spp);
  check(zoomed !== null && zoomed < sppBefore,
    `wheel zooms in (spp ${Math.round(sppBefore)} -> ${Math.round(zoomed)})`);

  // Zoom buttons and Fit.
  await page.click('.tl-zoom-btn:nth-child(1)'); // −
  const zoomedOut = await page.evaluate(() => window.__tl.spp);
  check(zoomedOut > zoomed, `zoom-out button works (spp ${Math.round(zoomed)} -> ${Math.round(zoomedOut)})`);
  await page.click('.tl-zoom-btn:nth-child(3)'); // Fit
  const fitAgain = await page.evaluate(() => ({ s0: window.__tl.s0, spp: window.__tl.spp }));
  check(fitAgain.s0 === 0 && fitAgain.spp === null, 'Fit button restores full view');

  // --- Seek: click at 25% of the track; progress and playhead must agree ---
  await track.click({ position: { x: box.width * 0.25, y: box.height / 2 } });
  const seek = await page.evaluate(() => ({
    progress: window.__pc.playback_progress(),
    playheadLeft: document.querySelector('.tl-playhead').getBoundingClientRect().left,
  }));
  check(Math.abs(seek.progress - 0.25) < 0.02, `click seeks to 25% (progress=${seek.progress.toFixed(3)})`);
  const expectedX = box.x + box.width * 0.25;
  check(Math.abs(seek.playheadLeft + 1 - expectedX) < 3,
    `playhead aligns with the click (at ${seek.playheadLeft.toFixed(0)}px, expected ~${expectedX.toFixed(0)}px)`);

  // --- View switching: pixel-probe each view for its signature color ---
  const probe = (sel, test) => page.evaluate(([sel, test]) => {
    const c = document.querySelectorAll('.tl-track-canvas')[sel];
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    const fns = {
      orange: (r, g, b) => r > 200 && g > 100 && g < 200 && b < 100,
      green: (r, g, b) => g > 200 && r < 150,
      hot: (r, g, b) => r + g + b > 250 && !(r === 255 && g === 255 && b === 255),
    };
    for (let i = 0; i < d.length; i += 4) if (fns[test](d[i], d[i + 1], d[i + 2])) n++;
    return n;
  }, [sel, test]);

  check(await probe(0, 'hot') > 500, 'input spectrogram shows energy');
  await page.selectOption('.tl-track .tl-view-select', 'pitch');
  check(await probe(0, 'orange') > 20, 'input pitch view shows detected contour (orange)');
  const outSelect = page.locator('.tl-track .tl-view-select').nth(1);
  await outSelect.selectOption('pitch');
  check(await probe(1, 'green') > 20, 'output pitch view shows corrected contour (green)');
  await outSelect.selectOption('waveform');
  check(await probe(1, 'green') > 100, 'output waveform view shows peaks');

  // --- Transport: play, progress advances, pause holds ---
  await page.click('#play-btn');
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('Playing'));
  await sleep(600);
  const midProgress = await page.evaluate(() => window.__pc.playback_progress());
  check(midProgress > 0.25, `playback advances (progress=${midProgress.toFixed(3)})`);
  await page.click('#play-btn'); // pause
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('Paused'));
  const pausedAt = await page.evaluate(() => window.__pc.playback_progress());
  await sleep(300);
  const stillAt = await page.evaluate(() => window.__pc.playback_progress());
  check(Math.abs(stillAt - pausedAt) < 0.01, 'pause holds the playhead');

  // --- Upload runs the offline path and lands in stopped state ---
  await page.setInputFiles('#upload-input', toneWav);
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('saved'),
    { timeout: 30000 });
  const uploaded = await page.evaluate(() => window.__pc.recording_len());
  check(uploaded === 192000, `upload decoded the full WAV (${uploaded} samples)`);
  check(await probe(0, 'hot') > 500, 'uploaded audio rendered to the input track');

  // --- No uncaught errors anywhere in the session ---
  const fatal = errors.filter(e => !/favicon/.test(e));
  if (fatal.length) console.log('  errors:', fatal.slice(0, 5));
  check(fatal.length === 0, 'no console errors or uncaught exceptions');
} finally {
  await close();
}
finish();
