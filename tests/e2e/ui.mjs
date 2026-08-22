// Headless UX test for the timeline UI: record → follow-scroll, stop → fit,
// wheel zoom, seek + playhead alignment, view switching (pixel-probed),
// play/pause transport, and WAV upload. Complements run.mjs (audio pipeline
// health); this file owns the viewport/interaction behavior.
import { launch, makeChecker, sleep, toneWav } from './harness.mjs';

const PORT = 8892;
const { check, finish } = makeChecker();

const { page, errors, close } = await launch({ port: PORT });
// Underrun warnings are rate-limited and gated to real gaps; a runaway here
// means a stream was left running unfed (regression).
let underruns = 0;
page.on('console', m => { if (m.text().includes('underrun')) underruns++; });
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

  // --- Re-record: a second take must be a fresh session on the SAME audio
  // graph (regression: a new WebPitchCorrector per take left the old
  // AudioContext live — runaway sample counts and dropped-closure errors) ---
  await page.click('#record-btn');
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('Recording'),
    { timeout: 15000 });
  const earlyLen = await page.evaluate(() => window.__pc.recording_len());
  check(earlyLen < 100000, `second take starts from zero (len=${earlyLen} right after start)`);
  await sleep(2500);
  await page.click('#stop-btn');
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('saved'));
  const second = await page.evaluate(() => ({
    total: window.__pc.recording_len(),
    tlTotal: window.__tl.total,
    s0: window.__tl.s0,
  }));
  check(second.total > 48000 && second.total < 48000 * 12,
    `second take grows at a sane rate (${second.total} samples)`);
  check(second.tlTotal === second.total && second.s0 === 0,
    'timeline reset and refit for the second take');

  // Pixel-probe helper: count pixels matching a signature color.
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

  // Horizontal scroll (trackpad swipe / tilt wheel) pans the viewport.
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); // over the track, not the zoom button
  const s0Before = await page.evaluate(() => window.__tl.s0);
  await page.mouse.wheel(300, 0);
  const s0Right = await page.evaluate(() => window.__tl.s0);
  check(s0Right > s0Before, `horizontal scroll pans right (s0 ${Math.round(s0Before)} -> ${Math.round(s0Right)})`);
  await page.mouse.wheel(-100000, 0); // hard left: must clamp at 0
  const s0Clamped = await page.evaluate(() => window.__tl.s0);
  check(s0Clamped === 0, `horizontal scroll clamps at the start (s0=${s0Clamped})`);

  // Deep zoom reaches the sample level; the waveform switches from the
  // min/max envelope to points joined by thin lines and still renders.
  await page.evaluate(() => window.__tl.zoomBy(1e-9, 200));
  const sppDeep = await page.evaluate(() => window.__tl.spp);
  check(sppDeep < 0.5, `deep zoom reaches sub-sample spp (${sppDeep})`);
  check(await probe(0, 'orange') > 50, 'sample-level waveform renders points and lines');

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

  // --- View switching: shared dropdown drives both tracks; split opts out ---
  check(await probe(0, 'orange') > 100, 'default view is the waveform (orange peaks)');
  const perTrackVisible = () => page.evaluate(() =>
    [...document.querySelectorAll('.tl-view-select')].map(s => s.style.display !== 'none'));
  check((await perTrackVisible()).every(v => !v), 'per-track selectors hidden by default');

  await page.selectOption('#view-select', 'pitch');
  check(await probe(0, 'orange') > 20, 'shared dropdown: input pitch view shows contour (orange)');
  check(await probe(1, 'green') > 20, 'shared dropdown: output pitch view shows contour (green)');

  await page.check('#split-views-cb');
  check((await perTrackVisible()).every(v => v), 'split reveals per-track selectors');
  check(await page.evaluate(() => document.getElementById('view-select').disabled),
    'shared dropdown disabled while split');
  const outSelect = page.locator('.tl-track .tl-view-select').nth(1);
  await outSelect.selectOption('waveform');
  check(await probe(1, 'green') > 100, 'split: output waveform view shows peaks');
  check(await probe(0, 'orange') > 20, 'split: input keeps its own view (pitch)');

  // Un-split re-unifies both tracks to the shared selection.
  await page.uncheck('#split-views-cb');
  await page.selectOption('#view-select', 'spectrogram');
  check(await probe(1, 'hot') > 500, 'un-split re-unifies views (output spectrogram)');

  // Zooming must keep the full width covered: the old content serves as a
  // stretch-blit placeholder while the exact repaint refines progressively
  // (regression: continuous zooming showed only the first ~quarter).
  await sleep(400); // let the fit spectrogram finish its progressive fill
  const specBox = await page.locator('.tl-track-canvas').nth(1).boundingBox();
  await page.mouse.move(specBox.x + specBox.width / 2, specBox.y + specBox.height / 2);
  await page.mouse.wheel(0, -300);
  const cover = await page.evaluate(() => {
    const c = document.querySelectorAll('.tl-track-canvas')[1];
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let maxX = -1;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] + d[i + 1] + d[i + 2] > 90) maxX = Math.max(maxX, (i / 4) % c.width);
    }
    return { maxX, w: c.width };
  });
  check(cover.maxX >= cover.w - 2,
    `zoom keeps the full width covered (maxX=${cover.maxX} of ${cover.w})`);
  await page.click('.tl-zoom-btn:nth-child(3)'); // restore Fit

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
  check(underruns < 20, `no underrun log runaway (${underruns} warnings all session)`);
  const fatal = errors.filter(e => !/favicon/.test(e));
  if (fatal.length) console.log('  errors:', fatal.slice(0, 5));
  check(fatal.length === 0, 'no console errors or uncaught exceptions');
} finally {
  await close();
}
finish();
