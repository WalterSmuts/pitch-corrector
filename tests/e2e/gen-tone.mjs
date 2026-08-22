// Generate tone.wav: mono 48kHz 16-bit, ~210Hz (between G#3 and A3) with a
// couple of harmonics so YIN has a clear fundamental. Fed to Chromium as the
// fake microphone via --use-file-for-fake-audio-capture.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const sr = 48000,
  dur = 4.0,
  f = 210.0;
const n = Math.floor(sr * dur);
const buf = Buffer.alloc(44 + n * 2);
let o = 0;
const s = (str) => {
  buf.write(str, o);
  o += str.length;
};
const u32 = (v) => {
  buf.writeUInt32LE(v, o);
  o += 4;
};
const u16 = (v) => {
  buf.writeUInt16LE(v, o);
  o += 2;
};
s('RIFF');
u32(36 + n * 2);
s('WAVE');
s('fmt ');
u32(16);
u16(1);
u16(1);
u32(sr);
u32(sr * 2);
u16(2);
u16(16);
s('data');
u32(n * 2);
for (let i = 0; i < n; i++) {
  const t = i / sr;
  let v =
    0.6 * Math.sin(2 * Math.PI * f * t) +
    0.25 * Math.sin(2 * Math.PI * 2 * f * t) +
    0.12 * Math.sin(2 * Math.PI * 3 * f * t);
  v *= 0.5;
  buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32767))), o);
  o += 2;
}
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'tone.wav');
fs.writeFileSync(out, buf);
console.log('wrote', out, n, 'frames', `${f}Hz ${sr}Hz mono 16-bit`);
