// WAV encode/decode + download helpers (mono, 16-bit PCM or float32).

export function encodeWav(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s * 0x7fff, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

export function decodeWav(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    let offset = 12;
    while (offset < view.byteLength - 8) {
        const id = String.fromCharCode(
            view.getUint8(offset),
            view.getUint8(offset + 1),
            view.getUint8(offset + 2),
            view.getUint8(offset + 3),
        );
        const size = view.getUint32(offset + 4, true);
        if (id === 'data') {
            const bitsPerSample = view.getUint16(34, true);
            const numChannels = view.getUint16(22, true);
            const samples = [];
            const dataStart = offset + 8;
            if (bitsPerSample === 16) {
                for (let i = 0; i < size; i += 2 * numChannels) {
                    samples.push(view.getInt16(dataStart + i, true) / 0x7fff);
                }
            } else if (bitsPerSample === 32) {
                for (let i = 0; i < size; i += 4 * numChannels) {
                    samples.push(view.getFloat32(dataStart + i, true));
                }
            }
            return new Float32Array(samples);
        }
        offset += 8 + size;
    }
    return null;
}

export function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
