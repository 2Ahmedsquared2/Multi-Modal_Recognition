/**
 * Extract a segment from an AudioBuffer and encode it as a WAV Blob.
 *
 * All processing is done in-browser — no server round-trip needed.
 */

/** Extract a segment and return a mono WAV Blob. */
export function extractSegmentAsWav(
  buffer: AudioBuffer,
  startSec: number,
  durationSec: number,
): Blob {
  const sampleRate = buffer.sampleRate;
  const startSample = Math.floor(startSec * sampleRate);
  const numSamples = Math.floor(durationSec * sampleRate);

  // Mix all channels to mono
  const mono = new Float32Array(numSamples);
  const numChannels = buffer.numberOfChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < numSamples; i++) {
      const idx = startSample + i;
      if (idx < channelData.length) {
        mono[i] += channelData[idx] / numChannels;
      }
    }
  }

  return encodeWav(mono, sampleRate);
}

/** Encode a Float32Array of mono samples into a WAV Blob. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const numSamples = samples.length;
  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = bytesPerSample; // mono
  const dataSize = numSamples * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);          // chunk size
  view.setUint16(20, 1, true);           // PCM format
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, sampleRate, true);  // sample rate
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);  // block align
  view.setUint16(34, 16, true);          // bits per sample

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert float samples to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/**
 * Downsample an AudioBuffer to ~500 points (min/max envelope)
 * for lightweight waveform rendering.
 */
export function getWaveformEnvelope(
  buffer: AudioBuffer,
  targetPoints: number = 800,
): number[] {
  // Mix to mono first
  const length = buffer.length;
  const mono = new Float32Array(length);
  const numChannels = buffer.numberOfChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] / numChannels;
    }
  }

  if (length <= targetPoints) {
    return Array.from(mono);
  }

  const nBuckets = targetPoints / 2;
  const bucketSize = length / nBuckets;
  const envelope: number[] = [];

  for (let i = 0; i < nBuckets; i++) {
    const start = Math.floor(i * bucketSize);
    const end = Math.floor((i + 1) * bucketSize);
    let mn = Infinity, mx = -Infinity;
    for (let j = start; j < end; j++) {
      if (mono[j] < mn) mn = mono[j];
      if (mono[j] > mx) mx = mono[j];
    }
    envelope.push(mn, mx);
  }

  return envelope;
}
