/**
 * dsp.js —— 纯前端音频信号分析（零依赖，可在浏览器与 Node 中运行）
 * 作用：把一首歌的波形解构为「推荐算法」需要的客观声学特征
 *   RMS 能量 / 谱质心（明亮度）/ 过零率 / 节拍 BPM / 调式（大调-小调）/ 色度向量
 * 再据此回归出 愉悦度 valence 与 唤醒度 arousal（Russell 情感环模型坐标）
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DSP = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  /* ---------------- 基 2 迭代 FFT（Cooley-Tukey） ---------------- */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cRe = 1, cIm = 0;
        for (let k = 0; k < half; k++) {
          const aRe = re[i + k], aIm = im[i + k];
          const bRe = re[i + k + half], bIm = im[i + k + half];
          const vRe = bRe * cRe - bIm * cIm;
          const vIm = bRe * cIm + bIm * cRe;
          re[i + k] = aRe + vRe; im[i + k] = aIm + vIm;
          re[i + k + half] = aRe - vRe; im[i + k + half] = aIm - vIm;
          const nRe = cRe * wRe - cIm * wIm;
          cIm = cRe * wIm + cIm * wRe; cRe = nRe;
        }
      }
    }
  }

  function hannWindow(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    return w;
  }

  /* ---------------- Krumhansl-Kessler 调性轮廓 ---------------- */
  const KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  function corr(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
  }

  function rotate(arr, r) {
    const out = new Array(12);
    for (let i = 0; i < 12; i++) out[i] = arr[(i - r + 12) % 12];
    return out;
  }

  function estimateKey(chroma) {
    let best = { r: 0, score: -2, mode: 'major' };
    for (let r = 0; r < 12; r++) {
      const cm = corr(chroma, rotate(KK_MAJOR, r));
      const cn = corr(chroma, rotate(KK_MINOR, r));
      if (cm > best.score) best = { r, score: cm, mode: 'major' };
      if (cn > best.score) best = { r, score: cn, mode: 'minor' };
    }
    const majBest = Math.max.apply(null, Array.from({ length: 12 }, (_, r) => corr(chroma, rotate(KK_MAJOR, r))));
    const minBest = Math.max.apply(null, Array.from({ length: 12 }, (_, r) => corr(chroma, rotate(KK_MINOR, r))));
    return {
      key: KEY_NAMES[best.r] + (best.mode === 'major' ? '' : 'm'),
      mode: best.mode,
      modeScore: clamp((majBest - minBest) * 3.2, -1, 1) // >0 偏大调(明亮) / <0 偏小调(暗淡)
    };
  }

  /**
   * 主分析入口
   * @param {Float32Array} samples 单声道 PCM，[-1,1]
   * @param {number} sampleRate
   * @param {object} opts { maxSeconds }
   */
  function analyze(samples, sampleRate, opts) {
    opts = opts || {};
    const maxSec = opts.maxSeconds || 45;
    const total = Math.min(samples.length, Math.floor(sampleRate * maxSec));
    const frame = 2048, hop = 1024;
    const win = hannWindow(frame);
    const bins = frame / 2;

    const chroma = new Array(12).fill(0);
    const flux = [];
    let rmsSum = 0, centroidSum = 0, zcrSum = 0, frames = 0;
    let prevMag = new Float64Array(bins);

    const re = new Float64Array(frame), im = new Float64Array(frame);

    for (let start = 0; start + frame <= total; start += hop) {
      let rms = 0, zc = 0;
      for (let i = 0; i < frame; i++) {
        const s = samples[start + i];
        rms += s * s;
        if (i > 0 && ((samples[start + i - 1] < 0) !== (s < 0))) zc++;
        re[i] = s * win[i]; im[i] = 0;
      }
      rms = Math.sqrt(rms / frame);
      fft(re, im);

      let magSum = 0, centNum = 0, fl = 0;
      for (let k = 1; k < bins; k++) {
        const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        const f = (k * sampleRate) / frame;
        magSum += mag;
        centNum += f * mag;
        const d = mag - prevMag[k];
        if (d > 0) fl += d;
        prevMag[k] = mag;

        if (f > 65 && f < 2100 && mag > 1e-4) {
          const pitch = 69 + 12 * Math.log2(f / 440);
          const pc = ((Math.round(pitch) % 12) + 12) % 12;
          chroma[pc] += mag;
        }
      }
      flux.push(fl);
      rmsSum += rms;
      if (magSum > 1e-6) centroidSum += centNum / magSum;
      zcrSum += zc / frame;
      frames++;
    }

    if (!frames) return null;

    /* ---- 节拍：起始点包络自相关 ---- */
    const env = flux.map(v => v);
    const envMean = env.reduce((a, b) => a + b, 0) / env.length;
    for (let i = 0; i < env.length; i++) env[i] = Math.max(0, env[i] - envMean);

    const fps = sampleRate / hop;                 // 每秒帧数 ≈ 43.07
    const lagMin = Math.floor((60 * fps) / 200);  // 200 BPM
    const lagMax = Math.floor((60 * fps) / 50);   // 50 BPM
    let bestLag = -1, bestVal = 0;
    for (let lag = lagMin; lag <= lagMax && lag < env.length; lag++) {
      let s = 0;
      for (let i = 0; i + lag < env.length; i++) s += env[i] * env[i + lag];
      s /= env.length - lag;
      // 以 120BPM 为中心的对数正态先验，抑制倍频/半频误判
      const bpmAtLag = (60 * fps) / lag;
      const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpmAtLag / 120) / 0.75, 2));
      const v = s * prior;
      if (v > bestVal) { bestVal = v; bestLag = lag; }
    }
    let bpm = bestLag > 0 ? (60 * fps) / bestLag : 0;
    while (bpm > 0 && bpm < 70) bpm *= 2;
    while (bpm > 180) bpm /= 2;

    const rmsAvg = rmsSum / frames;
    const centroidHz = centroidSum / frames;
    const zcr = zcrSum / frames;
    if (chroma.some(v => v > 0)) {
      const mx = Math.max.apply(null, chroma);
      for (let i = 0; i < 12; i++) chroma[i] /= mx || 1;
    }
    const keyInfo = estimateKey(chroma);

    /* ---- 回归到情感环坐标 ---- */
    const energy = clamp(rmsAvg / 0.22, 0, 1);
    const brightness = clamp(Math.log2(1 + centroidHz / 200) / Math.log2(1 + 6000 / 200), 0, 1);
    const tempoNorm = clamp((bpm - 60) / 110, 0, 1);

    const arousal = clamp(0.58 * energy + 0.34 * tempoNorm + 0.30 * brightness - 0.42, -1, 1);
    const valence = clamp(
      0.46 * keyInfo.modeScore + 0.52 * (brightness - 0.45) + 0.36 * (energy - 0.45),
      -1, 1
    );

    return {
      bpm: Math.round(bpm),
      rms: +rmsAvg.toFixed(4),
      energy: +energy.toFixed(3),
      centroidHz: Math.round(centroidHz),
      brightness: +brightness.toFixed(3),
      zcr: +zcr.toFixed(4),
      key: keyInfo.key,
      mode: keyInfo.mode,
      modeScore: +keyInfo.modeScore.toFixed(3),
      chroma: chroma.map(v => +v.toFixed(3)),
      valence: +valence.toFixed(3),
      arousal: +arousal.toFixed(3),
      analyzedSeconds: +(total / sampleRate).toFixed(1)
    };
  }

  return { fft, analyze, estimateKey, clamp };
});
