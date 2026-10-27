/**
 * emotion-engine.js —— 本地人脸情绪识别引擎（不上传任何图像）
 *
 * 算法总览（三层串联 + 一层时序滤波）：
 *   L1 人脸检测   Tiny-YOLO 结构的轻量 CNN（TinyFaceDetector），输出人脸框 + 5 关键点
 *   L2 表情分类   64×64 灰度输入的 7 类 CNN 分类器（FaceExpressionNet），输出 softmax 概率
 *   L3 几何先验   68 点 FaceLandmark → 5 个尺度归一化几何特征 → 线性打分 → softmax 先验
 *   L4 贝叶斯融合 log p(c) ∝ (1-λ)·log p_cnn(c) + λ·log p_geo(c)，再叠加 EMA 时序平滑
 *
 * 依赖：face-api.js（TensorFlow.js 后端，全部推理在本机浏览器完成）
 */
(function (root) {
  'use strict';

  /* ============ 情绪元数据（7 类 Ekman 基本情绪 + 平静） ============ */
  const EMOTIONS = [
    { key: 'happy',     cn: '高兴', color: '#f6a723', valence: 0.85,  arousal: 0.55,  desc: '积极 · 高唤醒' },
    { key: 'sad',       cn: '悲伤', color: '#5b8def', valence: -0.75, arousal: -0.45, desc: '消极 · 低唤醒' },
    { key: 'angry',     cn: '愤怒', color: '#e5484d', valence: -0.62, arousal: 0.78,  desc: '消极 · 高唤醒' },
    { key: 'surprised', cn: '惊讶', color: '#8b5cf6', valence: 0.30,  arousal: 0.85,  desc: '中性偏积极 · 高唤醒' },
    { key: 'fearful',   cn: '恐惧', color: '#7c6a9c', valence: -0.55, arousal: 0.72,  desc: '消极 · 高唤醒' },
    { key: 'disgusted', cn: '厌恶', color: '#6b8e23', valence: -0.68, arousal: 0.30,  desc: '消极 · 中唤醒' },
    { key: 'neutral',   cn: '平静', color: '#7a8699', valence: 0.02,  arousal: -0.10, desc: '中性 · 低唤醒' }
  ];
  const EMO_KEYS = EMOTIONS.map(e => e.key);
  const EMO_MAP = Object.fromEntries(EMOTIONS.map(e => [e.key, e]));

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const EPS = 1e-7;

  /* ============ 几何特征（L3） ============ */
  // dlib / iBUG 68 点索引约定
  const P = {
    jawL: 0, jawR: 16,
    browLOuter: 17, browLInner: 21,
    browROuter: 22, browRInner: 26,
    noseTip: 30,
    eyeLOuter: 36, eyeLInner: 39, eyeLTop: 37, eyeLBottom: 41,
    eyeROuter: 42, eyeRInner: 45, eyeRTop: 43, eyeRBottom: 47,
    mouthL: 48, mouthR: 54, mouthTopIn: 51, mouthBotIn: 57,
    lipTopIn: 62, lipBotIn: 66
  };

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const softmax = (arr, T) => {
    T = T || 1;
    const mx = Math.max.apply(null, arr);
    const ex = arr.map(v => Math.exp((v - mx) / T));
    const s = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map(v => v / s);
  };

  /**
   * 从 68 点计算 6 个尺度无关的几何量
   * 全部除以瞳距（两眼外眼角间距）做归一化，从而与人脸远近无关
   */
  function geometricFeatures(pts) {
    const io = Math.max(dist(pts[P.eyeLOuter], pts[P.eyeROuter]), 1e-6);

    // 1) 嘴角上扬度：(口内中心 y - 嘴角中心 y) / 瞳距，图像 y 向下为正 → 上扬为正
    const corner = mid(pts[P.mouthL], pts[P.mouthR]);
    const mouthCenter = mid(pts[P.mouthTopIn], pts[P.mouthBotIn]);
    const smile = (mouthCenter.y - corner.y) / io;

    // 2) 嘴部张开度：上下内唇间距 / 嘴宽
    const mouthWidth = Math.max(dist(pts[P.mouthL], pts[P.mouthR]), 1e-6);
    const mouthOpen = dist(pts[P.lipTopIn], pts[P.lipBotIn]) / mouthWidth;

    // 3) 眉眼距离（双侧平均）→ 眉毛抬高
    const eyeTopL = mid(pts[P.eyeLTop], pts[38]);
    const eyeTopR = mid(pts[P.eyeRTop], pts[44]);
    const browL = mid(pts[P.browLOuter], pts[P.browLInner]);
    const browR = mid(pts[P.browROuter], pts[P.browRInner]);
    const browRaise = ((eyeTopL.y - browL.y) + (eyeTopR.y - browR.y)) / 2 / io;

    // 4) 内眉抬升（悲伤/恐惧的典型特征）
    const innerBrowUp = ((pts[P.eyeLInner].y + pts[P.eyeRInner].y) / 2 - (pts[P.browLInner].y + pts[P.browRInner].y) / 2) / io;

    // 5) 眼睛开合度 EAR（双侧平均）
    const earL = (dist(pts[37], pts[41]) + dist(pts[38], pts[40])) / (2 * Math.max(dist(pts[36], pts[39]), 1e-6));
    const earR = (dist(pts[43], pts[47]) + dist(pts[44], pts[46])) / (2 * Math.max(dist(pts[42], pts[45]), 1e-6));
    const eyeOpen = (earL + earR) / 2;

    // 6) 嘴宽比（咧嘴 / 抿嘴）
    const mouthWide = mouthWidth / io;

    return { smile, mouthOpen, browRaise, innerBrowUp, eyeOpen, mouthWide, interocular: io };
  }

  // 经验均值与尺度（用于 z 标准化），取自中性表情的人脸统计
  const F_MEAN = { smile: 0.118, mouthOpen: 0.06, browRaise: 0.255, innerBrowUp: 0.145, eyeOpen: 0.31, mouthWide: 0.72 };
  const F_STD  = { smile: 0.075, mouthOpen: 0.05, browRaise: 0.045, innerBrowUp: 0.035, eyeOpen: 0.06,  mouthWide: 0.12 };
  const z = (f, k) => clamp((f[k] - F_MEAN[k]) / F_STD[k], -3, 3);

  /**
   * 几何打分：每个情绪类一个线性判别函数（系数经人工设定，可解释、可调）
   * 得分 → softmax(T=1.15) 得到几何先验分布
   */
  function geometricPrior(f) {
    const s = z(f, 'smile'), o = z(f, 'mouthOpen'), b = z(f, 'browRaise');
    const ib = z(f, 'innerBrowUp'), e = z(f, 'eyeOpen'), w = z(f, 'mouthWide');

    const score = {
      happy:     2.10 * s + 0.55 * w + 0.25 * e - 0.90,
      sad:      -1.90 * s + 1.55 * ib - 0.45 * w - 0.30 * e - 0.75,
      angry:    -0.85 * s - 1.35 * b + 0.85 * w - 0.45 * o - 0.55,
      surprised: 1.85 * o + 1.45 * b + 0.95 * e + 0.30 * w - 1.05,
      fearful:   1.15 * o + 0.95 * b + 0.85 * ib + 0.60 * e - 0.50 * s - 1.15,
      disgusted: -0.70 * s - 1.10 * b + 0.70 * w + 0.35 * ib - 0.85
    };
    // 平静：与全部判别器"都不活跃"的程度成正比
    const active = Math.max(score.happy, score.surprised, Math.abs(score.sad), Math.abs(score.angry));
    score.neutral = 0.55 - 0.75 * active;

    const vec = EMO_KEYS.map(k => score[k]);
    const probs = softmax(vec, 1.15);
    return Object.fromEntries(EMO_KEYS.map((k, i) => [k, probs[i]]));
  }

  /* ============ 质量评估 ============ */
  function blurScore(input, box) {
    // 人脸区域灰度裁剪 → 3×3 拉普拉斯卷积 → 方差（越大越清晰）
    const size = 64;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    try {
      ctx.drawImage(input, box.x, box.y, box.width, box.height, 0, 0, size, size);
    } catch (e) { return 0; }
    const d = ctx.getImageData(0, 0, size, size).data;
    const gray = new Float32Array(size * size);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    }
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const i = y * size + x;
        const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - size] - gray[i + size];
        sum += lap; sum2 += lap * lap; n++;
      }
    }
    const mean = sum / n;
    return Math.max(0, sum2 / n - mean * mean);
  }

  function evaluateQuality(input, det, pts) {
    const W = input.videoWidth || input.naturalWidth || input.width || 640;
    const H = input.videoHeight || input.naturalHeight || input.height || 480;
    const faceRatio = det.box.width / W;

    // 偏航角代理：鼻尖相对两眼中心的水平偏移 / 瞳距
    const io = Math.max(dist(pts[P.eyeLOuter], pts[P.eyeROuter]), 1e-6);
    const eyeC = mid(pts[P.eyeLOuter], pts[P.eyeROuter]);
    const yaw = Math.abs(pts[P.noseTip].x - eyeC.x) / io;

    const blur = blurScore(input, det.box);
    const blurN = clamp(blur / 220, 0, 1);

    let score = 1;
    if (faceRatio < 0.18) score *= 0.55;
    else if (faceRatio < 0.28) score *= 0.85;
    if (yaw > 0.30) score *= 0.7;
    score *= 0.35 + 0.65 * blurN;
    score = clamp(score, 0, 1);

    const tips = [];
    if (faceRatio < 0.18) tips.push('人脸偏小，请靠近一些');
    if (yaw > 0.30) tips.push('请正对镜头');
    if (blurN < 0.35) tips.push('画面偏模糊，注意光线');

    return { score: +score.toFixed(2), faceRatio: +faceRatio.toFixed(2), blur: +blurN.toFixed(2), yaw: +yaw.toFixed(2), tips };
  }

  /* ============ 引擎 ============ */
  class EmotionEngine {
    constructor(opts) {
      opts = opts || {};
      this.modelPath = opts.modelPath || 'assets/models';
      this.fusionLambda = opts.fusionLambda != null ? opts.fusionLambda : 0.22; // 几何先验权重
      this.emaAlpha = opts.emaAlpha || 0.42;
      this.smoothing = true;
      this.ready = false;
      this.backend = '-';
      this._ema = null;
      this._frames = 0;
    }

    async load() {
      if (typeof faceapi === 'undefined') throw new Error('face-api.js 未加载');
      try {
        await faceapi.tf.setBackend('webgl');
        await faceapi.tf.ready();
        this.backend = faceapi.tf.getBackend();
      } catch (e) {
        await faceapi.tf.setBackend('cpu');
        this.backend = 'cpu';
      }
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(this.modelPath),
        faceapi.nets.faceExpressionNet.loadFromUri(this.modelPath),
        faceapi.nets.faceLandmark68Net.loadFromUri(this.modelPath)
      ]);
      this.ready = true;
      return this;
    }

    resetSmoothing() { this._ema = null; this._frames = 0; }

    /**
     * 分析一帧
     * @param {object} opts { inputSize, threshold, multiScale, sizes }
     *   multiScale=true 时按 sizes 阶梯依次尝试（特写/远景等极端人脸占比更鲁棒）
     * @returns {null | object} 见 buildResult
     */
    async analyze(input, opts) {
      opts = opts || {};
      if (!this.ready) throw new Error('模型尚未加载完成');
      const sizes = opts.multiScale
        ? (opts.sizes || [416, 608, 320, 224])
        : [opts.inputSize || 320];
      const threshold = opts.threshold || 0.42;

      for (const size of sizes) {
        const options = new faceapi.TinyFaceDetectorOptions({ inputSize: size, scoreThreshold: threshold });
        const res = await faceapi
          .detectSingleFace(input, options)
          .withFaceLandmarks()
          .withFaceExpressions();
        if (res) return this.buildResult(input, res, opts);
      }
      this._decaySmoothing();
      return null;
    }

    /** 同步构造结果（便于复用已检测结果） */
    buildResult(input, res, opts) {
      opts = opts || {};
      const cnnRaw = res.expressions;
      const cnn = Object.fromEntries(EMO_KEYS.map(k => [k, cnnRaw[k] || 0]));

      const pts = res.landmarks.positions.map(p => ({ x: p.x, y: p.y }));
      const geo = geometricFeatures(pts);
      const geoPrior = geometricPrior(geo);

      // ---- L4 对数线性贝叶斯融合 ----
      const lam = this.fusionLambda;
      const fusedRaw = EMO_KEYS.map(k =>
        (1 - lam) * Math.log((cnn[k] || 0) + EPS) + lam * Math.log((geoPrior[k] || 0) + EPS)
      );
      const mx = Math.max.apply(null, fusedRaw);
      const ex = fusedRaw.map(v => Math.exp(v - mx));
      const sum = ex.reduce((a, b) => a + b, 0) || 1;
      let fused = Object.fromEntries(EMO_KEYS.map((k, i) => [k, ex[i] / sum]));

      // ---- 质量门控 ----
      const quality = evaluateQuality(input, res.detection, pts);

      // ---- 时序 EMA 平滑 ----
      if (this.smoothing) {
        const topConf = Math.max.apply(null, EMO_KEYS.map(k => fused[k]));
        const alpha = clamp(this.emaAlpha * (0.55 + 0.9 * topConf), 0.18, 0.72);
        if (!this._ema) {
          this._ema = Object.fromEntries(EMO_KEYS.map(k => [k, fused[k]]));
        } else {
          let acc = 0;
          for (const k of EMO_KEYS) {
            this._ema[k] = alpha * fused[k] + (1 - alpha) * this._ema[k];
            acc += this._ema[k];
          }
          for (const k of EMO_KEYS) this._ema[k] /= acc || 1;
        }
        fused = Object.assign({}, this._ema);
      }
      this._frames++;

      return buildResult(input, res, cnn, geo, geoPrior, fused, quality, this._frames);
    }

    _decaySmoothing() {
      if (!this._ema) return;
      for (const k of EMO_KEYS) this._ema[k] = this._ema[k] * 0.82 + 1 / 7 * 0.18;
      this._frames++;
    }
  }

  function buildResult(input, res, cnn, geo, geoPrior, fused, quality, frames) {
    const sorted = EMO_KEYS.slice().sort((a, b) => fused[b] - fused[a]);
    const dominant = sorted[0];
    const second = sorted[1];
    const conf = fused[dominant];
    const margin = conf - fused[second];

    // 情绪坐标（Russell 环形模型）：7 类概率的加权质心
    let v = 0, a = 0;
    for (const k of EMO_KEYS) { v += fused[k] * EMO_MAP[k].valence; a += fused[k] * EMO_MAP[k].arousal; }

    // 置信度：主导概率与领先幅度共同决定，再乘质量系数
    const confidence = clamp((conf * 0.65 + Math.min(1, margin / 0.35) * 0.35) * (0.6 + 0.4 * quality.score), 0, 1);

    const W = input.videoWidth || input.naturalWidth || input.width || 0;
    const H = input.videoHeight || input.naturalHeight || input.height || 0;

    return {
      found: true,
      box: { x: res.detection.box.x, y: res.detection.box.y, w: res.detection.box.width, h: res.detection.box.height, score: res.detection.score },
      frame: { w: W, h: H },
      landmarks: res.landmarks.positions.map(p => ({ x: p.x, y: p.y })),
      cnn, geoPrior, fused,
      geometry: geo,
      dominant, dominantCn: EMO_MAP[dominant].cn, dominantColor: EMO_MAP[dominant].color,
      confidence: +confidence.toFixed(3),
      margin: +margin.toFixed(3),
      valence: +v.toFixed(3),
      arousal: +a.toFixed(3),
      quality,
      frames
    };
  }

  /* ============ 画布叠加绘制 ============ */
  function drawOverlay(ctx, result, w, h, style) {
    style = style || {};
    ctx.clearRect(0, 0, w, h);
    if (!result || !result.found) return;

    // 与 video/img 的 object-fit:contain 保持一致：等比缩放 + 居中留边
    const fw = result.frame.w || w, fh = result.frame.h || h;
    const s = Math.min(w / fw, h / fh);
    const dx = (w - fw * s) / 2, dy = (h - fh * s) / 2;
    const b = result.box;
    const x = dx + b.x * s, y = dy + b.y * s, bw = b.w * s, bh = b.h * s;
    const color = result.dominantColor;

    // 四角框
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, w / 300);
    const arm = Math.min(bw, bh) * 0.18;
    ctx.beginPath();
    ctx.moveTo(x, y + arm); ctx.lineTo(x, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x + bw - arm, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + arm);
    ctx.moveTo(x + bw, y + bh - arm); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw - arm, y + bh);
    ctx.moveTo(x + arm, y + bh); ctx.lineTo(x, y + bh); ctx.lineTo(x, y + bh - arm);
    ctx.stroke();

    // 关键点
    if (style.landmarks && result.landmarks) {
      ctx.fillStyle = 'rgba(255,255,255,.9)';
      const r = Math.max(1.4, w / 420);
      for (const p of result.landmarks) {
        ctx.beginPath();
        ctx.arc(dx + p.x * s, dy + p.y * s, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 标签
    const label = `${result.dominantCn} ${(result.confidence * 100).toFixed(0)}%`;
    const fs = Math.max(12, w / 34);
    ctx.font = `600 ${fs}px -apple-system,"Segoe UI","PingFang SC",sans-serif`;
    const tw = ctx.measureText(label).width + fs * 0.9;
    const th = fs * 1.75;
    const ly = Math.max(0, y - th - 6);
    ctx.fillStyle = color;
    roundRect(ctx, x, ly, tw, th, th / 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + fs * 0.45, ly + th / 2);
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, h / 2, w / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  root.EmotionEngine = EmotionEngine;
  root.EMOTIONS = EMOTIONS;
  root.EMO_KEYS = EMO_KEYS;
  root.EMO_MAP = EMO_MAP;
  root.Geo = { geometricFeatures, geometricPrior, evaluateQuality };
  root.EmotionViz = { drawOverlay, roundRect };
})(typeof window !== 'undefined' ? window : globalThis);
