/**
 * library.js —— 曲库层
 *  A. 本地曲库：随项目一起下载到 assets/music 的音频文件，浏览器内用 Web Audio 解码 →
 *     dsp.js 提取 BPM / 调式 / 能量 / 明亮度，并回归出 (valence, arousal)
 *  B. 在线曲库：Apple iTunes Search API（公开、免密钥、支持 CORS 与 JSONP，返回 30s 试听片段）
 *  C. 用户本地导入：拖入自己的 mp3，同样在本地完成特征分析，不会上传
 */
(function (root) {
  'use strict';

  /* ---------------- A. 本地曲库清单 ---------------- */
  const LOCAL_TRACKS = [
    { id: 'local-1', title: 'SoundHelix #1', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-1.mp3', genre: '电子 / 氛围' },
    { id: 'local-2', title: 'SoundHelix #2', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-2.mp3', genre: '电子 / 氛围' },
    { id: 'local-3', title: 'SoundHelix #3', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-3.mp3', genre: '电子 / 氛围' },
    { id: 'local-4', title: 'SoundHelix #4', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-4.mp3', genre: '电子 / 氛围' },
    { id: 'local-5', title: 'SoundHelix #5', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-5.mp3', genre: '电子 / 氛围' },
    { id: 'local-6', title: 'SoundHelix #6', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-6.mp3', genre: '电子 / 氛围' },
    { id: 'local-7', title: 'SoundHelix #7', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-7.mp3', genre: '电子 / 氛围' },
    { id: 'local-8', title: 'SoundHelix #8', artist: 'SoundHelix（算法生成）', album: '离线演示曲库', file: 'assets/music/SoundHelix-Song-8.mp3', genre: '电子 / 氛围' }
  ].map(t => Object.assign(t, { source: 'local', url: t.file, lang: 'instrumental', cover: null }));

  const CACHE_KEY = 'emosong.audiofeat.v2';

  function loadCache() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY) || '{}'); } catch (e) { return {}; }
  }
  function saveCache(c) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(c)); } catch (e) { /* 忽略配额错误 */ }
  }

  /** 依声学特征生成可读标签，避免人工硬编码情绪标签 */
  function deriveTags(f) {
    const tags = [];
    if (f.valence >= 0.18) tags.push('happy', 'bright');
    else if (f.valence <= -0.18) tags.push('sad', 'dark');
    else tags.push('neutral');
    if (f.arousal >= 0.35) tags.push('energetic');
    else if (f.arousal <= -0.15) tags.push('calm');
    if (f.mode === 'major') tags.push('healing');
    if (f.brightness > 0.62) tags.push('bright');
    if (f.brightness < 0.35) tags.push('mellow');
    return Array.from(new Set(tags));
  }

  function featuresToTrack(track, f) {
    return Object.assign({}, track, {
      bpm: f.bpm,
      energy: f.energy,
      brightness: f.brightness,
      valence: f.valence,
      arousal: f.arousal,
      key: f.key,
      mode: f.mode,
      tags: deriveTags(f),
      analyzed: true,
      duration: f.duration || f.analyzedSeconds
    });
  }

  /* ---------------- 音频解码 + 分析 ---------------- */
  let sharedCtx = null;
  function getAudioContext() {
    if (!sharedCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      sharedCtx = new AC();
    }
    return sharedCtx;
  }

  async function analyzeAudioUrl(url) {
    const ctx = getAudioContext();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('音频读取失败 ' + resp.status);
    const buf = await resp.arrayBuffer();
    const audio = await ctx.decodeAudioData(buf);
    const chs = audio.numberOfChannels;
    const len = Math.min(audio.length, Math.floor(audio.sampleRate * 45));
    const mono = new Float32Array(len);
    for (let c = 0; c < chs; c++) {
      const data = audio.getChannelData(c);
      for (let i = 0; i < len; i++) mono[i] += data[i] / chs;
    }
    const f = DSP.analyze(mono, audio.sampleRate, { maxSeconds: 45 });
    if (f) f.duration = +audio.duration.toFixed(1);
    return f;
  }

  /**
   * 分析本地曲库（带缓存，首启约 8~15s，之后秒开）
   * @param {(done:number,total:number,label:string)=>void} onProgress
   */
  async function analyzeLocalLibrary(onProgress, force) {
    const cache = loadCache();
    const out = [];
    for (let i = 0; i < LOCAL_TRACKS.length; i++) {
      const t = LOCAL_TRACKS[i];
      onProgress && onProgress(i, LOCAL_TRACKS.length, t.title);
      let f = force ? null : cache[t.id];
      if (!f) {
        try {
          f = await analyzeAudioUrl(t.file);
          if (f) { cache[t.id] = f; saveCache(cache); }
        } catch (e) {
          console.warn('[library] 分析失败', t.title, e);
        }
      }
      if (f) out.push(featuresToTrack(t, f));
    }
    onProgress && onProgress(LOCAL_TRACKS.length, LOCAL_TRACKS.length, '完成');
    return out;
  }

  /** 用户导入的本地文件（File 对象） */
  async function importLocalFiles(files, startIndex) {
    const out = [];
    let idx = startIndex || 0;
    for (const file of files) {
      if (!/^audio\//.test(file.type) && !/\.(mp3|m4a|wav|ogg|flac)$/i.test(file.name)) continue;
      idx++;
      const url = URL.createObjectURL(file);
      const base = {
        id: 'user-' + idx + '-' + file.name,
        title: file.name.replace(/\.[^.]+$/, ''),
        artist: '本地导入',
        album: file.name,
        file: url,
        url,
        source: 'local',
        genre: '本地音频',
        lang: 'unknown',
        cover: null
      };
      try {
        const f = await analyzeAudioUrl(url);
        out.push(f ? featuresToTrack(base, f) : Object.assign(base, { valence: 0, arousal: 0, energy: 0.5, bpm: 100, tags: ['neutral'], analyzed: false }));
      } catch (e) {
        out.push(Object.assign(base, { valence: 0, arousal: 0, energy: 0.5, bpm: 100, tags: ['neutral'], analyzed: false }));
      }
    }
    return out;
  }

  /* ---------------- B. iTunes 在线检索 ---------------- */
  function jsonp(url, timeout) {
    return new Promise((resolve, reject) => {
      const cb = '__itunes_cb_' + Math.random().toString(36).slice(2);
      const script = document.createElement('script');
      const timer = setTimeout(() => { cleanup(); reject(new Error('JSONP 超时')); }, timeout || 8000);
      function cleanup() {
        clearTimeout(timer);
        delete window[cb];
        if (script.parentNode) script.parentNode.removeChild(script);
      }
      window[cb] = data => { cleanup(); resolve(data); };
      script.onerror = () => { cleanup(); reject(new Error('JSONP 失败')); };
      script.src = url + (url.indexOf('?') >= 0 ? '&' : '?') + 'callback=' + cb;
      document.head.appendChild(script);
    });
  }

  async function searchStore(term, country, limit) {
    const url = `https://itunes.apple.com/search?term=${encodeURIComponent(term)}&media=music&entity=song&limit=${limit}&country=${country}`;
    let data = null;
    try {
      // 8 秒超时：网络不可达时快速降级到本地曲库，而不是无限等待
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8000);
      const r = await fetch(url, { mode: 'cors', signal: ctrl.signal });
      clearTimeout(timer);
      if (r.ok) data = await r.json();
    } catch (e) { /* 超时或 CORS 失败，退化为 JSONP */ }
    if (!data) {
      try { data = await jsonp(url, 8000); } catch (e) { return []; }
    }
    return (data.results || []).filter(r => r.previewUrl && r.kind === 'song').map(r => ({
      id: 'itunes-' + r.trackId,
      title: r.trackName,
      artist: r.artistName,
      album: r.collectionName,
      cover: (r.artworkUrl100 || '').replace('100x100bb', '300x300bb'),
      url: r.previewUrl,
      source: 'online',
      genre: r.primaryGenreName,
      lang: /[一-龥]/.test(r.trackName + r.artistName) ? 'zh' : 'en',
      duration: r.trackTimeMillis ? Math.round(r.trackTimeMillis / 1000) : 30,
      storeUrl: r.trackViewUrl,
      store: country.toUpperCase(),
      trial: true
    }));
  }

  /**
   * 多关键词 × 多区搜索并合并去重
   * 中文检索走 HK 区（国区曲库无试听片段），英文检索走 US 区
   */
  async function searchOnline(queries, opts) {
    opts = opts || {};
    const limit = opts.limit || 12;
    const tasks = [];
    queries.forEach((q, i) => {
      const isCn = /[一-龥]/.test(q);
      tasks.push(searchStore(q, isCn ? 'hk' : 'us', limit).then(list => list.map(x => Object.assign(x, { matchedQuery: q }))));
    });
    const groups = await Promise.all(tasks);
    const seen = new Set();
    const merged = [];
    for (const list of groups) {
      for (const t of list) {
        // 只按歌名去重：同名不同艺人的翻唱/合集对演示体验弊大于利
        const k = (t.title || '').toLowerCase().trim();
        if (!k || seen.has(k)) continue;
        seen.add(k);
        merged.push(t);
      }
    }
    return merged;
  }

  /** 在线曲目补齐情绪特征（曲风先验 + 关键词微调） */
  function enrichOnline(track) {
    if (track.valence != null && track.tags) return track;
    const m = Recommender.inferTrackMood(track);
    return Object.assign({}, track, {
      valence: m.valence, arousal: m.arousal, energy: m.energy,
      bpm: m.bpm, tags: m.tags, analyzed: false
    });
  }

  /* ---------------- C. 封面渐变（无图时的确定性配色） ---------------- */
  function coverGradient(track) {
    const v = track.valence == null ? 0.1 : track.valence;
    const a = track.arousal == null ? 0.1 : track.arousal;
    const hue1 = v >= 0 ? 38 + (1 - v) * 20 : 210 - v * 30;      // 暖橙 ↔ 冷蓝
    const hue2 = v >= 0 ? 330 : 260;
    const sat = 62 + Math.round(Math.abs(a) * 22);
    const light = 62 - Math.round(Math.abs(a) * 8);
    return `linear-gradient(135deg, hsl(${hue1} ${sat}% ${light}%) 0%, hsl(${hue2} ${sat - 12}% ${light - 16}%) 100%)`;
  }

  root.MusicLibrary = {
    LOCAL_TRACKS, analyzeLocalLibrary, importLocalFiles, analyzeAudioUrl,
    searchOnline, enrichOnline, coverGradient, deriveTags, getAudioContext
  };
})(typeof window !== 'undefined' ? window : globalThis);
