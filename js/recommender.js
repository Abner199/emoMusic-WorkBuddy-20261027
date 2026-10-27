/**
 * recommender.js —— 基于情感环（Russell Circumplex）的歌曲推荐引擎
 *
 * 推荐流程（四阶段）：
 *   S1 情绪画像  →  7 类概率加权得到 (valence, arousal) 坐标
 *   S2 目标决策  →  「镜像」直接跟随；「修复」向目标锚点迁移；「智能」按效价自动选择
 *   S3 多因子打分 →  S = Σ wᵢ·Sᵢ（情感距离/情绪标签/节奏能量/偏好新颖/音源质量）
 *   S4 MMR 重排  →  argmax[λ·S − (1−λ)·maxSim]，兼顾相关性与歌单多样性
 */
(function (root) {
  'use strict';

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  /* ---------- 每种情绪的「修复锚点」：把听者引向何处 ---------- */
  const REPAIR_ANCHOR = {
    happy:     { v: 0.82, a: 0.62, tag: '保持并放大愉悦' },
    neutral:   { v: 0.55, a: 0.30, tag: '注入一点活力' },
    sad:       { v: 0.58, a: 0.10, tag: '先共情、再托举' },
    angry:     { v: 0.40, a: -0.15, tag: '降唤醒、泄压' },
    fearful:   { v: 0.45, a: -0.10, tag: '稳定安全感' },
    disgusted: { v: 0.45, a: 0.05,  tag: '转移注意力' },
    surprised: { v: 0.60, a: 0.35,  tag: '承接兴奋感' }
  };

  /* ---------- 情绪 → 在线检索意图 ---------- */
  const MOOD_QUERY = {
    happy:     { cn: ['欢快', '开心 歌单', '阳光 流行'], en: ['happy upbeat', 'feel good pop', 'sunshine songs'] },
    sad:       { cn: ['治愈 抒情', '温柔 慢歌', '温暖 治愈系'], en: ['healing acoustic', 'gentle sad songs', 'comfort songs'] },
    angry:     { cn: ['放松 纯音乐', '舒缓 钢琴', '安静'], en: ['calm piano', 'chill instrumental', 'relaxing ambient'] },
    surprised: { cn: ['节奏感 电子', '活力 流行'], en: ['energetic electronic', 'exciting pop', 'upbeat dance'] },
    fearful:   { cn: ['安静 轻音乐', '安全感 钢琴'], en: ['soothing piano', 'soft ambient', 'peaceful music'] },
    disgusted: { cn: ['清新 民谣', '轻快 民谣'], en: ['fresh folk', 'light acoustic', 'breezy indie'] },
    neutral:   { cn: ['流行 精选', '轻音乐', '日常 歌单'], en: ['easy listening', 'chill pop', 'daily mix'] }
  };

  /* ---------- 曲风先验：在线曲目缺音频特征时的兜底 ---------- */
  const GENRE_PRIOR = {
    '流行': { v: 0.25, a: 0.15, e: 0.6 }, 'Pop': { v: 0.30, a: 0.18, e: 0.62 },
    '摇滚': { v: -0.05, a: 0.62, e: 0.85 }, 'Rock': { v: -0.05, a: 0.62, e: 0.85 },
    '民谣': { v: 0.05, a: -0.25, e: 0.35 }, 'Folk': { v: 0.05, a: -0.25, e: 0.35 },
    '电子': { v: 0.15, a: 0.55, e: 0.8 }, 'Electronic': { v: 0.15, a: 0.55, e: 0.8 },
    'Dance': { v: 0.40, a: 0.68, e: 0.85 },
    '古典': { v: 0.10, a: -0.20, e: 0.35 }, 'Classical': { v: 0.10, a: -0.20, e: 0.35 },
    '爵士': { v: 0.12, a: 0.05, e: 0.45 }, 'Jazz': { v: 0.12, a: 0.05, e: 0.45 },
    '嘻哈': { v: 0.10, a: 0.45, e: 0.75 }, 'Hip-Hop': { v: 0.10, a: 0.45, e: 0.75 },
    'R&B': { v: 0.18, a: 0.20, e: 0.55 },
    '国语流行': { v: 0.22, a: 0.10, e: 0.55 }, '國語流行樂': { v: 0.22, a: 0.10, e: 0.55 },
    '粤语流行': { v: 0.20, a: 0.08, e: 0.55 },
    '原声': { v: 0.15, a: -0.05, e: 0.45 }, 'Soundtrack': { v: 0.15, a: -0.05, e: 0.45 },
    'Ambient': { v: 0.05, a: -0.55, e: 0.25 },
    'New Age': { v: 0.12, a: -0.40, e: 0.30 }
  };
  const DEFAULT_PRIOR = { v: 0.10, a: 0.05, e: 0.55 };

  const MOOD_KEYWORDS = [
    { w: 0.55, tag: 'happy',   words: ['happy', 'joy', 'sun', 'smile', 'fun', 'party', 'dance', '欢', '快', '笑'] },
    { w: -0.55, tag: 'sad',    words: ['sad', 'cry', 'tear', 'blue', 'alone', 'miss', 'lonely', '悲', '伤', '泪', '孤独', '别'] },
    { w: 0.45, tag: 'angry',   words: ['angry', 'rage', 'fight', 'burn', '怒', '恨'] },
    { w: -0.35, tag: 'fearful', words: ['fear', 'dark', 'ghost', 'scare', '恐', '暗'] },
    { w: 0.35, tag: 'surprised', words: ['wow', 'magic', 'dream', 'fire', '惊', '梦'] },
    { w: 0.40, tag: 'healing', words: ['heal', 'light', 'hope', 'home', 'warm', 'love', '光', '暖', '家', '爱', '晴天'] },
    { w: -0.25, tag: 'calm',   words: ['calm', 'quiet', 'sleep', 'rain', 'slow', '静', '雨', '夜', '慢'] }
  ];

  /** 在线曲目情绪推断：曲风先验 + 标题/艺人关键词微调 */
  function inferTrackMood(track) {
    const prior = GENRE_PRIOR[track.genre] || DEFAULT_PRIOR;
    let v = prior.v, a = prior.a, e = prior.e;
    const text = `${track.title || ''} ${track.album || ''} ${(track.tags || []).join(' ')}`.toLowerCase();
    const hitTags = [];
    for (const group of MOOD_KEYWORDS) {
      for (const w of group.words) {
        if (w && text.indexOf(w.toLowerCase()) >= 0) {
          v += group.w * 0.35;
          hitTags.push(group.tag);
          break;
        }
      }
    }
    return {
      valence: clamp(v, -1, 1),
      arousal: clamp(a, -1, 1),
      energy: clamp(e, 0, 1),
      bpm: track.bpm || Math.round(62 + (clamp(a, -1, 1) + 1) / 2 * 78),
      tags: Array.from(new Set([...(track.tags || []), ...hitTags])),
      estimated: true
    };
  }

  /** 目标情绪坐标 */
  function planTarget(profile, strategy) {
    const v0 = profile.valence, a0 = profile.arousal;
    const dom = profile.dominant || 'neutral';
    let strategyUsed = strategy;
    if (strategy === 'auto') strategyUsed = v0 < -0.15 ? 'repair' : 'mirror';

    if (strategyUsed === 'repair') {
      const anchor = REPAIR_ANCHOR[dom] || REPAIR_ANCHOR.neutral;
      const k = 0.62;
      const v = v0 + k * (anchor.v - v0);
      const a = a0 + k * (anchor.a - a0);
      return {
        strategy: 'repair', v: +clamp(v, -1, 1).toFixed(3), a: +clamp(a, -1, 1).toFixed(3),
        from: { v: +v0.toFixed(3), a: +a0.toFixed(3) }, label: `情绪修复 · ${anchor.tag}`
      };
    }
    return {
      strategy: 'mirror', v: +v0.toFixed(3), a: +a0.toFixed(3),
      from: { v: +v0.toFixed(3), a: +a0.toFixed(3) }, label: '情绪镜像 · 与你同频'
    };
  }

  const W = { va: 0.40, emo: 0.22, rhythm: 0.18, pref: 0.11, quality: 0.09 };

  function vaScore(track, target) {
    const dv = track.valence - target.v, da = track.arousal - target.a;
    const d = Math.sqrt(dv * dv + da * da) / Math.SQRT2; // 归一化到 [0,1]
    return Math.exp(-(d * d) / (2 * 0.42 * 0.42));
  }

  function emoScore(track, profile) {
    // 情绪标签命中率（Jaccard 加权）；无标签时退化为效价符号一致性
    const tags = track.tags || [];
    const want = [profile.dominant];
    if (profile.dominant === 'sad') want.push('healing');
    if (profile.dominant === 'angry' || profile.dominant === 'fearful') want.push('calm');
    if (profile.dominant === 'happy') want.push('healing');
    const hit = want.filter(t => tags.indexOf(t) >= 0).length;
    const tagPart = hit / Math.max(1, want.length);
    const signPart = (Math.sign(track.valence) === Math.sign(profile.valence) || Math.abs(profile.valence) < 0.1) ? 1 : 0.35;
    return 0.6 * tagPart + 0.4 * signPart;
  }

  function rhythmScore(track, target) {
    const targetBpm = 60 + (target.a + 1) / 2 * 82;  // 60~142 BPM
    const bpmPart = Math.exp(-Math.pow((track.bpm - targetBpm) / 30, 2));
    const energyPart = 1 - Math.abs(clamp(track.energy, 0, 1) - (target.a + 1) / 2);
    return 0.55 * bpmPart + 0.45 * energyPart;
  }

  function prefScore(track, ctx) {
    const prefs = ctx.preferences || {};
    let s = 0.5;
    if (prefs.genre && track.genre === prefs.genre) s += 0.2;
    if (prefs.lang && track.lang === prefs.lang) s += 0.15;
    if ((prefs.liked || []).indexOf(track.id) >= 0) s += 0.25;
    if ((ctx.playedIds || []).indexOf(track.id) >= 0) s -= 0.45; // 新颖度惩罚
    if ((ctx.dislikedIds || []).indexOf(track.id) >= 0) s -= 0.8;
    return clamp(s, 0, 1);
  }

  function qualityScore(track) {
    let s = track.source === 'local' ? 1 : 0.86;
    if (!track.url) s *= 0.2;
    if (track.duration && track.duration < 40) s *= 0.92;
    return s;
  }

  /* ---------- 主推荐函数 ---------- */
  function recommend(tracks, profile, ctx) {
    ctx = ctx || {};
    const target = planTarget(profile, ctx.strategy || 'auto');
    const scored = tracks.map(t => {
      const sVa = vaScore(t, target);
      const sEmo = emoScore(t, profile);
      const sRhy = rhythmScore(t, target);
      const sPref = prefScore(t, ctx);
      const sQ = qualityScore(t);
      const total = W.va * sVa + W.emo * sEmo + W.rhythm * sRhy + W.pref * sPref + W.quality * sQ;
      return { track: t, total, parts: { va: sVa, emo: sEmo, rhythm: sRhy, pref: sPref, quality: sQ }, target };
    });

    scored.sort((a, b) => b.total - a.total);

    // ---- MMR 多样性重排 ----
    const N = ctx.limit || 8;
    const lambda = ctx.lambda != null ? ctx.lambda : 0.75;
    const maxS = scored.length ? scored[0].total || 1 : 1;
    const pool = scored.slice();
    const picked = [];
    const artistCount = {};
    while (picked.length < N && pool.length) {
      let bestIdx = 0, bestVal = -Infinity;
      for (let i = 0; i < pool.length; i++) {
        const c = pool[i];
        let sim = 0;
        for (const p of picked) {
          const sameArtist = c.track.artist && c.track.artist === p.track.artist ? 0.55 : 0;
          const sameGenre = c.track.genre && c.track.genre === p.track.genre ? 0.42 : 0;
          const moodSim = 0.15 * (1 - Math.abs(c.track.valence - p.track.valence) / 2);
          sim = Math.max(sim, sameArtist + sameGenre + moodSim);
        }
        // 同一艺人渐进惩罚：前 2 首不受影响，之后逐步降权，避免刷屏但不搞一刀切
        const ac = artistCount[c.track.artist] || 0;
        const val = lambda * (c.total / maxS) - (1 - lambda) * sim - 0.26 * Math.max(0, ac - 1);
        if (val > bestVal) { bestVal = val; bestIdx = i; }
      }
      const chosen = pool.splice(bestIdx, 1)[0];
      artistCount[chosen.track.artist] = (artistCount[chosen.track.artist] || 0) + 1;
      picked.push(chosen);
    }

    return {
      target,
      list: picked.map((p, i) => ({
        ...p.track,
        rank: i + 1,
        score: +(p.total * 100).toFixed(1),
        parts: {
          va: +(p.parts.va * 100).toFixed(0),
          emo: +(p.parts.emo * 100).toFixed(0),
          rhythm: +(p.parts.rhythm * 100).toFixed(0),
          pref: +(p.parts.pref * 100).toFixed(0),
          quality: +(p.parts.quality * 100).toFixed(0)
        },
        reason: buildReason(p, target, profile)
      })),
      candidates: scored.length,
      weights: W
    };
  }

  function buildReason(item, target, profile) {
    const t = item.track;
    const dv = (t.valence - target.v).toFixed(2);
    const da = (t.arousal - target.a).toFixed(2);
    const bits = [];
    if (item.parts.va > 0.72) bits.push(`情感坐标高度贴合（Δv=${dv}, Δa=${da}）`);
    else if (item.parts.va > 0.5) bits.push(`情感坐标较接近（Δv=${dv}, Δa=${da}）`);
    if (item.parts.emo >= 0.8) bits.push(`标签命中「${(t.tags || []).slice(0, 3).join('/') || profile.dominantCn}」`);
    if (item.parts.rhythm >= 0.75) bits.push(`${t.bpm || '—'} BPM 匹配当前唤醒需求`);
    if (!bits.length) bits.push('综合得分进入候选前列');
    return bits.join('；');
  }

  /** 在线检索关键词：按目标情绪 + 修复/镜像策略生成 2~3 组查询 */
  function buildQueries(profile, target, prefs) {
    const dom = profile.dominant || 'neutral';
    const base = MOOD_QUERY[dom] || MOOD_QUERY.neutral;
    const list = [];
    if (target.strategy === 'repair') {
      const uplift = ['治愈 温暖', 'healing uplifting', '轻快 明亮'];
      list.push(...uplift.slice(0, 2));
    }
    list.push(...base.cn.slice(0, 2));
    list.push(...base.en.slice(0, 2));
    if (prefs && prefs.artist) list.unshift(prefs.artist);
    return Array.from(new Set(list)).slice(0, 3);
  }

  root.Recommender = {
    recommend, planTarget, inferTrackMood, buildQueries,
    REPAIR_ANCHOR, MOOD_QUERY, WEIGHTS: W
  };
})(typeof window !== 'undefined' ? window : globalThis);
