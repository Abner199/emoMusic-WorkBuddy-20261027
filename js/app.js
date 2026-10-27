/**
 * app.js —— Vue3 应用主逻辑
 * 采集（摄像头 / 图片 / 视频 / 拍照） → 情绪识别 → 目标决策 → 推荐 → 播放
 */
(function () {
  'use strict';

  const { createApp } = Vue;

  const EMOTION_ORDER = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'];

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  createApp({
    data() {
      return {
        /* 模型 */
        modelReady: false,
        modelProgress: '初始化…',
        modelError: false,
        backend: '-',

        /* 采集 */
        modes: [
          { k: 'camera', label: '实时摄像头' },
          { k: 'image', label: '上传图片' },
          { k: 'video', label: '打开视频' }
        ],
        mode: 'camera',
        cameraOn: false,
        imageSrc: '',
        videoSrc: '',
        videoPaused: false,
        dragging: false,
        showLandmarks: true,
        smoothing: true,
        demoFaces: [
          { file: 'face-happy.jpg', label: '高兴' },
          { file: 'face-sad.jpg', label: '悲伤' },
          { file: 'face-angry.jpg', label: '愤怒' },
          { file: 'face-surprised.jpg', label: '惊讶' },
          { file: 'face-fearful.jpg', label: '恐惧' },
          { file: 'face-disgusted.jpg', label: '厌恶' },
          { file: 'face-neutral.jpg', label: '平静' }
        ],

        /* 情绪 */
        result: null,
        fps: 0,
        history: [],

        /* 推荐 */
        strategy: 'auto',
        strategies: [
          { k: 'auto', label: '智能' },
          { k: 'mirror', label: '同频镜像' },
          { k: 'repair', label: '情绪修复' }
        ],
        onlineEnabled: true,
        loadingRec: false,
        recommendations: [],
        recInfo: null,
        prefs: { lang: '', genre: '', artist: '' },

        /* 曲库 */
        localLibrary: [],
        localReady: false,
        libProgress: '待分析',
        importing: false,

        /* 播放 */
        volume: 0.8,
        autoPlay: true,          // 识别出情绪 / 生成歌单后自动起播
        playBlocked: false,      // 被浏览器自动播放策略拦截
        player: { id: '', track: null, playing: false, currentTime: 0, duration: 0, auto: false },
        playedIds: [],

        /* 其他 */
        showAlgo: false,
        toast: ''
      };
    },

    computed: {
      stageActive() {
        if (this.mode === 'camera') return this.cameraOn;
        if (this.mode === 'image') return !!this.imageSrc;
        return !!this.videoSrc;
      },
      emptyHint() {
        if (this.mode === 'camera') return '点击「开启摄像头」，正对镜头保持光线充足；也可以直接把图片 / 视频拖到这个区域。';
        if (this.mode === 'image') return '支持 JPG / PNG / WebP，建议人脸清晰、单一主体。';
        return '支持 MP4 / WebM / MOV，系统会边播放边逐帧识别。';
      },
      emoBars() {
        if (!this.result) return [];
        const f = this.result.fused;
        return EMOTION_ORDER
          .map(k => ({
            key: k,
            cn: EMO_MAP[k].cn,
            color: EMO_MAP[k].color,
            pct: Math.round((f[k] || 0) * 100)
          }))
          .sort((a, b) => b.pct - a.pct);
      },
      strategyName() {
        const s = this.strategies.find(x => x.k === this.strategy);
        return s ? s.label : '智能';
      },
      strategyDesc() {
        if (this.strategy === 'mirror') return '同频镜像：让音乐情感坐标尽量贴近你此刻的情绪，适合共鸣式聆听。';
        if (this.strategy === 'repair') return '情绪修复：以 0.62 的步长把目标坐标推向该情绪的修复锚点，用于消极情绪的疏导。';
        return '智能：愉悦度低于 −0.15 时自动切换到「情绪修复」，否则使用「同频镜像」。';
      }
    },

    async mounted() {
      this.engine = new EmotionEngine({ modelPath: 'assets/models' });
      this.engine.smoothing = this.smoothing;

      // 自动播放偏好（上次选择会被记住）
      try {
        const saved = localStorage.getItem('emosong.autoPlay');
        if (saved !== null) this.autoPlay = saved === '1';
        const vol = localStorage.getItem('emosong.volume');
        if (vol !== null) this.volume = Math.min(1, Math.max(0, Number(vol) || 0.8));
      } catch (e) { /* 隐私模式忽略 */ }

      this.loadModels();
      this.loadLibrary();

      window.addEventListener('resize', this.onResize);
      window.addEventListener('beforeunload', this.cleanup);
      this.renderVA();
    },

    beforeUnmount() {
      this.cleanup();
      window.removeEventListener('resize', this.onResize);
    },

    methods: {
      /* ==================== 初始化 ==================== */
      async loadModels() {
        try {
          this.modelProgress = '加载 TinyFaceDetector / FaceExpressionNet / FaceLandmark68…';
          await this.engine.load();
          this.backend = this.engine.backend;
          this.modelReady = true;
          this.modelProgress = '就绪';
        } catch (e) {
          console.error(e);
          this.modelError = true;
          this.modelProgress = '模型加载失败：' + e.message;
          this.showToast('模型加载失败，请确认 assets/models 目录完整');
        }
      },

      async loadLibrary() {
        try {
          this.localLibrary = await MusicLibrary.analyzeLocalLibrary((i, total, label) => {
            this.libProgress = `${i}/${total} · ${label}`;
          });
          this.localReady = true;
          this.libProgress = `${this.localLibrary.length} 首已分析`;
          document.documentElement.dataset.libReady = '1';
        } catch (e) {
          console.error(e);
          this.libProgress = '本地曲库分析失败';
          document.documentElement.dataset.libReady = '0';
        }
      },

      async onImportAudio(e) {
        const files = Array.from(e.target.files || []);
        e.target.value = '';
        if (!files.length) return;
        this.importing = true;
        try {
          const added = await MusicLibrary.importLocalFiles(files, this.localLibrary.length);
          this.localLibrary = this.localLibrary.concat(added);
          this.showToast(added.length ? `已导入 ${added.length} 首并完成声学分析` : '未识别到可用的音频文件');
          if (added.length && this.result) this.recommend(true);
        } catch (err) {
          console.error(err);
          this.showToast('导入失败：' + err.message);
        } finally {
          this.importing = false;
        }
      },

      /* ==================== 采集源切换 ==================== */
      setMode(k) {
        if (k === this.mode) return;
        this.stopLoop();
        if (this.mode === 'camera' && k !== 'camera') this.stopCamera(true);
        this.mode = k;
        this.result = null;
        this.history = [];
        this.engine.resetSmoothing();
        this._manualPauseAt = 0;   // 新的一次识别 → 解除“刚手动暂停”保护
        this.clearOverlay();
        this.renderVA();

        if (k === 'video' && this.videoSrc) {
          this.$nextTick(() => this.startVideoLoop());
        }
        if (k === 'image' && this.imageSrc) {
          this.$nextTick(() => this.analyzeImage());
        }
      },

      async startCamera() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false
          });
          this.stream = stream;
          const v = this.$refs.video;
          v.srcObject = stream;
          await v.play();
          this.cameraOn = true;
          this.result = null;
          this.history = [];
          this.engine.resetSmoothing();
          this._manualPauseAt = 0;   // 重新开启摄像头 → 解除“刚手动暂停”保护
          this.startLoop(v, 224);
        } catch (e) {
          console.error(e);
          this.showToast('无法访问摄像头：' + (e.message || e.name) + '（需 https 或 localhost）');
        }
      },

      stopCamera(silent) {
        if (this.stream) {
          this.stream.getTracks().forEach(t => t.stop());
          this.stream = null;
        }
        this.cameraOn = false;
        if (!silent) {
          this.stopLoop();
          this.result = null;
          this.clearOverlay();
          this.renderVA();
        }
      },

      async shoot() {
        const v = this.$refs.video;
        if (!v || !this.cameraOn) return;
        const c = document.createElement('canvas');
        c.width = v.videoWidth || 1280;
        c.height = v.videoHeight || 720;
        c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
        const data = c.toDataURL('image/jpeg', 0.92);

        this.stopLoop();
        this.stopCamera(true);
        this.mode = 'image';
        this.imageSrc = data;
        this.result = null;
        this.clearOverlay();
        await this.$nextTick();
        await this.analyzeImage();
        this.showToast('已拍摄并完成识别，可切换回摄像头继续实时分析');
      },

      stopLoop() {
        this._stopped = true;
        if (this._raf) cancelAnimationFrame(this._raf);
        this._raf = null;
      },

      startLoop(source, inputSize) {
        this.stopLoop();
        this._stopped = false;
        this._src = source;
        this._size = inputSize || 224;
        this._busy = false;
        this._lastAnalyze = 0;
        this._fpsWindow = performance.now();
        this._fpsCount = 0;

        const tick = async () => {
          if (this._stopped) return;
          const now = performance.now();
          if (!this._busy && now - this._lastAnalyze > 165) {
            this._lastAnalyze = now;
            this._busy = true;
            try {
              await this.analyzeFrame(source, this._size);
            } catch (e) {
              console.warn(e);
            } finally {
              this._busy = false;
            }
          }
          this._raf = requestAnimationFrame(tick);
        };
        this._raf = requestAnimationFrame(tick);
      },

      async analyzeFrame(source, inputSize) {
        if (!this.modelReady) return;
        const res = await this.engine.analyze(source, { inputSize });
        this.applyResult(res);
        this._fpsCount++;
        const now = performance.now();
        if (now - this._fpsWindow >= 1000) {
          this.fps = Math.round((this._fpsCount * 1000) / (now - this._fpsWindow));
          this._fpsWindow = now;
          this._fpsCount = 0;
        }
      },

      applyResult(res, opts) {
        this.result = res;
        if (res) {
          this.history.push({ v: res.valence, a: res.arousal, t: Date.now() });
          if (this.history.length > 90) this.history.shift();
          this.maybeAutoRecommend(res, opts);
        }
        this.drawOverlay();
        this.renderVA();
      },

      /* ==================== 图片 / 视频 ==================== */
      onImageFile(e) {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        this.loadImageFile(f);
        e.target.value = '';
      },

      async loadImageFile(file) {
        if (!/^image\//.test(file.type)) return this.showToast('请选择图片文件');
        this.stopLoop();
        this.mode = 'image';
        this.imageSrc = URL.createObjectURL(file);
        this.result = null;
        this.history = [];
        this.engine.resetSmoothing();
        this._manualPauseAt = 0;   // 新的一次识别 → 解除“刚手动暂停”保护
        await this.$nextTick();
        await this.analyzeImage();
      },

      async analyzeImage() {
        const img = this.$refs.img;
        if (!img || !this.imageSrc || !this.modelReady) return;
        await this.waitImage(img);
        try {
          // 静态图走多尺度阶梯：特写大脸 / 远景小脸都能检出
          const res = await this.engine.analyze(img, { multiScale: true, sizes: [416, 608, 320, 224] });
          // 静态图只分析一帧，给一次「立即重算歌单」的机会，不等 6 秒防抖
          this.applyResult(res, { force: true });
          if (!res) this.showToast('未检测到人脸，试试换一张正脸清晰的照片');
        } catch (e) {
          console.error(e);
          this.showToast('识别失败：' + e.message);
        }
      },

      async loadDemoImage(name) {
        this.stopLoop();
        if (this.cameraOn) this.stopCamera(true);
        this.mode = 'image';
        this.imageSrc = 'assets/demo/' + name;
        this.result = null;
        this.history = [];
        this.engine.resetSmoothing();
        this._manualPauseAt = 0;   // 新的一次识别 → 解除“刚手动暂停”保护
        await this.$nextTick();
        await this.analyzeImage();
      },

      waitImage(img) {
        return new Promise(resolve => {
          if (img.complete && img.naturalWidth) return resolve();
          img.onload = () => resolve();
          img.onerror = () => resolve();
        });
      },

      onVideoFile(e) {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        this.loadVideoFile(f);
        e.target.value = '';
      },

      async loadVideoFile(file) {
        if (!/^video\//.test(file.type)) return this.showToast('请选择视频文件');
        this.stopLoop();
        if (this.cameraOn) this.stopCamera(true);
        this.mode = 'video';
        this.videoSrc = URL.createObjectURL(file);
        this.result = null;
        this.history = [];
        this.engine.resetSmoothing();
        this._manualPauseAt = 0;   // 新的一次识别 → 解除“刚手动暂停”保护
        await this.$nextTick();
        this.startVideoLoop();
      },

      async startVideoLoop() {
        const v = this.$refs.vid;
        if (!v) return;
        try {
          v.currentTime = 0;
          await v.play();
          this.videoPaused = false;
        } catch (e) { /* 自动播放可能被拦截 */ }
        this.startLoop(v, 320);
      },

      toggleVideoPlay() {
        const v = this.$refs.vid;
        if (!v) return;
        if (v.paused) { v.play(); this.videoPaused = false; this.startLoop(v, 320); }
        else { v.pause(); this.videoPaused = true; this.stopLoop(); }
      },

      onDrop(e) {
        this.dragging = false;
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!f) return;
        if (/^image\//.test(f.type)) this.loadImageFile(f);
        else if (/^video\//.test(f.type)) this.loadVideoFile(f);
        else this.showToast('仅支持图片或视频文件');
      },

      /* ==================== 绘制 ==================== */
      clearOverlay() {
        const cv = this.$refs.overlay;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, cv.width, cv.height);
      },

      drawOverlay() {
        const stage = this.$refs.stage;
        const cv = this.$refs.overlay;
        if (!stage || !cv) return;
        const w = stage.clientWidth, h = stage.clientHeight;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
          cv.width = Math.round(w * dpr);
          cv.height = Math.round(h * dpr);
        }
        const ctx = cv.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        EmotionViz.drawOverlay(ctx, this.result, w, h, { landmarks: this.showLandmarks });
      },

      renderVA() {
        const cv = this.$refs.va;
        if (!cv) return;
        const w = cv.clientWidth || 360, h = cv.clientHeight || 196;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
          cv.width = Math.round(w * dpr);
          cv.height = Math.round(h * dpr);
        }
        const ctx = cv.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const pad = 20;
        const iw = w - pad * 2, ih = h - pad * 2;
        const X = v => pad + ((v + 1) / 2) * iw;
        const Y = a => pad + ((1 - a) / 2) * ih;

        // 象限底色
        const quads = [
          { x: pad, y: pad, w: iw / 2, h: ih / 2, c: 'rgba(255,214,102,.10)', label: '紧张 · 高唤醒' },
          { x: pad + iw / 2, y: pad, w: iw / 2, h: ih / 2, c: 'rgba(255,138,180,.11)', label: '愉悦 · 高唤醒' },
          { x: pad, y: pad + ih / 2, w: iw / 2, h: ih / 2, c: 'rgba(120,150,255,.10)', label: '低沉 · 低唤醒' },
          { x: pad + iw / 2, y: pad + ih / 2, w: iw / 2, h: ih / 2, c: 'rgba(140,220,180,.12)', label: '安详 · 低唤醒' }
        ];
        ctx.font = '10px -apple-system,"PingFang SC",sans-serif';
        for (const q of quads) {
          ctx.fillStyle = q.c;
          ctx.fillRect(q.x, q.y, q.w, q.h);
          ctx.fillStyle = 'rgba(120,128,150,.55)';
          ctx.textAlign = q.x < pad + iw / 2 ? 'left' : 'right';
          ctx.textBaseline = q.y < pad + ih / 2 ? 'top' : 'bottom';
          ctx.fillText(q.label, q.x + (q.x < pad + iw / 2 ? 8 : q.w - 8), q.y + (q.y < pad + ih / 2 ? 7 : q.h - 7));
        }

        // 坐标轴
        ctx.strokeStyle = 'rgba(140,148,170,.28)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(0), pad); ctx.lineTo(X(0), pad + ih);
        ctx.moveTo(pad, Y(0)); ctx.lineTo(pad + iw, Y(0));
        ctx.stroke();

        ctx.fillStyle = 'rgba(120,128,150,.7)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('愉悦度 →', pad + iw / 2, pad + ih - 13);
        ctx.save();
        ctx.translate(11, pad + ih / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('唤醒度 →', 0, 0);
        ctx.restore();

        // 七类情绪锚点（标签交替上下，避免相邻情绪文字重叠）
        EMOTION_ORDER.forEach((k, i) => {
          const e = EMO_MAP[k];
          ctx.beginPath();
          ctx.arc(X(e.valence), Y(e.arousal), 3, 0, Math.PI * 2);
          ctx.fillStyle = e.color + '99';
          ctx.fill();
          ctx.fillStyle = 'rgba(110,118,140,.75)';
          ctx.font = '10px -apple-system,"PingFang SC",sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = i % 2 === 0 ? 'bottom' : 'top';
          const dy = i % 2 === 0 ? -5 : 7;
          ctx.fillText(e.cn, X(e.valence), Y(e.arousal) + dy);
        });

        if (!this.result) return;

        // 历史轨迹
        if (this.history.length > 1) {
          ctx.lineWidth = 1.6;
          for (let i = 1; i < this.history.length; i++) {
            const alpha = 0.05 + 0.5 * (i / this.history.length);
            ctx.strokeStyle = `rgba(107,91,250,${alpha})`;
            ctx.beginPath();
            ctx.moveTo(X(this.history[i - 1].v), Y(this.history[i - 1].a));
            ctx.lineTo(X(this.history[i].v), Y(this.history[i].a));
            ctx.stroke();
          }
        }

        // 目标点（策略决策结果）
        const target = Recommender.planTarget(
          { valence: this.result.valence, arousal: this.result.arousal, dominant: this.result.dominant },
          this.strategy
        );
        ctx.strokeStyle = '#f2609e';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(X(target.v), Y(target.a), 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // 当前点
        const cx = X(this.result.valence), cy = Y(this.result.arousal);
        const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 18);
        glow.addColorStop(0, this.result.dominantColor + '66');
        glow.addColorStop(1, this.result.dominantColor + '00');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = this.result.dominantColor;
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#15171f';
        ctx.font = '600 11px -apple-system,"PingFang SC",sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const label = `${this.result.dominantCn} (${this.result.valence.toFixed(2)}, ${this.result.arousal.toFixed(2)})`;
        const tx = clamp(cx + 11, pad, w - 130);
        ctx.fillText(label, tx, cy - 13);
      },

      onResize() {
        this.drawOverlay();
        this.renderVA();
      },

      /* ==================== 推荐 ==================== */
      currentProfile() {
        if (!this.result) return null;
        return {
          valence: this.result.valence,
          arousal: this.result.arousal,
          dominant: this.result.dominant,
          dominantCn: this.result.dominantCn
        };
      },

      async getOnlineCandidates(profile, target) {
        if (!this.onlineEnabled) return { list: [], queries: [] };
        const queries = Recommender.buildQueries(profile, target, this.prefs);
        const key = queries.join('|');
        if (this._onlineCache && this._onlineCache[key] && Date.now() - this._onlineCache[key].t < 600000) {
          return { list: this._onlineCache[key].list, queries };
        }
        let list = [];
        try {
          list = await MusicLibrary.searchOnline(queries, { limit: 12 });
        } catch (e) {
          console.warn('在线检索失败', e);
          this.showToast('在线曲库暂时不可用，已仅使用本地曲目');
        }
        this._onlineCache = this._onlineCache || {};
        this._onlineCache[key] = { t: Date.now(), list };
        return { list, queries };
      },

      async recommend(manual) {
        const profile = this.currentProfile();
        if (!profile) {
          if (manual) this.showToast('请先完成一次人脸情绪识别');
          return;
        }
        this.loadingRec = true;
        try {
          const target = Recommender.planTarget(profile, this.strategy);
          const online = await this.getOnlineCandidates(profile, target);

          let candidates = this.localLibrary.slice();
          const onlineTracks = online.list.map(t => MusicLibrary.enrichOnline(t));
          candidates = candidates.concat(onlineTracks);

          if (this.prefs.lang) {
            const filtered = candidates.filter(t => t.lang === this.prefs.lang);
            if (filtered.length >= 4) candidates = filtered;
            else if (manual) this.showToast(`「${this.prefs.lang}」曲目不足，已放宽语种限制`);
          }

          if (!candidates.length) {
            this.showToast('曲库为空：请等待本地音频分析完成，或打开在线试听');
            return;
          }

          const res = Recommender.recommend(candidates, profile, {
            strategy: this.strategy,
            preferences: this.prefs,
            playedIds: this.playedIds,
            limit: 8
          });

          this.recommendations = res.list;
          this.recInfo = {
            target: res.target,
            candidates: res.candidates,
            weights: res.weights,
            sourceLabel: onlineTracks.length
              ? `本地 ${this.localLibrary.length} + 在线 ${onlineTracks.length}`
              : `仅本地 ${this.localLibrary.length} 首`
          };
          this._lastRecAt = Date.now();
          this._lastRecEmotion = profile.dominant;

          // 识别 → 推荐完成后自动起播（清单第一首）
          this.tryAutoPlay(res.list, profile, manual);
        } catch (e) {
          console.error(e);
          this.showToast('推荐失败：' + e.message);
        } finally {
          this.loadingRec = false;
          // 上一次推荐还在路上时情绪又变了 → 补算一次，避免歌单“卡”在旧情绪
          if (this._recQueued) {
            this._recQueued = false;
            if (this.result) setTimeout(() => this.recommend(false), 30);
          }
        }
      },

      /**
       * 自动起播：歌单一生成就播第一首可播放曲目
       * 跳过条件：关闭了自动播放 / 用户刚手动暂停 / 正在播的就是这首
       */
      tryAutoPlay(list, profile, manual) {
        if (!this.autoPlay) {
          if (manual) this.showToast('歌单已生成（自动播放已关闭）');
          return;
        }
        if (!list || !list.length) return;
        if (this._manualPauseAt && Date.now() - this._manualPauseAt < 30000) return;
        const track = list.find(t => t.url);
        if (!track) return;
        if (this.player.id === track.id && this.player.playing) return;
        this.play(track, { auto: true });
        const who = profile && profile.dominantCn ? `「${profile.dominantCn}」` : '';
        this.showToast(`${who}已自动播放：${track.title}`);
      },

      maybeAutoRecommend(res, opts) {
        const force = !!(opts && opts.force);
        const now = Date.now();
        const changed = res.dominant !== this._lastRecEmotion;
        const stable = res.margin > 0.12 && res.confidence > 0.42;
        if (this.loadingRec) {
          if (changed && stable) this._recQueued = true;
          return;
        }
        if (!this.recommendations.length) {
          if (stable && this.localReady && (!this._bootRecAt || now - this._bootRecAt > 4000)) {
            this._bootRecAt = now;
            this.recommend(false);
          }
          return;
        }
        if (changed && stable && (force || now - (this._lastRecAt || 0) > 6000)) {
          this.recommend(false);
        }
      },

      onStrategyChange() {
        this.renderVA();
        if (this.result && this.recommendations.length) this.recommend(true);
      },

      /* ==================== 播放 ==================== */
      play(t, opts) {
        opts = opts || {};
        if (!t || !t.url) return;
        const audio = this.$refs.audio;
        if (this.player.id === t.id) {
          // 自动播放时同一首不重新开始，避免打断
          if (opts.auto) {
            if (audio.paused) this.startPlayback();
          } else {
            this.togglePlay();
          }
          return;
        }
        this.player = {
          id: t.id, track: t, playing: false, currentTime: 0,
          duration: t.duration || 0, auto: !!opts.auto
        };
        audio.src = t.url;
        audio.volume = this.volume;
        audio.currentTime = 0;
        this.startPlayback();
        if (this.playedIds.indexOf(t.id) < 0) this.playedIds.push(t.id);
      },

      startPlayback() {
        const audio = this.$refs.audio;
        if (!audio) return;
        const p = audio.play();
        if (!p || !p.then) return;
        p.then(() => { this.playBlocked = false; }).catch(err => {
          if (err && err.name === 'NotAllowedError') {
            // 浏览器自动播放策略拦截：给出一次点击入口
            this.playBlocked = true;
          } else if (err && err.name !== 'AbortError') {
            this.showToast('播放失败：' + err.message);
          }
        });
      },

      resumePlay() {
        this.playBlocked = false;
        this._manualPauseAt = 0;
        this.startPlayback();
      },

      togglePlay() {
        const audio = this.$refs.audio;
        if (!this.player.id || !audio.src) return;
        if (audio.paused) {
          this._manualPauseAt = 0;
          this.startPlayback();
        } else {
          this._manualPauseAt = Date.now();  // 手动暂停后 30s 内不再自动打断
          audio.pause();
        }
      },

      nextTrack() {
        const list = this.recommendations;
        if (!list.length) return;
        const idx = list.findIndex(t => t.id === this.player.id);
        const next = list[(idx + 1) % list.length];
        this.play(next);
      },

      onTimeUpdate(e) {
        this.player.currentTime = e.target.currentTime || 0;
        this.player.duration = e.target.duration || this.player.duration || 0;
      },

      onLoaded(e) {
        this.player.duration = e.target.duration || this.player.duration || 0;
      },

      onEnded() {
        this.nextTrack();
      },

      seek(val) {
        const audio = this.$refs.audio;
        if (audio) audio.currentTime = Number(val) || 0;
      },

      applyVolume() {
        const audio = this.$refs.audio;
        if (audio) audio.volume = this.volume;
      },

      fmt(sec) {
        if (!sec || isNaN(sec)) return '0:00';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
      },

      /* ==================== 工具 ==================== */
      coverStyle(t) {
        if (!t) return 'linear-gradient(135deg,#6b5bfa,#f2609e)';
        return MusicLibrary.coverGradient(t);
      },

      scoreTitle(t) {
        if (!t.parts) return '';
        return `情感距离 ${t.parts.va} · 情绪标签 ${t.parts.emo} · 节奏能量 ${t.parts.rhythm} · 偏好新颖 ${t.parts.pref} · 音源质量 ${t.parts.quality}`;
      },

      showToast(msg) {
        this.toast = msg;
        clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => { this.toast = ''; }, 2600);
      },

      cleanup() {
        this.stopLoop();
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
      }
    },

    watch: {
      smoothing(val) {
        if (this.engine) this.engine.smoothing = val;
        if (!val && this.engine) this.engine.resetSmoothing();
      },
      showLandmarks() {
        this.drawOverlay();
      },
      autoPlay(val) {
        try { localStorage.setItem('emosong.autoPlay', val ? '1' : '0'); } catch (e) { /* ignore */ }
        if (!val) this.playBlocked = false;
        else if (this.result && !this.recommendations.length && this.localReady) this.recommend(false);
      },
      volume(val) {
        try { localStorage.setItem('emosong.volume', String(val)); } catch (e) { /* ignore */ }
      }
    }
  }).mount('#app');
})();
