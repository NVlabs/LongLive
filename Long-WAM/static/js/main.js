/* Long-WAM project page — no external dependencies. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const ease = x => 1 - Math.pow(1 - clamp(x), 3);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const D = window.LW, CFG = window.LONGWAM_CONFIG || { links: {}, authors: {} };
  const NS = 'http://www.w3.org/2000/svg';
  const svgText = html => html.replace(/<sub>(.*?)<\/sub>/g, '<tspan baseline-shift="sub" font-size="75%">$1</tspan>');
  const svgEl = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const cssVar = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const COL = { robot: cssVar('--robot'), ar: cssVar('--ar'), bi: cssVar('--bi'), pi: cssVar('--pi'), fast: cssVar('--fast'), ours: cssVar('--ours'), human: cssVar('--human'), bg: cssVar('--bg') };
  const onView = (node, fn, threshold = .3) => { const o = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { o.disconnect(); fn(); } }, { threshold }); o.observe(node); };
  const animate = (dur, fn) => new Promise(res => { if (reduced) { fn(1); return res(); } const t0 = performance.now(); const step = now => { const p = clamp((now - t0) / dur); fn(ease(p)); p < 1 ? requestAnimationFrame(step) : res(); }; requestAnimationFrame(step); });

  /* ─────────── 0 · Film ─────────── */
  const film = $('#film'), frame = $('#filmFrame'), video = $('#filmVideo'), cta = $('#filmCta'), ctaText = $('#filmCtaText');
  const muteBtn = $('#filmMute'), bar = $('#filmBar'), tLabel = $('#filmTime'), controls = $('.film-controls'), skip = $('.film-skip'), nav = $('#nav');
  let soundChosen = false, wantPlaying = true, autoPaused = false, userScrolled = false;
  const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const setMuted = m => { video.muted = m; muteBtn.classList.toggle('is-muted', m); muteBtn.setAttribute('aria-label', m ? 'Turn sound on' : 'Mute'); };
  const tryPlay = () => { const p = video.play(); if (p && p.catch) p.catch(() => { ctaText.textContent = 'Play the film'; document.body.classList.add('cta-on'); }); };
  setMuted(true);
  function intro() {
    const done = () => { document.body.classList.remove('is-intro'); document.body.classList.add('intro-done'); };
    if (reduced) { done(); document.body.classList.add('cta-on'); tryPlay(); return; }
    setTimeout(tryPlay, 950);                 // motion starts while the aperture opens
    setTimeout(done, 2150);
    setTimeout(() => { if (!soundChosen) document.body.classList.add('cta-on'); }, 2400);
  }
  cta.addEventListener('click', () => {
    soundChosen = true; wantPlaying = true; document.body.classList.remove('cta-on'); film.classList.remove('is-ended');
    video.currentTime = 0; setMuted(false); video.play().catch(() => { setMuted(true); video.play().catch(() => {}); });
  });
  muteBtn.addEventListener('click', () => { setMuted(!video.muted); if (!video.muted) { soundChosen = true; document.body.classList.remove('cta-on'); if (video.paused) video.play().catch(() => {}); } });
  $('#filmReplay').addEventListener('click', () => { film.classList.remove('is-ended'); wantPlaying = true; video.currentTime = 0; video.play().catch(() => {}); });
  $('#filmFull').addEventListener('click', () => {
    if (document.fullscreenElement) return document.exitFullscreen();
    if (frame.requestFullscreen) frame.requestFullscreen(); else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
  });
  video.addEventListener('timeupdate', () => { const d = video.duration || 158; bar.style.width = `${video.currentTime / d * 100}%`; if (tLabel) tLabel.textContent = `${mmss(video.currentTime)} / ${mmss(d)}`; });
  video.addEventListener('ended', () => {
    film.classList.add('is-ended'); wantPlaying = false; ctaText.textContent = 'Watch again with sound';
    if (!userScrolled && scrollY < 40) setTimeout(() => $('#paper').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth' }), 800);
  });
  addEventListener('wheel', () => { userScrolled = true; }, { once: true, passive: true });
  addEventListener('touchmove', () => { userScrolled = true; }, { once: true, passive: true });
  addEventListener('keydown', e => { if (['ArrowDown', 'PageDown', ' ', 'End'].includes(e.key)) userScrolled = true; });
  function onScroll() {
    const max = Math.max(1, film.offsetHeight - innerHeight), p = clamp(scrollY / max), on = document.body.classList.contains('intro-done');
    if (!reduced && on) { frame.style.transform = `translateY(${-3 * p}vh) scale(${1 - .3 * p})`; frame.style.borderRadius = `${28 * p}px`; frame.style.opacity = String(1 - .6 * p); }
    if (on) { controls.style.opacity = String(clamp(1 - p * 1.8)); skip.style.opacity = String(clamp(1 - p * 2.5)); }
    const away = scrollY > film.offsetHeight - innerHeight * .45;
    if (away && !video.paused) { video.pause(); autoPaused = true; }
    else if (!away && autoPaused && wantPlaying) { autoPaused = false; video.play().catch(() => {}); }
    nav.classList.toggle('is-on', scrollY > film.offsetHeight - 90);
  }
  addEventListener('scroll', onScroll, { passive: true }); addEventListener('resize', onScroll);
  intro(); onScroll();

  /* ─────────── nav: active section ─────────── */
  const navMap = { results: 'results', benchmarks: 'results', realworld: 'results', efficiency: 'efficiency', planning: 'planning', demos: 'demos', bibtex: 'bibtex' };
  const navLinks = $$('.nav-links a');
  const secIo = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) navLinks.forEach(a => a.classList.toggle('is-active', a.getAttribute('href') === '#' + navMap[e.target.id])); }), { rootMargin: '-45% 0px -50% 0px' });
  Object.keys(navMap).forEach(id => { const n = document.getElementById(id); if (n) secIo.observe(n); });

  /* ─────────── links, authors, BibTeX ─────────── */
  $$('[data-link]').forEach(a => {
    const url = (CFG.links || {})[a.dataset.link];
    if (url) { a.href = url; a.target = '_blank'; a.rel = 'noopener'; }
    else { a.classList.add('is-soon'); a.removeAttribute('href'); a.setAttribute('aria-disabled', 'true'); a.title = 'Coming soon'; }
  });
  $$('#authors .author').forEach(s => {
    const name = s.firstChild.textContent.trim(), url = (CFG.authors || {})[name];
    if (url) { const a = document.createElement('a'); a.href = url; a.target = '_blank'; a.rel = 'noopener'; a.textContent = name; s.replaceChild(a, s.firstChild); }
  });
  if (CFG.arxivId) $('#bibId').textContent = ' arXiv:' + CFG.arxivId;
  $('#bibCopy').addEventListener('click', async e => {
    const text = $('#bibText').textContent;
    try { await navigator.clipboard.writeText(text); } catch { const r = document.createRange(); r.selectNodeContents($('#bibText')); getSelection().removeAllRanges(); getSelection().addRange(r); document.execCommand('copy'); }
    e.target.textContent = 'Copied'; setTimeout(() => { e.target.textContent = 'Copy'; }, 1600);
  });

  /* ─────────── paper header: context-tile field ─────────── */
  (function tiles() {
    const cv = $('#tiles'), ctx = cv.getContext('2d'), host = cv.parentElement;
    let W = 0, H = 0, S = 24, cols = 0, rows = 0, visible = true, mx = -1e4, my = -1e4, raf = 0;
    const hash = (x, y) => { let h = (x * 374761393 + y * 668265263) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967295; };
    function size() {
      const dpr = Math.min(2, devicePixelRatio || 1); W = cv.clientWidth; H = cv.clientHeight;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      S = W < 700 ? 18 : 24; cols = Math.ceil(W / S) + 1; rows = Math.ceil(H / S) + 1;
    }
    function draw(t) {
      ctx.clearRect(0, 0, W, H);
      const nowX = Math.round(W * .76 / S) * S, win = (0.22 + 0.2 * (0.5 + 0.5 * Math.sin(t * .00022))) * W;
      const sweep = ((t * .05) % (W + 600)) - 300, q = S - 7;
      for (let j = 0; j < rows; j++) {
        const y = j * S, vf = Math.pow(Math.sin(Math.PI * clamp((y + S / 2) / H)), .7);
        for (let i = 0; i < cols; i++) {
          const x = i * S, n = hash(i, j), dx = nowX - x;
          let g = 0;
          if (dx > 0 && dx < win) g = Math.pow(1 - dx / win, 1.8) * (.45 + .55 * (0.5 + 0.5 * Math.sin(t * .0016 + n * 6.283)));
          else if (dx <= 0 && dx > -S * 5) g = .3 * n * (0.5 + 0.5 * Math.sin(t * .005 + i * 1.7 + j));   // imagined future
          const md = Math.hypot(x - mx, y - my); if (md < 170) g = Math.max(g, (1 - md / 170) * .75);
          g += Math.exp(-Math.pow((x - sweep) / 70, 2)) * .35 * n;
          const a = (.028 + .045 * n) * vf;
          ctx.fillStyle = g > .03 ? `rgba(179,237,85,${Math.min(.85, a + g * .5 * vf).toFixed(3)})` : `rgba(236,242,234,${a.toFixed(3)})`;
          ctx.fillRect(x + 3.5, y + 3.5, q, q);
        }
      }
      ctx.fillStyle = 'rgba(255,255,255,.28)'; ctx.fillRect(nowX + S / 2, 0, 1, H);
    }
    const loop = t => { draw(t); raf = visible ? requestAnimationFrame(loop) : 0; };
    size(); addEventListener('resize', () => { size(); if (reduced) draw(4000); });
    host.addEventListener('pointermove', e => { const r = cv.getBoundingClientRect(); mx = e.clientX - r.left; my = e.clientY - r.top; });
    host.addEventListener('pointerleave', () => { mx = my = -1e4; });
    if (reduced) { draw(4000); return; }
    new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && !raf) raf = requestAnimationFrame(loop); }).observe(host);
  })();

  /* ─────────── reveal on scroll ─────────── */
  const rio = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('is-in'); rio.unobserve(e.target); } }), { rootMargin: '0px 0px -6% 0px' });
  $$('[data-reveal]').forEach(n => rio.observe(n));

  /* ─────────── tooltip helper ─────────── */
  function tipFor(host) { let t = host.querySelector('.tip'); if (!t) { t = document.createElement('div'); t.className = 'tip'; host.appendChild(t); } return t; }
  function placeTip(tip, host, svg, x, y, html) {
    const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    tip.innerHTML = html; tip.style.opacity = 1;
    let left = r.left - hr.left + x * r.width / vb.width, top = r.top - hr.top + y * r.height / vb.height;
    left = clamp(left, tip.offsetWidth / 2, hr.width - tip.offsetWidth / 2); tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }

  /* ─────────── charts (rendered at the card's pixel width; redrawn on resize) ─────────── */
  const mounted = [];
  function mount(host, build, dur = 1200) {
    const c = { host, shown: false };
    c.draw = () => { host.querySelectorAll('svg').forEach(n => n.remove()); c.set = build(host, Math.max(280, host.clientWidth)); c.set(c.shown ? 1 : 0); };
    c.draw(); mounted.push(c);
    onView(host, () => { c.shown = true; animate(dur, p => c.set(p)); });
  }
  let rz = 0; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => mounted.forEach(c => c.draw()), 160); });

  function lineChart(host, { labels, series, min, max, ticks, xTitle, notes = [] }) {
    mount(host, (host, W) => {
      const H = Math.round(clamp(W * .6, 260, 340)), L = 40, R = 16, T = 26, B = 50;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': xTitle }, host);
      const x = i => L + i * (W - L - R) / (labels.length - 1), y = v => T + (1 - (v - min) / (max - min)) * (H - T - B);
      const grid = svgEl('g', { class: 'grid' }, svg);
      ticks.forEach(v => { svgEl('line', { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid); svgEl('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v; });
      labels.forEach((l, i) => { svgEl('text', { x: x(i), y: H - B + 20, 'text-anchor': 'middle' }, svg).textContent = l; });
      svgEl('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = xTitle;
      const id = 'clip' + Math.random().toString(36).slice(2), cp = svgEl('clipPath', { id }, svgEl('defs', {}, svg));
      const rect = svgEl('rect', { x: 0, y: 0, width: 0, height: H }, cp), g = svgEl('g', { 'clip-path': `url(#${id})` }, svg);
      const rad = s => s.dash ? 3.6 : 5;
      const dots = series.map(s => {
        svgEl('polyline', { points: s.values.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.dash ? 2.3 : 3.4, 'stroke-dasharray': s.dash ? '7 6' : '', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
        return s.values.map((v, i) => svgEl('circle', { cx: x(i), cy: y(v), r: rad(s), fill: COL.bg, stroke: s.color, 'stroke-width': 2.3 }, g));
      });
      notes.forEach(n => { const s = series[n.s]; svgEl('text', { x: x(n.i) + (n.dx || 0), y: y(s.values[n.i]) + (n.dy || -12), 'text-anchor': n.anchor || 'middle', class: 'val', style: `fill:${s.color}` }, g).textContent = n.text || s.values[n.i].toFixed(1); });
      const guide = svgEl('line', { x1: 0, x2: 0, y1: T, y2: H - B, stroke: 'rgba(236,242,234,.25)', 'stroke-dasharray': '3 4', opacity: 0 }, svg);
      const hit = svgEl('rect', { x: L - 12, y: T, width: W - L - R + 24, height: H - T - B, fill: 'transparent' }, svg), tip = tipFor(host);
      const pick = e => { const r = svg.getBoundingClientRect(); return Math.round(clamp(((e.clientX - r.left) * W / r.width - L) / (W - L - R)) * (labels.length - 1)); };
      hit.addEventListener('pointermove', e => {
        const i = pick(e); guide.setAttribute('x1', x(i)); guide.setAttribute('x2', x(i)); guide.setAttribute('opacity', 1);
        dots.forEach((ds, k) => ds.forEach((d, j) => d.setAttribute('r', j === i ? 6.5 : rad(series[k]))));
        placeTip(tip, host, svg, x(i), Math.min(...series.map(s => y(s.values[i]))) - 8,
          `<b>${labels[i]} s context</b>` + series.map(s => `<div><i style="background:${s.color}"></i>${s.name}: <strong>${s.values[i].toFixed(1)}%</strong></div>`).join(''));
      });
      hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; guide.setAttribute('opacity', 0); dots.forEach((ds, k) => ds.forEach(d => d.setAttribute('r', rad(series[k])))); });
      return p => rect.setAttribute('width', L - 10 + p * (W - L + 10));
    }, 1400);
  }

  function barChart(host, { groups, series, max = 100, ticks = [0, 25, 50, 75, 100], xTitle, unit = '%', labelFmt, tipFmt }) {
    mount(host, (host, W) => {
      const H = Math.round(clamp(W * .5, 260, 360)), L = 36, R = 8, T = 24, B = xTitle ? 54 : 34;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' }, host);
      const y = v => T + (1 - v / max) * (H - T - B), gw = (W - L - R) / groups.length, gap = 4, bw = Math.min(40, (gw * .8 - gap * (series.length - 1)) / series.length);
      const grid = svgEl('g', { class: 'grid' }, svg);
      ticks.forEach(v => { svgEl('line', { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid); svgEl('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v; });
      const bars = [], tip = tipFor(host);
      groups.forEach((gname, gi) => {
        const cx = L + gw * (gi + .5);
        svgEl('text', { x: cx, y: H - B + 20, 'text-anchor': 'middle', style: 'fill:#dfe6de' }, svg).textContent = gname;
        series.forEach((s, si) => {
          const v = s.values[gi], bx = cx - (series.length * bw + (series.length - 1) * gap) / 2 + si * (bw + gap);
          const r = svgEl('rect', { x: bx, y: y(0), width: bw, height: 0, rx: 3, fill: s.color }, svg);
          const lab = svgEl('text', { x: bx + bw / 2, y: y(v) - 7, 'text-anchor': 'middle', class: 'val', style: `fill:${s.color};opacity:0;font-size:${bw < 18 ? 10 : 12.5}px` }, svg);
          lab.textContent = labelFmt ? labelFmt(v, s, gi) : (Number.isInteger(v) ? v : v.toFixed(1));
          const hit = svgEl('rect', { x: bx - gap / 2, y: T, width: bw + gap, height: H - T - B, fill: 'transparent' }, svg);
          hit.addEventListener('pointermove', () => placeTip(tip, host, svg, bx + bw / 2, y(v) - 10, tipFmt ? tipFmt(v, s, gi) : `<b>${gname}</b><i style="background:${s.color}"></i>${s.name}: <strong>${v}${unit}</strong>`));
          hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; });
          bars.push({ r, lab, v });
        });
      });
      if (xTitle) svgEl('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = xTitle;
      return p => bars.forEach(b => { const h = (y(0) - y(b.v)) * p; b.r.setAttribute('y', y(0) - h); b.r.setAttribute('height', h); b.lab.style.opacity = p > .85 ? (p - .85) / .15 : 0; });
    }, 1100);
  }

  function hBars(host, { rows, max, mean }) {
    mount(host, (host, W) => {
      const stacked = W < 480, rowH = stacked ? 50 : 40, L = stacked ? 0 : Math.round(clamp(W * .34, 110, 230)), R = stacked ? 92 : 96, T = 6;
      const H = T + rows.length * rowH + (mean ? 28 : 4);
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' }, host), x = v => L + v / max * (W - L - R);
      const bars = rows.map((r, i) => {
        const yy = T + i * rowH, by = stacked ? yy + 22 : yy + 11;
        svgEl('text', stacked ? { x: 0, y: yy + 14 } : { x: L - 12, y: yy + 25, 'text-anchor': 'end' }, svg).innerHTML = svgText(r.label);
        svg.lastChild.setAttribute('style', `fill:${r.ours ? COL.ours : '#dfe6de'};font:${r.ours ? 600 : 400} 13.5px var(--sans)`);
        svgEl('rect', { x: L, y: by, width: W - L - R, height: 18, rx: 4, fill: 'rgba(236,242,234,.05)' }, svg);
        const b = svgEl('rect', { x: L, y: by, width: 0, height: 18, rx: 4, fill: r.color }, svg);
        const t = svgEl('text', { x: L + 8, y: by + 14, class: 'val', style: `fill:${r.color};opacity:0` }, svg); t.textContent = r.text;
        return { b, t, v: r.value };
      });
      if (mean) { const mx = x(mean.value); svgEl('line', { x1: mx, x2: mx, y1: T, y2: H - 22, stroke: COL.ours, 'stroke-dasharray': '4 4' }, svg);
        svgEl('text', { x: mx, y: H - 6, 'text-anchor': 'middle', class: 'val', style: `fill:${COL.ours}` }, svg).textContent = mean.label; }
      return p => bars.forEach(b => { const w = (x(b.v) - L) * p; b.b.setAttribute('width', w); b.t.setAttribute('x', L + w + 8); b.t.style.opacity = p > .8 ? (p - .8) / .2 : 0; });
    }, 1100);
  }

  /* ─────────── tables ─────────── */
  function table(host, spec, { fmt } = {}) {
    const { cols, rows, better = [], note, digits } = spec;
    const best = cols.map((_, j) => { if (!better[j]) return null; const v = rows.map(r => r[j]).filter(v => typeof v === 'number'); return better[j] === 'max' ? Math.max(...v) : Math.min(...v); });
    let h = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(r => {
      const tag = r[cols.length] || '';
      h += `<tr class="${tag}">` + cols.map((_, j) => {
        const v = r[j]; if (j === 0) return `<td>${v}</td>`;
        const s = typeof v !== 'number' ? '—' : fmt ? fmt(v, j) : v.toFixed(digits ? digits[j] : 1);
        return `<td class="${v === best[j] ? 'best' : ''}">${s}</td>`;
      }).join('') + '</tr>';
    });
    host.innerHTML = h + '</tbody></table>' + (note ? `<p class="tnote">${note}</p>` : '');
  }

  /* ─────────── build results ─────────── */
  const S = D.scaling;
  const trio = d => [{ name: 'LongLive2.0-Robot (AR)', color: COL.robot, values: d.robot }, { name: 'LongLive2.0 (AR)', color: COL.ar, values: d.ar, dash: true }, { name: 'Wan2.2 (bidirectional)', color: COL.bi, values: d.bi, dash: true }];
  lineChart($('#chartGR1'), { labels: S.gr1.ctx, series: trio(S.gr1), min: 58, max: 82, ticks: [60, 65, 70, 75, 80], xTitle: 'Observed context window (s)',
    notes: [{ s: 0, i: 4, dy: -14, text: '78.7' }] });
  lineChart($('#chartLIBERO'), { labels: S.libero.ctx, series: trio(S.libero), min: 92, max: 100.6, ticks: [92, 94, 96, 98, 100], xTitle: 'Observed context window (s)',
    notes: [{ s: 0, i: 1, dy: -14, text: '99.5' }] });
  table($('#tblGR1'), D.gr1Table);
  table($('#tblCtxLatency'), D.ctxLatencyTable);
  table($('#tblLIBERO'), D.libero); table($('#tblRoboTwin'), D.robotwin); table($('#tblDOMINO'), D.domino);

  const C = D.conveyor, sc = { pi: COL.pi, fast: COL.fast, ours: COL.ours, human: COL.human };
  barChart($('#chartConveyor'), { groups: C.speeds.map(s => s + ' cm/s'), series: C.series.map(s => ({ ...s, color: sc[s.key] })), xTitle: 'Conveyor speed',
    tipFmt: (v, s, gi) => `<b>${C.speeds[gi]} cm/s</b><i style="background:${sc[s.key]}"></i>${s.name}: <strong>${s.counts[gi]}</strong> (${v}%)` });
  hBars($('#chartStack'), { rows: D.stacking.map(r => ({ label: r.name, value: r.value, color: sc[r.key], ours: r.key === 'ours', text: `${r.count} · ${r.value}%` })), max: 100 });
  hBars($('#chartYAM'), { rows: D.yam.map(([l, v]) => ({ label: l, value: v, color: COL.ours, text: v + '%' })), max: 100, mean: { value: D.yamMean, label: `mean ${D.yamMean}%` } });

  const lat = D.latency.rows;
  hBars($('#chartLatency'), { rows: lat.map(r => ({ label: r[0], value: r[1], color: r[3] === 'ours' ? COL.ours : '#6f7d82', ours: r[3] === 'ours', text: r[1].toFixed(1) + ' ms' })), max: 3700 });
  table($('#tblLatency'), D.latency);
  table($('#tblAsync'), D.async);
  const A = D.accel;
  table($('#tblAccel'), {
    cols: ['Optimization', 'RTX 5090 (ms)', '×', 'DGX Spark (ms)', '×', 'Jetson AGX Thor (ms)', '×'], better: [null, 'min', 'max', 'min', 'max', 'min', 'max'],
    rows: A.stages.map((s, i) => [s, A.devices[0].ms[i], A.devices[0].x[i], A.devices[1].ms[i], A.devices[1].x[i], A.devices[2].ms[i], A.devices[2].x[i], i === 1 || i === 6 ? 'sep' : '']),
    note: 'Rows 2–6: shared optimizations · rows 7–8: device-specific operator tuning. All speedups relative to BF16 eager.'
  }, { fmt: (v, j) => j % 2 === 0 ? v.toFixed(1) + '×' : v.toFixed(1) });

  const R = D.rc365, pick = n => R.rows.find(r => r[0] === n);
  const rcSeries = [['GPT-6 Astra', '#9aa7ad'], ['π<sub>0.5</sub> + GPT-6 Astra', COL.pi], ['Long-WAM', COL.ar], ['Long-WAM + GPT-6 Astra', COL.ours]].map(([n, c]) => ({ name: n, color: c, values: pick(n).slice(1, 5) }));
  barChart($('#chartRC365'), { groups: ['Atomic seen', 'Composite seen', 'Composite unseen', 'Overall'], series: rcSeries,
    tipFmt: (v, s, gi) => `<b>${['Atomic-Seen', 'Composite-Seen', 'Composite-Unseen', 'Overall'][gi]}</b><i style="background:${s.color}"></i>${s.name}: <strong>${v.toFixed(1)}%</strong>`,
    labelFmt: v => v.toFixed(1) });
  table($('#tblRC365'), R);

  /* ─────────── explorers ─────────── */
  (function contextExplorer() {
    const tl = $('#ctxTimeline'), now = tl.querySelector('.now');
    for (let k = 0; k < 48; k++) tl.insertBefore(document.createElement('i'), now);
    const tiles = $$('i', tl), secs = [0, 2.4, 4.8, 9.6, 19.2, 38.4], slider = $('#ctxSlider');
    const rows = [['LongLive2.0-Robot (AR)', 'robot'], ['LongLive2.0 (AR)', 'ar'], ['Wan2.2 (bidirectional)', 'bi']].map(([n, k]) => {
      const d = document.createElement('div'); d.className = 'ctx-bar'; d.innerHTML = `<span>${n}</span><span class="track"><i style="background:${COL[k]}"></i></span><b></b>`; $('#ctxBars').appendChild(d); return { d, k };
    });
    const note = document.createElement('p'); note.className = 'fine'; $('#ctxExplorer').appendChild(note);
    function set(i) {
      const n = Math.round(secs[i] / .8);
      tiles.forEach((t, k) => t.classList.toggle('on', k >= 48 - n));
      $('#ctxSec').textContent = secs[i] + ' s';
      rows.forEach(({ d, k }) => { const v = S.gr1[k][i]; d.querySelector('i').style.width = v + '%'; d.querySelector('b').textContent = v.toFixed(1) + '%'; });
      const l = D.ctxLatency[S.gr1.ctx[i]]; $('#ctxLat').textContent = l ? l.toFixed(1) + ' ms' : 'n/a';
      note.innerHTML = `RoboCasa GR-1 success for each video-expert initialization (one separately trained model per window). Latency: Long-WAM infrastructure on RTX 5090${i === 5 ? '. At 38.4 s, ' + S.padding384 + '% of sampled history frames are padding and latency is not reported.' : '; 0 s keeps only the current observation.'}`;
    }
    let touched = false; slider.addEventListener('input', () => { touched = true; set(+slider.value); });
    set(4);
    onView($('#ctxExplorer'), async () => { if (reduced) return; for (let i = 0; i <= 4; i++) { if (touched) return; slider.value = i; set(i); await new Promise(r => setTimeout(r, 650)); } }, .5);
  })();

  (function accelExplorer() {
    const dots = $('#accDots'), host = $('#accDevices'), slider = $('#accSlider');
    A.stages.forEach((_, i) => { const d = document.createElement('i'); if (A.groups[i].startsWith('Device')) d.classList.add('dev'); dots.appendChild(d); });
    const cards = A.devices.map(dv => { const c = document.createElement('div'); c.className = 'dev';
      c.innerHTML = `<img src="static/images/${dv.img}" alt="${dv.name}" loading="lazy"><div class="body"><h4>${dv.name}</h4><div class="ms"><span></span><small>ms</small></div><div><span class="x"></span> <span class="muted">vs. BF16 eager</span></div><div class="track"><i></i></div></div>`;
      host.appendChild(c); return c; });
    function set(i) {
      $('#accName').textContent = A.stages[i];
      const g = $('#accGroup'); g.textContent = A.groups[i] === 'Shared' ? 'Shared optimization' : A.groups[i] === 'Baseline' ? 'Baseline' : 'Device-specific tuning'; g.classList.toggle('dev', A.groups[i].startsWith('Device'));
      $$('i', dots).forEach((d, k) => d.classList.toggle('on', k <= i));
      cards.forEach((c, k) => { const dv = A.devices[k]; c.querySelector('.ms span').textContent = dv.ms[i].toFixed(1); c.querySelector('.x').textContent = dv.x[i].toFixed(1) + '×'; c.querySelector('.track i').style.width = (dv.ms[i] / 1400 * 100) + '%'; });
    }
    let touched = false; slider.addEventListener('input', () => { touched = true; set(+slider.value); });
    set(7);
    onView($('#accExplorer'), async () => { if (reduced) return; for (let i = 0; i <= 7; i++) { if (touched) return; slider.value = i; set(i); await new Promise(r => setTimeout(r, 520)); } }, .5);
  })();

  /* ─────────── tabs ─────────── */
  $$('.tab').forEach(b => b.addEventListener('click', () => {
    $$('.tab').forEach(x => x.classList.toggle('is-on', x === b));
    $$('.tabpane').forEach(p => p.classList.toggle('is-on', p.dataset.pane === b.dataset.tab));
  }));

  /* ─────────── demo videos: load and play only when visible ─────────── */
  const vio = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) { if (!v.getAttribute('src')) v.src = v.dataset.src; v.play().catch(() => {}); }
    else if (!v.paused) v.pause();
  }), { rootMargin: '150px 0px', threshold: .15 });
  $$('video[data-src]').forEach(v => vio.observe(v));
})();
