(function () {
  'use strict';
  const D = window.AGL;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const RM = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const cut = (x, n) => x.length > n ? x.slice(0, n - 1) + '…' : x;
  const NICK = { graphics_card_parts: 'GPU board', component_boxes: 'Parts boxes', workshop: 'Workshop' };
  const nick = b => NICK[b.id] || b.name;

  function sentences(p) { return (p || '').match(/[^.!?]+[.!?]+["”’)]*|[^.!?]+$/g) || []; }
  function shotText(r, cap = 140) {
    const ss = sentences(r.prompt).map(s => s.trim()).filter(s => s && !(/^(Still|Back) (at|in|on|inside)\b/.test(s) && s.length < 70));
    let out = '';
    for (const s of ss) { if (!out) out = s; else if ((out + ' ' + s).length <= cap) out += ' ' + s; else break; }
    if (out.length > cap + 30) out = out.slice(0, cap).replace(/\s+\S*$/, '') + '…';
    return out || r.short;
  }
  function typewrite(el, text, color) {
    clearInterval(el._t);
    if (RM) { el.textContent = text; return; }
    let n = 0; el.innerHTML = '<span class="caret"></span>';
    el._t = setInterval(() => {
      n += 2; el.innerHTML = esc(text.slice(0, n)) + (n < text.length ? '<span class="caret"></span>' : '');
      if (n >= text.length) clearInterval(el._t);
    }, 16);
  }
  function fit(box, stage, W, H, maxH) {
    const w = box.clientWidth;
    let s = w / W;
    if (maxH) s = Math.min(s, maxH / H);
    stage.style.width = W + 'px'; stage.style.height = H + 'px';
    stage.style.transform = 'scale(' + s + ')';
    stage.style.left = Math.max(0, (w - W * s) / 2) + 'px';
    box.style.height = (H * s) + 'px';
    return s;
  }
  function tipFor(host) { let t = $('.tip', host); if (!t) { t = document.createElement('div'); t.className = 'tip'; host.appendChild(t); } return t; }

  const gpu = (function () {
    const h3 = D.demos.find(d => d.id === 'gpu_build_pc') || D.demos[0], fh = D.rt && D.rt.fasth3 && D.rt.fasth3[h3.id];
    if (!fh) return h3;
    return Object.assign({}, h3, { video: fh.video, poster: fh.poster, shots: h3.shots.map((r, j) => fh.shots[j]
      ? Object.assign({}, r, { t0: fh.shots[j].t0, t1: fh.shots[j].t1, hero: fh.shots[j].hero || r.hero }) : r) });
  })();

  const hero = (function () {
    const heroEl = $('#hero'), inner = $('#heroInner'), st = $('#heroStage'), box = $('#heroBox');
    const qn = $('#heroQ .qn'), qt = $('#heroQ .qt'), tag = $('#heroTag'), qEl = $('#heroQ');
    const NARROW = window.innerWidth < 720;
    const W = NARROW ? 560 : 960, H = 400, VL = NARROW ? 64 : 100, VT = 26, VH = 268, NOWX = NARROW ? 290 : 440;
    const NOW = VL + NOWX, BT = VT + VH + 12;
    const FW = 150, FG = 6, FS = FW + FG;
    const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const easeOut = t => 1 - Math.pow(1 - t, 3);
    const spring = t => t <= 0 ? 0 : t >= 1 ? 1 : 1 - Math.exp(-6 * t) * Math.cos(9 * t);
    const seg = (x, a, b) => clamp((x - a) / (b - a), 0, 1);
    const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);

    const TAG = [['A '], ['real-time', 1], [' video generator'], [' that drifts', 'x'], ['\n'], ['with a memory agent, running '], ['in parallel', 1], ['.']];
    const GA = [61, 103, 10], GB = [94, 154, 18];
    let part = 1;
    tag.innerHTML = TAG.map(([t, em]) => {
      if (t === '\n') { part = 2; return '<br>'; }
      const x = em === 'x';
      const s = [...t].map((c, i) => `<span class="ch" data-p="${x ? 'x' : part}"${em === 1 ? ` style="color:rgb(${GA.map((v, j) => Math.round(lerp(v, GB[j], i / (t.length - 1)))).join(',')})"` : ''}>${esc(c)}</span>`).join('');
      return em === 1 ? `<em>${s}</em>` : x ? `<span class="flaw">${s}</span>` : s;
    }).join('') + '<i class="tcaret" aria-hidden="true"></i>';
    const CH = $$('.ch', tag), caret = $('.tcaret', tag);
    const NX = CH.findIndex(c => c.dataset.p === 'x'), NB = CH.findIndex(c => c.dataset.p === '2');
    const DL = CH.map((el, i) => .024 + ((i * 7919) % 11) / 11 * .022 + (i && CH[i - 1].textContent === ',' ? .3 : 0) +
      (i && el.parentNode.tagName === 'EM' && CH[i - 1].parentNode.tagName !== 'EM' ? .12 : 0));
    const TT = [], TO = CH.map(() => Infinity);
    let tt = .7;
    for (let i = 0; i < NX; i++) TT[i] = tt += DL[i];
    const T1 = tt, T_UP = tt + .55, T_AB = T_UP + .45;
    const TA = 4.3;
    tt = T_AB + 3.1;
    const TX0 = tt;
    for (let i = NX; i < NB; i++) TT[i] = tt += DL[i] + .012;
    const TX1 = tt, TD0 = T_AB + TA + .4;
    tt = TD0;
    for (let i = NB - 1; i >= NX; i--) TO[i] = tt += .03;
    const TD1 = tt;
    tt = TD1 + .28;
    const TB0 = tt;
    for (let i = NB; i < CH.length; i++) TT[i] = tt += DL[i];
    const T_END = tt;
    const BUSY = [[.7, T1], [TX0, TX1], [TD0, TD1], [TB0, T_END]];
    let typed = 0, lastC = 0, slide = 0, slideAt = -9;
    const lineH = () => parseFloat(getComputedStyle(tag).fontSize) * 1.12;
    function placeCaret() {
      const lh = lineH();
      let x = tag.clientWidth / 2, y = 0;
      if (typed) { const r = CH[typed - 1]; x = r.offsetLeft + r.offsetWidth; y = r.offsetTop + (r.offsetHeight - lh) / 2; }
      caret.style.transform = `translate(${f1(x)}px,${f1(y)}px)`;
      caret.style.height = f1(lh) + 'px';
    }
    function centerTag(t) {
      const lh = lineH(), r = typed ? CH[typed - 1] : null;
      const c = r ? r.offsetTop + r.offsetHeight / 2 + lh / 2 : lh;
      if (lastC && Math.abs(c - lastC) > 2) { slide = (c - lastC) / 2; slideAt = t; }
      lastC = c;
      const dy = inner.clientHeight / 2 - (tag.offsetTop + c / 2) + slide * (1 - easeOut(seg(t, slideAt, slideAt + .45)));
      tag.style.transform = `translateY(${f1(dy)}px)`;
    }
    function reserve() {
      const on = CH.map(c => c.classList.contains('on'));
      tag.style.minHeight = ''; CH.forEach(c => c.classList.add('on'));
      const h = tag.offsetHeight;
      CH.forEach((c, i) => { if (!on[i]) c.classList.remove('on'); });
      tag.style.minHeight = h + 'px';
    }

    const AW = 120, AH = 68, AP = AW + 6, STP = 28;
    const NF = 22, IC = 11, FEEDS = [11, 15];
    const P0 = -160, V0 = 210, V1 = 100, R1 = 8.6, R2 = 9.6, TW = 9.6, WIPE = 1.6;
    const AX = NARROW ? VL + 74 : NOW - 250, AY = 200;
    const DOCK = NARROW ? [4, 176, .4] : [VL - 42, 147, .42];
    const PRW = 84, PRX = NARROW ? AX + 52 : AX - 190, PRY = AY - 14;
    const drift = i => i < IC ? clamp((i - 2.5) / 5, 0, 1) : 0;
    const rnd = (i, k) => { const x = Math.sin(i * 127.1 + k * 311.7) * 43758.5453; return x - Math.floor(x); };
    function frameArt(i, d) {
      const j = k => (rnd(i, k) - .5) * 2 * d;
      const cx = 42 + 12 * Math.sin(i * .5) + 16 * j(1), cy = 25 + 8 * j(2);
      const rx = 11 * (1 + .45 * Math.abs(j(3))), ry = 11 * (1 - .3 * Math.abs(j(4)));
      const tilt = 9 * j(5), fx = 84 + 3 * Math.sin(i * .7) + 20 * j(6), fh = 15 * (1 + .5 * j(7));
      let s = '';
      if (d > .25) s += `<ellipse cx="${f1(cx + 12 * d)}" cy="${f1(cy - 5 * d)}" rx="${f1(rx)}" ry="${f1(ry)}" fill="#E6A73E" opacity="${f2(.45 * d)}"/>`;
      s += `<ellipse cx="${f1(cx)}" cy="${f1(cy)}" rx="${f1(rx)}" ry="${f1(ry)}" fill="#E6A73E"/>`;
      s += `<g transform="rotate(${f1(tilt)} 60 50)"><path d="M-12 45Q28 38 60 44T132 43V82H-12z" fill="#A9C78A"/><path d="M-12 54Q50 49 132 53V82H-12z" fill="#83AB64"/></g>`;
      s += `<rect x="${f1(fx)}" y="${f1(50 - fh)}" width="7" height="${f1(fh)}" rx="3.5" fill="#2F3B33"/>`;
      if (d > .35) for (let k = 0; k < 3; k++) s += `<rect x="0" y="${f1(8 + rnd(i, 10 + k) * 52)}" width="${AW}" height="${f1(1.5 + rnd(i, 20 + k) * 4)}" fill="${k % 2 ? '#B45CD6' : '#4FC3D9'}" opacity="${f2(d * .45)}"/>`;
      if (d > .6) {
        s += `<ellipse cx="${f1(20 + rnd(i, 30) * 80)}" cy="${f1(14 + rnd(i, 31) * 40)}" rx="${f1(14 + rnd(i, 32) * 18)}" ry="${f1(4 + rnd(i, 33) * 6)}" fill="#E6A73E" opacity="${f2((d - .6) * 1.2)}"/>`;
        const ty = 18 + rnd(i, 34) * 30;
        s += `<rect x="${f1(-10 + rnd(i, 35) * 30)}" y="${f1(ty)}" width="${AW}" height="${f1(6 + rnd(i, 36) * 8)}" fill="#83AB64" opacity="${f2((d - .6) * 1.6)}"/>`;
      }
      return s;
    }
    const AFR = [];
    let film = `<div class="fl-base" style="left:-900px;width:${NF * AP + 1800}px"></div>`;
    for (let i = 0; i < NF; i++) {
      const d = drift(i), x = i * AP;
      const base = d ? `hue-rotate(${Math.round(d * 110)}deg) saturate(${f2(1 + d * .9)}) contrast(${f2(1 - d * .28)}) blur(${f2(d * d * 1.8)}px)` : '';
      film += `<div class="fl-fr${FEEDS.includes(i) ? ' mk' : ''}" style="left:${x}px"><div class="sc"${base ? ` style="filter:${base}"` : ''}><svg width="${AW}" height="${AH}" viewBox="0 0 ${AW} ${AH}">${frameArt(i, d)}</svg></div><div class="nz" style="opacity:${f2(Math.min(1, d * 1.05))}"></div></div>`;
      AFR.push({ x, base, s: null });
    }
    const MB = [0, 1, 2, 3].map(i => `<div class="mb" style="left:${VL + i * 76}px;top:${BT}px"><i></i><b style="width:${[34, 26, 38, 30][i]}px"></b><b style="width:${[22, 34, 18, 28][i]}px"></b><b style="width:${[28, 20, 30, 16][i]}px"></b></div>`).join('');
    const ORG = [[0, 1, 2], [1, 3, 2]];
    const AGENT = `<svg viewBox="0 0 64 64" width="64" height="64">
      <ellipse class="ag-sh" cx="32" cy="61" rx="19" ry="3" fill="#2F4A18" opacity=".16"/>
      <g class="ag-b"><path class="ag-body" fill="#78BE3C"/>
        <ellipse class="ag-e1" cx="25" cy="35" rx="4.7" ry="6.5" fill="#fff"/><ellipse class="ag-e2" cx="39" cy="35" rx="4.7" ry="6.5" fill="#fff"/></g></svg><i class="ag-pulse"></i>`;

    const ALL = gpu.shots;
    const nextOf = r => ALL[ALL.indexOf(r) + 1];
    const SEQ = ALL.filter(r => r.hero && nextOf(r)), N = SEQ.length;
    const T = 6.4;
    const SW = 4 * FS + 18;
    const STEPS = [['read prompt', 92], ['update state', 112], ['draw', 136], ['check', 68]];
    const DRAW = 6 + STEPS[0][1] + 6 + STEPS[1][1] + 6, CHECK = DRAW + STEPS[2][1] + 6;
    const COMMIT = 6 + STEPS.reduce((a, s) => a + s[1] + 6, 0) + 14;
    const US = 6 + STEPS[0][1] + 6, USW = STEPS[1][1];
    const NOTE_IDS = ['alex', 'graphics_card_parts', 'component_boxes', 'workshop'];
    const NOTE_FIX = { alex: ['blond hair', 'light stubble', 'navy apron', 'wrist strap'], component_boxes: ['sealed', 'stacked by the mat'], workshop: ['broad windows', 'pegboards', 'concrete floor'] };
    const B1 = ['chip soldered', 'memory ring', 'caps and chokes', 'gold PCIe fingers', 'thermal pads', 'paste on the die'], B2 = ['heatsink secured', 'screws cross-tightened', 'fans in the shroud'];
    const BOARD = { 4: ['chip seated'], 5: B1.slice(0, 1), 6: B1.slice(0, 2), 7: B1.slice(0, 3), 8: B1.slice(0, 4), 9: B1.slice(0, 5), 10: B1, 11: [...B1, 'heatsink ready'], 12: B2.slice(0, 2), 13: B2, 15: [...B2, 'backplate on'] };
    const noteOf = i => { let k = i; while (k > 0 && !BOARD[k]) k--; return Object.assign({ graphics_card_parts: BOARD[k] || [] }, NOTE_FIX); };
    const PW = NARROW ? 46 : 64, NW = NARROW ? 66 : 92, PG = NARROW ? 3 : 4, PP = PW + PG + NW + (NARROW ? 8 : 14);
    function tnoteHTML(r, nx, x0) {
      const a = noteOf(nx.i), b = noteOf(r.i), L = [];
      let title = 'no change';
      NOTE_IDS.forEach(id => {
        const A = a[id] || [], B = b[id] || [];
        let p = 0; while (p < A.length && p < B.length && A[p] === B[p]) p++;
        const drop = B.slice(p), add = A.slice(p);
        if (!drop.length && !add.length) return;
        title = NICK[id] || 'Alex';
        if (drop.length > 2) L.push(`<s>${drop.length} old facts</s>`); else drop.forEach(t => L.push(`<s>${esc(t)}</s>`));
        add.forEach(t => L.push(`+ ${esc(t)}`));
      });
      return `<div class="hz-tnote${L.length ? '' : ' idle'}" data-k="card" data-x="${x0 + US + 8}" style="left:${x0 + US - 2}px"><b>${esc(title)}</b>${L.map((l, k) => `<span style="animation-delay:${(k * .22).toFixed(2)}s">${l}</span>`).join('')}</div>`;
    }
    let w = '', paths = '';
    for (let c = 0; c < 3; c++) SEQ.forEach((r, k) => {
      const x0 = (c * N + k) * SW, nx = nextOf(r);
      r.hero.forEach((src, q) => w += `<div class="hz-fr" data-k="fr" data-x="${x0 + q * FS}" style="left:${x0 + q * FS}px"><img src="${src}" alt="" decoding="async"><div class="nz"></div></div>`);
      let x = x0 + 6;
      STEPS.forEach(([lab, ww]) => { w += `<div class="hz-step" data-k="step" data-x="${x}" data-w="${ww}" style="left:${x}px;width:${ww}px">${lab}</div>`; x += ww + 6; });
      const xc = x0 + COMMIT;
      w += `<div class="hz-dia" data-k="dia" data-x="${xc}" style="left:${xc}px"></div>`;
      w += `<div class="hz-step idle" style="left:${xc + 18}px;width:${x0 + SW - 12 - (xc + 18)}px"></div>`;
      const ch = (nx.bank || []).filter(b => b.changed && b.kind !== 'scene').slice(0, 2);
      ch.forEach((b, q) => w += `<div class="hz-card" data-k="card" data-x="${x0 + DRAW + 30 + q * 50}" style="left:${x0 + DRAW - 14 + q * 72}px"><div class="im" style="width:64px;height:64px"><img src="${b.img}" alt=""></div><div class="nm" style="width:66px;overflow:hidden;text-overflow:ellipsis"><b>${esc(nick(b))}</b><br>${b.new ? 'new' : 'v' + b.version}</div></div>`);
      const mu = (nx.used || []).filter(u => u.role === 'M').pop() || (nx.used || [])[0];
      if (mu) w += `<div class="hz-card" data-k="card" data-x="${x0 + CHECK + 8}" style="left:${x0 + CHECK - 4}px"><div class="im" style="width:112px;height:63px"><img src="${mu.img}" alt=""></div><div class="nm">memory for shot ${nx.i + 1}</div></div>`;
      w += tnoteHTML(r, nx, x0);
      paths += `<path data-x="${xc}" d="M ${xc} 120 C ${xc} 98, ${x0 + SW + 75} 118, ${x0 + SW + 75} 94" style="opacity:0"/>`;
    });
    const WW = 3 * N * SW;
    st.innerHTML = `
      <div class="hz-lab" style="top:${VT + 40}px">Video</div>
      <div class="hz-lab lz" style="top:${VT + 123}px;opacity:0">Agent</div>
      <div class="hz-lab lz" style="top:${BT + 22}px;opacity:0">Memory</div>
      <div class="ab">
        <div class="fl-view" style="left:${VL}px;top:${STP}px;width:${W - VL}px;height:96px"><div class="fl-world">${film}</div></div>
        ${MB}
        <svg class="ab-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><path class="ab-feed"/></svg>
        <div class="ab-pr" style="left:${PRX}px;top:${PRY}px;width:${PRW}px"><span>prompt</span><em><b></b><b></b></em></div>
        ${'<i class="pt"></i>'.repeat(8)}${'<i class="pm"></i>'.repeat(3)}${'<i class="pq"></i>'.repeat(3)}<i class="pf"></i>
      </div>
      <div class="hz-real" style="visibility:hidden">
        <div class="hz-film" style="left:${VL}px;top:${VT - 3}px;width:${NOWX}px"></div>
        <div class="hz-view" style="left:${VL}px;top:${VT}px;height:${VH}px;width:${W - VL}px">
          <div class="hz-world" style="width:${WW}px">${w}<svg class="hz-svg" width="${WW}" height="${VH}" viewBox="0 0 ${WW} ${VH}">${paths}<circle class="hz-dot" r="4.5" fill="var(--g)" style="opacity:0"/></svg></div></div>
        <div class="hz-bank" style="top:${BT}px;left:${VL}px;gap:${PP - PW}px"></div>
        ${NOTE_IDS.map((id, k) => `<div class="tn" data-id="${id}" style="left:${VL + k * PP + PW + PG}px;top:${BT}px;width:${NW}px;height:${PW}px"><div class="v"><div class="in"></div></div><em class="tg"></em></div>`).join('')}
        <svg class="tm-fly" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><path/><circle r="4.5"/></svg>
      </div>
      <div class="hz-now" style="left:${NOW}px;top:${VT - 10}px;height:${VH + 10}px;opacity:0"><i class="halo"></i><i class="spill"></i><i class="beam"></i><span>now</span></div>
      <div class="hz-live" style="top:${BT + 22}px"><i></i><span>streaming</span></div>
      <div class="ab-glow"></div>
      <div class="hz-agent">${AGENT}</div>`;
    st.classList.toggle('nw', NARROW);
    const live = $('.hz-live span', st), lanes = $$('.lz', st), nowEl = $('.hz-now', st);

    const agent = (function () {
      const el = $('.hz-agent', st), body = $('.ag-body', el), grp = $('.ag-b', el), sh = $('.ag-sh', el);
      const e1 = $('.ag-e1', el), e2 = $('.ag-e2', el), pulse = $('.ag-pulse', el);
      let lastT = 0, gx = 0, gy = 0, hopAt = -99, wob = 0;
      const NPT = 36;
      function blob(t, k, sx, sy) {
        const P = [];
        for (let i = 0; i < NPT; i++) {
          const th = i / NPT * Math.PI * 2;
          const r = 23.5 * (1 + k * (.034 * Math.sin(2 * th + t * 1.5) + .02 * Math.sin(3 * th - t * 2.1 + 1) + .009 * Math.sin(5 * th + t * 2.9)));
          const sn = Math.sin(th);
          P.push([32 + r * Math.cos(th) * sx, 33 + r * sn * sy * (sn > 0 ? 1 - .1 * sn : 1)]);
        }
        const mid = (A, B) => [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2];
        let m = mid(P[NPT - 1], P[0]), d = `M${f1(m[0])} ${f1(m[1])}`;
        for (let i = 0; i < NPT; i++) { const q = mid(P[i], P[(i + 1) % NPT]); d += `Q${f1(P[i][0])} ${f1(P[i][1])} ${f1(q[0])} ${f1(q[1])}`; }
        return d + 'Z';
      }
      return {
        hop(t) { hopAt = t; },
        draw(t, o) {
          const dt = clamp(t - lastT, 0, .1); lastT = t;
          el.style.opacity = f2(o.a);
          el.style.transform = `translate(${f1(o.x)}px,${f1(o.y)}px) scale(${o.s.toFixed(3)})`;
          let tx, ty;
          if (o.look) {
            const cx = o.x + 32 * o.s, cy = o.y + 33 * o.s;
            tx = clamp((o.look.x - cx) / 120, -1, 1); ty = clamp((o.look.y - cy) / 90, -1, 1);
          } else {
            const k = Math.floor(t / 1.35), u = ease(seg(t - k * 1.35, 0, .32));
            const A = [rnd(k, 41) * 2 - 1, (rnd(k, 42) * 2 - 1) * .55], B = [rnd(k + 1, 41) * 2 - 1, (rnd(k + 1, 42) * 2 - 1) * .55];
            tx = lerp(A[0], B[0], u) * .9; ty = lerp(A[1], B[1], u);
          }
          const kk = 1 - Math.exp(-dt * 9);
          gx += (tx - gx) * kk; gy += (ty - gy) * kk;
          wob += dt * (1 + o.busy * 1.4);
          const br = Math.sin(t * 2.6) * .025, h = t - hopAt;
          const hy = h > 0 && h < .55 ? -9 * Math.sin(Math.PI * h / .55) : 0;
          const sq = h > 0 && h < .9 ? Math.sin(h * 18) * Math.exp(-h * 5) * .12 : 0;
          const sx = 1 + br + sq, sy = 1 - br - sq;
          body.setAttribute('d', blob(wob * 1.6, 1 + o.busy * .3, sx, sy));
          grp.setAttribute('transform', `translate(${f2(gx * 1.6)} ${f2(hy)}) rotate(${f2(gx * 5)} 32 40)`);
          sh.setAttribute('rx', f1(19 * (1 + hy / 30) * sx)); sh.setAttribute('opacity', f2(.18 * (1 + hy / 18)));
          const bt = (t + .9) % 3.6, bl = bt < .16 ? 1 - Math.sin(Math.PI * bt / .16) * .9 : 1;
          const ex = gx * 7, ey = gy * 5 - (sy - 1) * 10, pinch = Math.abs(gx) * 1.4;
          e1.setAttribute('cx', f2(25 + ex + pinch * (gx > 0 ? 1 : 0))); e2.setAttribute('cx', f2(39 + ex - pinch * (gx < 0 ? 1 : 0)));
          e1.setAttribute('cy', f2(35 + ey)); e2.setAttribute('cy', f2(35 + ey));
          e1.setAttribute('rx', f2(4.7 * (1 - .24 * Math.max(0, -gx)))); e2.setAttribute('rx', f2(4.7 * (1 - .24 * Math.max(0, gx))));
          e1.setAttribute('ry', f2(6.5 * bl)); e2.setAttribute('ry', f2(6.5 * bl));
          const fp = o.pp >= 0 ? o.pp : seg(h, 0, .9), on = o.pp >= 0 ? o.pp > 0 && o.pp < 1 : h < .9;
          pulse.style.opacity = on ? f2(.8 * (1 - fp)) : 0;
          pulse.style.transform = `scale(${f2(1 + .6 * easeOut(fp))})`;
        }
      };
    })();

    const ab = $('.ab', st), fworld = $('.fl-world', st), glow = $('.ab-glow', st);
    $$('.fl-fr', st).forEach((el, i) => { AFR[i].el = el; AFR[i].sc = el.firstChild; });
    const blocks = $$('.mb', st), feedP = $('.ab-feed', st), PT = $$('.pt', st), PM = $$('.pm', st), PQ = $$('.pq', st), PF = $('.pf', st), PR = $('.ab-pr', st);
    function absPos(ta) {
      if (ta <= R1) return P0 + V0 * ta;
      if (ta <= R2) { const d = ta - R1; return P0 + V0 * R1 + V0 * d - (V0 - V1) * d * d / (2 * (R2 - R1)); }
      return P0 + V0 * R1 + (V0 + V1) / 2 * (R2 - R1) + V1 * (ta - R2);
    }
    const BOT = STP + 13 + AH;
    const qb = (a, c, b, q) => (1 - q) * (1 - q) * a + 2 * (1 - q) * q * c + q * q * b;
    function renderAbs(ta) {
      const pos = absPos(ta);
      fworld.style.transform = `translateX(${f1(NOWX - pos)}px)`;
      for (const f of AFR) {
        const dd = (f.x - pos) / AP, s = dd <= 0 ? 'p' : dd < 1 ? 'd' : 'w';
        if (s === 'd') { f.sc.style.filter = `${f.base} blur(${f1(dd * 5)}px) brightness(${f2(1 + dd * .5)})`; f.sc.style.opacity = f2(1 - dd * .85); }
        if (s !== f.s) { f.el.classList.toggle('pend', s === 'w'); if (s === 'p') { f.sc.style.filter = f.base; f.sc.style.opacity = 1; } f.s = s; }
      }
      const a = seg(ta, TA, TA + .6);
      lanes.forEach(l => l.style.opacity = f2(a));
      blocks.forEach(b => b.style.opacity = f2(a));
      let busy = 0, pp = -1, feedO = 0, pfOn = false, look = null, hop = -1;
      PT.forEach(p => p.style.opacity = 0); PM.forEach(p => p.style.opacity = 0);
      PQ.forEach(p => p.style.opacity = 0);
      let prO = 0, prX = 0;
      FEEDS.forEach((f, c) => {
        const pf = (f - 1) * AP;
        const c0 = pf - 420;
        if (pos >= c0 - 70 && pos < pf - 190) {
          prO = seg(pos, c0 - 70, c0 - 20) * (1 - seg(pos, pf - 230, pf - 190));
          prX = (NARROW ? 16 : -16) * (1 - easeOut(seg(pos, c0 - 70, c0 - 10)));
        }
        const ex = NARROW ? PRX : PRX + PRW;
        PT.forEach((p, k) => {
          const l0 = c0 + k * 18, q = seg(pos, l0, l0 + 110);
          if (q <= 0 || q >= 1) return;
          const sx = ex + (NARROW ? 6 : -6) - rnd(k, c + 3) * 10, sy = PRY + 6 + rnd(k, c + 7) * 16;
          const e = ease(q), x = qb(sx, (sx + AX) / 2, AX, e), y = qb(sy, AY - 22 + rnd(k, c + 9) * 44, AY, e);
          p.style.transform = `translate(${f1(x)}px,${f1(y)}px) scale(${f2(1 - .5 * q)})`;
          p.style.opacity = f2(Math.min(1, q * 6, (1 - q) * 5));
        });
        if (c > 0) ORG[c - 1].forEach((b, k) => {
          const l0 = c0 + 30 + k * 26, q = seg(pos, l0, l0 + 120), p = PM[k];
          if (q <= 0 || q >= 1) return;
          const bx = VL + b * 76 + 32, by = BT + 18, e = ease(q);
          p.style.transform = `translate(${f1(qb(bx, (bx + AX) / 2, AX, e))}px,${f1(qb(by, AY + 50, AY + 14, e))}px) scale(${f2(1 - .4 * q)})`;
          p.style.opacity = f2(Math.min(1, q * 6, (1 - q) * 5));
        });
        if (pos >= c0 && pos < pf - 200) { busy = Math.max(busy, Math.min(1, seg(pos, c0, c0 + 60) * 1.2)); look = { x: PRX + PRW / 2, y: PRY + 14 }; }
        ORG[c].forEach((b, k) => {
          const l0 = pf - 250 + k * 22, q = seg(pos, l0, l0 + 90), p = PQ[k];
          if (q > 0 && q < 1) {
            const bx = VL + b * 76 + 32, by = BT + 32, e = ease(q);
            p.style.transform = `translate(${f1(qb(AX, (AX + bx) / 2, bx, e))}px,${f1(qb(AY + 20, AY + 60, by, e))}px) rotate(${f1(q * 90)}deg)`;
            p.style.opacity = f2(Math.min(1, q * 6, (1 - q) * 6));
          }
        });
        if (pos >= pf - 250 && pos < pf - 140) { busy = Math.max(busy, .7); const b = ORG[c][clamp(Math.floor((pos - pf + 250) / 22), 0, 2)]; look = { x: VL + b * 76 + 32, y: BT + 32 }; }
        if (pos >= pf - 140 && pos < pf + 90) {
          const tx = NOW - pos + f * AP + AW / 2, ty = BOT + 15, ax = AX + 16, ay = AY - 30;
          feedP.setAttribute('d', `M${ax} ${ay}C${ax + 80} ${ay - 34} ${f1(tx - 70)} ${ty + 70} ${f1(tx)} ${ty}`);
          feedO = seg(pos, pf - 140, pf - 120) * (1 - seg(pos, pf, pf + 90));
          const q = seg(pos, pf - 130, pf);
          if (q > 0 && q < 1) {
            const L = feedP.getTotalLength(), pt = feedP.getPointAtLength(L * ease(q));
            PF.style.transform = `translate(${f1(pt.x)}px,${f1(pt.y)}px) scale(${f2(.6 + .4 * spring(seg(q, 0, .3)))})`;
            pfOn = true; look = { x: pt.x, y: pt.y };
          }
          if (pos < pf - 30) pp = seg(pos, pf - 140, pf - 30);
          if (pos >= pf - 132) hop = pf - 132;
        }
      });
      blocks.forEach((el, b) => {
        let on = false, hit = false;
        ORG.forEach((list, c) => {
          const k = list.indexOf(b); if (k < 0) return;
          const arr = (FEEDS[c] - 1) * AP - 250 + k * 22 + 90;
          if (pos >= arr) on = true;
          if (pos >= arr && pos < arr + 110) hit = true;
        });
        if (on !== el._on) { el._on = on; el.classList.toggle('on', on); }
        if (hit !== el._hit) { el._hit = hit; el.classList.toggle('hit', hit); }
      });
      feedP.style.opacity = f2(feedO); PF.style.opacity = pfOn ? 1 : 0;
      PR.style.opacity = f2(prO); PR.style.transform = `translateX(${f1(prX)}px)`;
      return { a, busy, pp, look, hop };
    }
    function wipe(e) {
      const x = -140 + e * (W + 280), l = f1(x - 80) + 'px', r = f1(x + 80) + 'px';
      ab.style.webkitMaskImage = ab.style.maskImage = `linear-gradient(90deg,transparent ${l},#000 ${r})`;
      real.style.webkitMaskImage = real.style.maskImage = `linear-gradient(90deg,#000 ${l},transparent ${r})`;
      glow.style.left = f1(x) + 'px'; glow.style.opacity = f2(Math.sin(Math.PI * e) * .9);
    }

    const real = $('.hz-real', st), world = $('.hz-world', real), dot = $('.hz-dot', real), bankEl = $('.hz-bank', real), filmEl = $('.hz-film', real);
    const items = $$('[data-k]', world).map(el => ({ el, k: el.dataset.k, x: +el.dataset.x, w: +(el.dataset.w || 0), s: null }));
    const P = $$('path', world).map(p => ({ p, x: +p.dataset.x, L: p.getTotalLength(), on: false }));
    const BASE = N * SW;
    let lastShot = -1, lastCommit = -99, prevBank = {};
    const tagOf = (adds, drops) => adds && drops > 2 ? 'rewritten' : adds && drops ? 'edited' : adds ? `+${adds} fact${adds > 1 ? 's' : ''}` : '';
    const NR = {};
    $$('.tn', real).forEach(el => NR[el.dataset.id] = { el, v: $('.v', el), inn: $('.in', el), tg: $('.tg', el) });
    const flyP = $('.tm-fly path', real), flyD = $('.tm-fly circle', real);
    let memJ = -99, memCur = null, MT = null;
    function fitNote(x) { const ov = x.inn.offsetHeight - x.v.clientHeight; x.inn.style.transform = ov > 0 ? `translateY(${-ov}px)` : ''; x.v.classList.toggle('ov', ov > 0); }
    function memSet(r, seq) {
      const nx = nextOf(r), A = noteOf(nx.i), B = seq && memCur ? memCur : noteOf(r.i);
      memCur = A;
      const rows = NOTE_IDS.map(id => {
        const a = A[id] || [], b = B[id] || [], x = NR[id];
        let p = 0; while (p < a.length && p < b.length && a[p] === b[p]) p++;
        const drop = b.slice(p), add = a.slice(p);
        const line = (t, c) => `<div class="it${c}"><u>${t}</u></div>`;
        x.inn.innerHTML = a.slice(0, p).map(t => line(esc(t), '')).join('') + drop.map(t => line(esc(t), ' x0')).join('') + add.map(() => line('', ' nw e')).join('');
        x.inn.style.transform = ''; fitNote(x);
        x.el.classList.remove('hot'); x.tg.textContent = ''; x.tg.classList.remove('on');
        const be = (nx.bank || []).find(e => e.id === id);
        x.v.title = be ? [be.desc, ...(be.facts || []).map(f => `${f.attr}: ${f.value}`)].filter(Boolean).join('\n') : '';
        return { x, id, drop: $$('.x0', x.inn), add: $$('.nw', x.inn), txt: add, n: add.reduce((s, t) => s + t.length, 0), tag: tagOf(add.length, drop.length) };
      });
      MT = { q: -1, rows, hit: rows.find(w => w.n || w.drop.length) };
    }
    function memTick(pos) {
      const jc = Math.floor((pos - COMMIT) / SW), d = pos - COMMIT - jc * SW;
      if (jc !== memJ) { const seq = jc === memJ + 1 && ((jc % N) + N) % N !== 0; memJ = jc; memSet(SEQ[((jc % N) + N) % N], seq); }
      if (MT.hit && d >= 0 && d < 95) {
        const sx = NARROW ? NOW - d : VL + NOWX - pos + jc * SW + US + 48, sy = NARROW ? VT + 146 : VT + 214;
        const tx = VL + NOTE_IDS.indexOf(MT.hit.id) * PP + PW + PG + NW / 2, ty = BT - 3, ly = BT - 7, dir = tx > sx ? 1 : -1;
        flyP.setAttribute('d', Math.abs(tx - sx) < 80 ? `M${f1(sx)} ${sy}C${f1(sx)} ${sy + 60} ${f1(tx)} ${ty - 50} ${f1(tx)} ${ty}`
          : `M${f1(sx)} ${sy}C${f1(sx)} ${ly - 4} ${f1(sx + 10 * dir)} ${ly} ${f1(sx + 30 * dir)} ${ly}L${f1(tx - 30 * dir)} ${ly}C${f1(tx - 10 * dir)} ${ly} ${f1(tx)} ${ly + 1} ${f1(tx)} ${ty}`);
        flyP.style.opacity = f2(1 - seg(d, 70, 95));
        const qd = seg(d, 0, 70);
        if (qd < 1) { const L = flyP.getTotalLength(), pt = flyP.getPointAtLength(L * ease(qd)); flyD.setAttribute('cx', f1(pt.x)); flyD.setAttribute('cy', f1(pt.y)); flyD.style.opacity = 1; }
        else flyD.style.opacity = 0;
      } else { flyP.style.opacity = 0; flyD.style.opacity = 0; }
      const q = clamp((d - 70) / USW, 0, 1);
      if (q === MT.q) return;
      MT.q = q;
      for (const w of MT.rows) {
        const busy = w.n || w.drop.length;
        if (!busy) continue;
        w.x.el.classList.toggle('hot', q > 0 && q < 1);
        if (q > .12) w.x.tg.textContent = w.tag;
        w.x.tg.classList.toggle('on', q > .12 && q < 1);
        w.drop.forEach(el => { el.classList.toggle('x', q > 0); el.classList.toggle('out', q > .3); if (q > .5 && el.isConnected) el.remove(); });
        if (w.n) {
          let n = Math.round(seg(q, w.drop.length ? .52 : .08, .95) * w.n);
          w.add.forEach((el, k) => { const t = w.txt[k], m = clamp(n, 0, t.length); if (el.textContent.length !== m) el.firstChild.textContent = t.slice(0, m); el.classList.toggle('e', m === 0); n -= t.length; });
        }
        fitNote(w.x);
      }
    }
    function setPrompt(r) {
      qn.textContent = 'Prompt ' + (r.i + 1);
      typewrite(qt, shotText(r));
      live.textContent = `streaming · shot ${r.i + 1}`;
    }
    function setBank(r, seq) {
      const bank = (nextOf(r).bank || []).slice(0, 8);
      bankEl.innerHTML = bank.map(b => {
        const up = seq && prevBank[b.id] !== b.version;
        return `<div class="e ${up ? 'up' : ''}" data-id="${esc(b.id)}" title="${esc(b.name)}"><img src="${b.img}" alt="">${up ? `<em>${prevBank[b.id] == null ? 'new' : 'v' + b.version}</em>` : ''}<span>${esc(nick(b))}</span></div>`;
      }).join('');
      prevBank = {}; bank.forEach(b => prevBank[b.id] = b.version);
      if (seq) agent.hop(clock);
    }
    function renderReal(pos) {
      world.style.transform = `translateX(${f1(NOWX - pos)}px)`;
      filmEl.style.backgroundPosition = `${f1(((NOWX - pos) % 14 + 14) % 14)}px 0,0 0`;
      const lo = pos - NOWX - 260, hi = pos + (W - VL - NOWX) + 60;
      for (const it of items) {
        if (it.x < lo || it.x > hi) continue;
        let s;
        if (it.k === 'fr') {
          const d = (it.x - pos) / FS;
          s = d <= 0 ? 'p' : d < 1 ? 'd' : 'w';
          if (s === 'd') { it.el.firstChild.style.filter = `blur(${f1(d * 7)}px)`; it.el.lastChild.style.opacity = f2(d * .85); }
          if (s !== it.s) { it.el.classList.toggle('pend', s === 'w'); if (s === 'p') { it.el.firstChild.style.filter = ''; it.el.lastChild.style.opacity = 0; } }
        } else if (it.k === 'step') {
          s = pos >= it.x + it.w ? 'done' : pos >= it.x ? 'on' : '';
          if (s !== it.s) { it.el.classList.toggle('done', s === 'done'); it.el.classList.toggle('on', s === 'on'); }
        } else {
          s = pos >= it.x ? 'on' : '';
          if (s !== it.s) it.el.classList.toggle(it.k === 'dia' ? 'done' : 'on', s === 'on');
        }
        it.s = s;
      }
      let reading = false;
      for (const it of items) if (it.k === 'step' && it.s === 'on' && it.el.textContent === 'read prompt') reading = true;
      if (reading !== qEl._r) { qEl._r = reading; qEl.classList.toggle('reading', reading); }
      let moving = null;
      for (const q of P) {
        const on = pos >= q.x;
        if (on !== q.on) { q.on = on; q.p.style.opacity = on ? 1 : 0; }
        if (on && pos < q.x + 120) moving = q;
      }
      if (moving) { const pt = moving.p.getPointAtLength(moving.L * (pos - moving.x) / 120); dot.setAttribute('cx', pt.x); dot.setAttribute('cy', pt.y); dot.style.opacity = 1; } else dot.style.opacity = 0;
      const j = Math.floor(pos / SW);
      if (j !== lastShot) { lastShot = j; setPrompt(SEQ[j % N]); }
      const jc = Math.floor((pos - COMMIT) / SW);
      if (jc !== lastCommit) { const seq = jc === lastCommit + 1; lastCommit = jc; setBank(SEQ[((jc % N) + N) % N], seq); }
      memTick(pos);
    }

    let clock = 0, last = null, running = false, phase = 0, realT0 = 0, lastHop = -1;
    const shown = CH.map(() => false);
    function step(t) {
      let n = 0;
      CH.forEach((c, i) => {
        const on = TT[i] <= t && t < TO[i];
        if (on !== shown[i]) { shown[i] = on; c.classList.toggle('on', on); }
        if (on) n = i + 1;
      });
      typed = n;
      if (!caret.classList.contains('gone')) {
        placeCaret();
        caret.classList.toggle('blink', !BUSY.some(([a, b]) => t > a - .02 && t < b + .08));
        if (t > T_END + 1.3) caret.classList.add('gone');
      }
      if (phase === 0) {
        centerTag(t);
        if (t < T_UP) return;
        phase = 1;
        heroEl.classList.add('abs'); heroEl.classList.remove('intro');
        tag.style.transform = '';
      }
      const ta = Math.max(0, t - T_AB);
      const m = phase < 3 ? renderAbs(ta) : null;
      if (m && m.hop >= 0 && m.hop !== lastHop) { lastHop = m.hop; agent.hop(t); }
      if (phase === 1 && ta >= TW) { phase = 2; realT0 = t; real.style.visibility = ''; nowEl.style.opacity = 1; }
      if (phase === 2) {
        const e = seg(ta, TW, TW + WIPE);
        wipe(ease(e));
        if (ta >= TW + .45) heroEl.classList.remove('abs');
        if (e >= 1) {
          phase = 3; ab.style.display = 'none'; glow.style.opacity = 0;
          real.style.webkitMaskImage = real.style.maskImage = '';
        }
      }
      if (phase >= 2) renderReal(BASE + (((t - realT0) / T + .25) * SW) % BASE);
      const dk = ease(seg(ta, TW + .05, TW + 1.1)), hop = 34 * Math.sin(Math.PI * dk);
      const sIn = .82 + .18 * spring(seg(ta, TA, TA + .8)), s = lerp(sIn, DOCK[2], dk);
      agent.draw(t, {
        x: lerp(AX - 32 * sIn, DOCK[0], dk), y: lerp(AY - 32 * sIn, DOCK[1], dk) - hop, s,
        a: m ? Math.max(m.a, dk) : 1, busy: m && dk === 0 ? m.busy : 0, pp: m && dk === 0 ? m.pp : -1, look: m && dk === 0 ? m.look : null
      });
    }
    function frame(now) {
      if (!running) return;
      if (last != null) clock += Math.min(.1, (now - last) / 1000);
      last = now;
      step(clock);
      requestAnimationFrame(frame);
    }
    function size() {
      reserve();
      fit(box, st, W, H, Math.max(200, window.innerHeight - tag.offsetHeight - qEl.offsetHeight - 150));
      if (phase === 0) { lastC = 0; placeCaret(); centerTag(clock); }
    }
    heroEl.classList.add('typing');
    size(); window.addEventListener('resize', size);
    if (document.fonts) document.fonts.ready.then(size);
    if (RM) {
      phase = 3; CH.forEach(c => c.classList.toggle('on', c.dataset.p !== 'x')); caret.classList.add('gone');
      heroEl.classList.remove('intro', 'abs'); ab.style.display = 'none'; real.style.visibility = '';
      lanes.forEach(l => l.style.opacity = 1); nowEl.style.opacity = 1;
      renderReal(BASE + 2.55 * SW);
      agent.draw(0, { x: DOCK[0], y: DOCK[1], s: DOCK[2], a: 1, busy: 0, pp: -1, look: null });
      return { pause() {}, resume() {} };
    }
    running = true; requestAnimationFrame(frame);
    return {
      pause() { running = false; },
      resume() { if (!running) { running = true; last = null; requestAnimationFrame(frame); } }
    };
  })();

  (function () {
    const inner = $('#heroInner'), nav = $('#nav'), heroEl = $('#hero'), hint = $('.hero-hint');
    let ticking = false;
    function onScroll() {
      const h = window.innerHeight, y = window.scrollY, p = clamp(y / (h * .9), 0, 1);
      inner.style.transform = `translateY(${-p * 22}vh) scale(${1 - .12 * p})`;
      inner.style.opacity = String(1 - p * 1.15);
      hint.style.opacity = String(Math.max(0, 1 - p * 3));
      nav.classList.toggle('on', y > h * .75);
      if (p >= 1) hero.pause(); else hero.resume();
      ticking = false;
    }
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
    heroEl.addEventListener('click', e => { e.preventDefault(); window.scrollTo({ top: $('#paper').offsetTop - 10, behavior: 'smooth' }); });
    onScroll();
  })();

  (function () {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
    $$('.reveal').forEach(el => io.observe(el));
  })();
  $$('[data-pending]').forEach(a => a.addEventListener('click', e => e.preventDefault()));

  (function helix() {
    const cv = $('#helix'); if (!cv) return;
    const ctx = cv.getContext('2d');
    let w = 0, h = 0, dpr = 1, vis = true;
    function size() { dpr = Math.min(2, window.devicePixelRatio || 1); w = cv.clientWidth; h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr; }
    size(); window.addEventListener('resize', size);
    const ANG = -36 * Math.PI / 180, NS = 13, TURNS = 1.25;
    const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    const SAND = [225, 182, 75], INK = [31, 33, 36], GREEN = rgb(css('--g') || '#5E9A12'), PAPER = rgb(css('--bg') || '#F4F3EE');
    const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`;
    function draw(now) {
      const t = now / 1000;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      ctx.translate(w / 2, h / 2); ctx.rotate(ANG);
      const L = w * 1.12, R = h * .2, k = TURNS * 2 * Math.PI / L, sp = L / (NS - 1);
      const pts = [];
      for (let i = 0; i < NS; i++) {
        const x = -L / 2 + i * sp, ph = k * (x + L / 2) + t * 1.15;
        pts.push({ i, x, ya: R * Math.sin(ph), za: Math.cos(ph), yb: -R * Math.sin(ph), zb: -Math.cos(ph), sl: R * k * Math.cos(ph) });
      }
      const lit = Math.floor((t * 2.2) % NS);
      pts.forEach(p => {
        ctx.beginPath(); ctx.moveTo(p.x, p.ya); ctx.lineTo(p.x, p.yb);
        ctx.strokeStyle = p.i === lit ? `rgba(${GREEN},.8)` : `rgba(${INK},.24)`; ctx.lineWidth = p.i === lit ? 1.6 : 1; ctx.stroke();
      });
      const items = [];
      pts.forEach(p => { items.push({ s: 'f', p, y: p.ya, z: p.za, sl: p.sl }); items.push({ s: 'm', p, y: p.yb, z: p.zb, sl: -p.sl }); });
      items.sort((a, b) => a.z - b.z);
      items.forEach(o => {
        const d = (o.z + 1) / 2, sc = .72 + .28 * d, fade = .55 * (1 - d);
        ctx.save(); ctx.translate(o.p.x, o.y); ctx.rotate(Math.atan(o.sl));
        if (o.s === 'f') {
          const fw = sp * .92, fh = 9.5 * sc * (w / 96);
          ctx.fillStyle = mix(INK, PAPER, fade); ctx.fillRect(-fw / 2, -fh / 2, fw, fh);
          ctx.fillStyle = mix(SAND, PAPER, fade); ctx.fillRect(-fw / 2 + 1.2, -fh * .26, fw - 2.4, fh * .52);
          ctx.fillStyle = mix(PAPER, PAPER, 0); const hs = Math.max(.9, fh * .1);
          ctx.fillRect(-hs / 2, -fh / 2 + fh * .07, hs, hs); ctx.fillRect(-hs / 2, fh / 2 - fh * .07 - hs, hs, hs);
        } else {
          const r = 3.3 * sc * (w / 96);
          ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fillStyle = mix(GREEN, PAPER, fade); ctx.fill();
          if (o.p.i % 3 === 0) { ctx.beginPath(); ctx.arc(0, 0, r * .38, 0, Math.PI * 2); ctx.fillStyle = mix(PAPER, GREEN, .1); ctx.fill(); }
        }
        ctx.restore();
      });
      if (!RM && vis) requestAnimationFrame(draw);
    }
    new IntersectionObserver(es => { const was = vis; vis = es[0].isIntersecting; if (vis && !was && !RM) requestAnimationFrame(draw); }).observe(cv);
    requestAnimationFrame(draw);
  })();

  const ICON = {
    play: '<svg class="i1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><svg class="i2" viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>',
    snd: '<svg class="i1" viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M16.5 9.5l5 5M21.5 9.5l-5 5" stroke="#fff" stroke-width="1.8" fill="none"/></svg><svg class="i2" viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" stroke="#fff" stroke-width="1.8" fill="none"/></svg>'
  };
  function setupPlayer(pane, video, onTime) {
    const bp = $('.vplay', pane), bs = $('.vsnd', pane);
    const upd = () => bp.classList.toggle('alt', !video.paused);
    bp.addEventListener('click', e => { e.stopPropagation(); video.paused ? video.play() : video.pause(); });
    if (bs) bs.addEventListener('click', e => { e.stopPropagation(); video.muted = !video.muted; bs.classList.toggle('alt', !video.muted); });
    $('.vwrap', pane).addEventListener('click', () => { video.paused ? video.play() : video.pause(); });
    video.addEventListener('play', upd); video.addEventListener('pause', upd);
    let raf = 0;
    const loop = () => { onTime(video.currentTime); raf = video.paused ? 0 : requestAnimationFrame(loop); };
    video.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(loop); });
    video.addEventListener('seeked', () => onTime(video.currentTime));
    video.addEventListener('timeupdate', () => { if (!raf) onTime(video.currentTime); });
    return { play() { const p = video.play(); if (p) p.catch(() => {}); }, pause() { video.pause(); } };
  }
  function segBar(pane, parts, onSeek) {
    const segs = $('.segs', pane);
    segs.innerHTML = parts.map((p, j) => `<button class="${p.cls || ''}" style="flex:${p.b - p.a}" data-j="${j}" aria-label="${esc(p.label || '')}"><span class="f"></span></button>`).join('');
    const bs = $$('button', segs);
    bs.forEach(b => b.addEventListener('click', () => onSeek(parts[+b.dataset.j].a + .05)));
    return t => bs.forEach((b, j) => {
      const p = parts[j], u = clamp((t - p.a) / (p.b - p.a), 0, 1);
      b.classList.toggle('done', u >= 1 && !p.keep); b.querySelector('.f').style.width = (u >= 1 && !p.keep ? '' : (u * 100) + '%');
    });
  }
  const seekPlay = v => t => { v.currentTime = t; if (v.paused) { const p = v.play(); if (p) p.catch(() => {}); } };

  function buildStory(d, pane, models, mode0) {
    const shots = d.shots, N = shots.length;
    let M = models.find(m => m.key === mode0) || models[0];
    pane.innerHTML = `
      <p class="hook">${esc(d.hook)}</p>
      <div class="dgrid">
        <div>
          <div class="vwrap"><video playsinline muted loop preload="none" poster="${M.poster}" src="${M.video}"></video>
            <div class="vctl"><button class="vsnd" aria-label="Sound on or off">${ICON.snd}</button><button class="vplay" aria-label="Play or pause">${ICON.play}</button></div></div>
          <div class="segs"></div>
          <div class="pline"><span class="pk">Prompt 1</span><div><span class="tx"></span><span class="nx"></span></div></div>
        </div>
        <aside class="mem">
          <h4>Memory <span class="mh"></span></h4>
          <div class="mbig"><img class="a" alt="memory image"><img class="b" alt="" style="opacity:0"></div>
          <div class="msmall"></div>
          <h4 class="s2">Entities <span class="en"></span></h4>
          <div class="egrid"></div>
        </aside>
      </div>`;
    const v = $('video', pane), pk = $('.pk', pane), tx = $('.tx', pane), nxl = $('.nx', pane);
    const mh = $('.mh', pane), A = $('.mbig .a', pane), Bm = $('.mbig .b', pane), small = $('.msmall', pane), grid = $('.egrid', pane), en = $('.en', pane), aside = $('.mem', pane);
    let front = A, k = -1, last = -1;
    const T = () => M.times;
    let seg = segBar(pane, T().map((r, j) => ({ a: r.t0, b: r.t1, label: 'Shot ' + (j + 1) })), seekPlay(v));
    function render(kk) {
      const r = shots[kk], seq = kk === last + 1; last = kk;
      pk.textContent = 'Prompt ' + (r.i + 1); typewrite(tx, shotText(r, 170));
      nxl.textContent = kk + 1 < N ? `Prompt ${r.i + 2} is hidden until shot ${r.i + 2} starts` : '';
      mh.textContent = r.wm >= 0 ? `written after prompt ${r.wm + 1}` : 'from the cast description';
      const used = r.used || [], big = used[used.length - 1];
      if (big && front.getAttribute('src') !== big.img) {
        const back = front === A ? Bm : A;
        back.src = big.img; back.style.opacity = 1; front.style.opacity = 0; front = back;
      }
      small.innerHTML = used.slice(0, -1).map(u => `<img src="${u.img}" alt="">`).join('');
      const bank = r.bank || [];
      en.textContent = bank.length ? bank.length + '' : '';
      grid.innerHTML = bank.slice(0, 9).map((b, q) => {
        const up = b.changed;
        return `<div class="ent ${up ? 'up' : ''} ${up && seq ? 'fresh' : ''} ${b.active ? '' : 'dim'}" data-q="${q}"><div class="ph"><img src="${b.img}" alt="${esc(b.name)}" loading="lazy"></div>${up ? `<em>${b.new ? 'new' : 'v' + b.version}</em>` : ''}<span>${esc(nick(b))}</span></div>`;
      }).join('') || '<div style="grid-column:1/-1;font-size:13px;color:var(--muted)">Empty until the first prompt.</div>';
      grid._bank = bank;
    }
    const tip = tipFor(aside);
    aside.addEventListener('mousemove', e => {
      const el = e.target.closest('.ent'); if (!el) { tip.style.opacity = 0; return; }
      const b = grid._bank[+el.dataset.q], rc = aside.getBoundingClientRect();
      const facts = (b.facts || []).map(f => `${esc(f.attr)}: ${esc(cut(f.value, 120))}`).join('<br>');
      tip.innerHTML = `<b>${esc(b.name)}</b> · v${b.version}${facts ? '<br>' + facts : ''}`;
      tip.style.left = Math.min(rc.width - 270, e.clientX - rc.left + 12) + 'px'; tip.style.top = (e.clientY - rc.top + 14) + 'px'; tip.style.opacity = 1;
    });
    aside.addEventListener('mouseleave', () => tip.style.opacity = 0);
    function onTime(t) {
      const tm = T();
      let kk = 0; while (kk + 1 < N && t >= tm[kk + 1].t0) kk++;
      if (kk !== k) { k = kk; render(kk); }
      seg(t);
    }
    const ctl = setupPlayer(pane, v, onTime);
    onTime(0);
    ctl.setModel = key => {
      const nm = models.find(m => m.key === key); if (!nm || nm === M) return;
      const tm = T(), t = v.currentTime; let kk = 0; while (kk + 1 < N && t >= tm[kk + 1].t0) kk++;
      const u = clamp((t - tm[kk].t0) / (tm[kk].t1 - tm[kk].t0), 0, .999);
      M = nm; seg = segBar(pane, T().map((r, j) => ({ a: r.t0, b: r.t1, label: 'Shot ' + (j + 1) })), seekPlay(v));
      const playing = !v.paused; v.poster = M.poster; v.src = M.video;
      v.addEventListener('loadedmetadata', function f() { v.removeEventListener('loadedmetadata', f); v.currentTime = T()[kk].t0 + u * (T()[kk].t1 - T()[kk].t0); if (playing) { const pr = v.play(); if (pr) pr.catch(() => {}); } });
      k = -1; onTime(T()[kk].t0);
    };
    return ctl;
  }

  const AV = {};
  const agentData = id => AV[id] || (AV[id] = fetch(`data/agent/${id}.json`).then(r => r.json()).catch(() => { delete AV[id]; return null; }));
  const BLOB = '<svg class="ablob" viewBox="0 0 64 64" aria-hidden="true"><path fill="#78BE3C" d="M32 9c13 0 23 9 24 22 1 12-4 22-24 22S7 43 8 31C9 18 19 9 32 9z"/><ellipse cx="25" cy="33" rx="4.4" ry="6" fill="#fff"/><ellipse cx="39" cy="33" rx="4.4" ry="6" fill="#fff"/></svg>';
  const ROLE = { G: 'the global image G', M: 'the shot memory image M' };
  function buildWall(box, stories, modelsOf, getMode) {
    const wall = $('.wall', box), insp = $('.insp', box), ibox = $('.ibox', insp);
    const WIDE = () => box.clientWidth >= 860 && matchMedia('(hover: hover) and (pointer: fine)').matches;
    let inView = false, open = null, hoverT = 0, leaveT = 0, lastXY = '';
    const timesOf = M => M.times;
    const shotAt = (tm, t) => { let k = 0; while (k + 1 < tm.length && t >= tm[k + 1].t0) k++; return k; };
    const tiles = stories.map(d => {
      const el = document.createElement('div');
      el.className = 'wtile'; el.tabIndex = 0; el.setAttribute('role', 'button');
      el.setAttribute('aria-label', `${d.title}. Open the agent behind this video.`);
      el.innerHTML = `<div class="wvid"><video muted playsinline loop preload="none"></video><span class="wbadge">${BLOB}<span>Agent view</span><i></i></span><span class="wcap"><span class="pk"></span><span class="tx"></span></span><span class="wpend"><b>Pending</b><span>Agentic LongLive (Offline) is still rendering this story.</span></span></div>
        <div class="segs"></div>
        <div class="wmeta"><b>${esc(d.title)}</b><span class="wl">${esc(d.hook || '')}</span></div>`;
      wall.appendChild(el);
      const T = { d, el, v: $('video', el), k: -1, pk: $('.pk', el), tx: $('.tx', el), seg: null, M: null };
      T.bS = document.createElement('button'); T.bS.className = 'wsnd'; T.bS.setAttribute('aria-label', 'Sound on or off');
      T.bS.innerHTML = ICON.snd; $('.wvid', el).appendChild(T.bS);
      T.bS.addEventListener('click', e => {
        e.stopPropagation(); if (!T.M || T.M.pending) return;
        const on = T.v.muted;
        tiles.forEach(x => { if (x !== T) { x.v.muted = true; x.bS.classList.remove('alt'); } });
        T.v.muted = !on; T.bS.classList.toggle('alt', on);
        if (on && T.v.paused) T.play();
      });
      T.setModel = key => {
        const ms = modelsOf(d), nm = ms.find(m => m.key === key) || ms.find(m => m.key === 'longlive_h3');
        if (nm === T.M) return;
        T.v.muted = true; T.bS.classList.remove('alt');
        let k = 0, u = 0;
        const tNow = T.parked ? T.parked.t : T.v.currentTime; T.parked = null;
        if (T.M) { const tm = timesOf(T.M), t = tNow; k = shotAt(tm, t); u = clamp((t - tm[k].t0) / (tm[k].t1 - tm[k].t0), 0, .999); }
        const playing = !T.v.paused; T.M = nm;
        el.classList.toggle('pend', !!nm.pending);
        if (nm.pending) { T.v.pause(); T.v.poster = ''; T.v.removeAttribute('src'); T.v.load(); k = 0; u = 0; }
        else { T.v.poster = nm.poster; T.v.src = nm.video; }
        T.seg = segBar(el, timesOf(nm).map((r, j) => ({ a: r.t0, b: r.t1, label: 'Shot ' + (j + 1) })), T.seek);
        if (k || u) T.v.addEventListener('loadedmetadata', function f() { T.v.removeEventListener('loadedmetadata', f); const tn = timesOf(nm), r = tn[Math.min(k, tn.length - 1)]; T.v.currentTime = r.t0 + u * (r.t1 - r.t0); if (playing) T.play(); });
        T.k = -1; T.tick();
      };
      T.tick = () => {
        const t = T.v.currentTime, k = shotAt(timesOf(T.M), t);
        if (k !== T.k) { T.k = k; T.pk.textContent = 'Prompt ' + (k + 1); T.tx.textContent = shotText(((T.M && T.M.shots) || d.shots)[k], 180); }
        T.seg(t);
      };
      T.park = () => { if (T.parked || !T.M || T.M.pending || !T.v.currentSrc) return; T.parked = { t: T.v.currentTime }; T.v.pause(); T.v.removeAttribute('src'); T.v.load(); };
      T.unpark = () => {
        if (!T.parked) return;
        const t = T.parked.t; T.parked = null; T.v.src = T.M.video;
        T.v.addEventListener('loadedmetadata', function f() { T.v.removeEventListener('loadedmetadata', f); T.v.currentTime = t; });
      };
      T.seek = t => {
        if (!T.M || !T.M.noAgent || T.M.pending) return;
        if (T.parked) T.parked.t = t;
        else if (T.v.readyState < 1) T.v.addEventListener('loadedmetadata', function f() { T.v.removeEventListener('loadedmetadata', f); T.v.currentTime = t; });
        else T.v.currentTime = t;
        T.play();
      };
      $('.segs', el).addEventListener('click', e => { if (T.M && T.M.noAgent) e.stopPropagation(); });
      T.play = () => { if (T.M && T.M.pending) return; if (T.parked) T.unpark(); if (T.v.preload === 'none') T.v.preload = 'auto'; const p = T.v.play(); if (p) p.catch(() => {}); };
      T.v.addEventListener('timeupdate', T.tick); T.v.addEventListener('seeked', T.tick);
      T.v.addEventListener('play', () => { if (open === T) { bPlay.classList.add('alt'); if (!raf) raf = requestAnimationFrame(loop); } });
      T.v.addEventListener('pause', () => { if (open === T) { bPlay.classList.remove('alt'); onTime(T.v.currentTime); } });
      T.v.addEventListener('seeked', () => { if (open === T) onTime(T.v.currentTime); });
      T.setModel(getMode());
      el.addEventListener('mousemove', e => {
        const xy = e.clientX + ',' + e.clientY; if (xy === lastXY) return; lastXY = xy;
        if (open || !WIDE() || T.M.noAgent || el.classList.contains('arm')) return;
        el.classList.add('arm'); clearTimeout(hoverT); hoverT = setTimeout(() => show(T), 480);
      });
      el.addEventListener('mouseleave', () => { el.classList.remove('arm'); clearTimeout(hoverT); });
      el.addEventListener('click', () => { clearTimeout(hoverT); if (T.M.pending) return; if (T.M.noAgent) { T.v.paused ? T.play() : T.v.pause(); } else show(T); });
      el.addEventListener('keydown', e => { if ((e.key === 'Enter' || e.key === ' ') && !T.M.noAgent) { e.preventDefault(); show(T); } });
      return T;
    });
    const playTiles = () => tiles.forEach(T => { if (T.vis && !RM) T.play(); else T.v.pause(); }), pauseTiles = () => tiles.forEach(T => T.v.pause());
    const tio = new IntersectionObserver(es => es.forEach(e => {
      const T = tiles.find(x => x.el === e.target); T.vis = e.isIntersecting;
      if (open || RM) return;
      if (T.vis) T.play(); else T.v.pause();
    }), { threshold: .45 });
    tiles.forEach(T => tio.observe(T.el));
    new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { o.disconnect(); stories.forEach(d => { agentData(d.id); Object.values(d.models || {}).forEach(m => m.agent && agentData(m.agent)); }); } }, { rootMargin: '900px 0px' }).observe(box);
    ibox.innerHTML = `
      <button class="iclose" aria-label="Close the agent view"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
      <div class="ileft">
        <div class="vwrap"><div class="vctl"><button class="vsnd" aria-label="Sound on or off">${ICON.snd}</button><button class="vplay" aria-label="Play or pause">${ICON.play}</button></div></div>
        <div class="segs"></div>
        <div class="ifade">
          <p class="ihead"><b class="ititle"></b> <span class="ihook"></span></p>
          <div class="pline"><span class="pk"></span><div><span class="tx"></span></div></div>
          <div class="imem">
            <h5>In the generator for this shot <span class="mh"></span><span class="inext" hidden><em>next shot</em><span class="inx"></span></span></h5>
            <div class="iused"></div>
            <h5>Entity memory <span class="en"></span></h5>
            <div class="ibank"></div>
          </div>
        </div>
      </div>
      <div class="acon ifade">
        <div class="ahead">${BLOB}<div><b>Agent</b><span class="atask"></span></div><span class="aclock"></span></div>
        <div class="alanes"><div data-l="plan"><i></i>Planner<span></span></div><div data-l="draw"><i></i>Painter<span></span></div><div data-l="check"><i></i>Checker<span></span></div></div>
        <div class="alog" tabindex="0" aria-live="off"></div>
        <p class="afoot"></p>
      </div>`;
    const ivWrap = $('.vwrap', ibox), bPlay = $('.vplay', ibox), bSnd = $('.vsnd', ibox), log = $('.alog', ibox), clock = $('.aclock', ibox), task = $('.atask', ibox);
    const lanes = { plan: $('[data-l=plan]', ibox), draw: $('[data-l=draw]', ibox), check: $('[data-l=check]', ibox) };
    let S = null;
    const iseg = () => segBar(ibox, timesOf(S.M).map((r, j) => ({ a: r.t0, b: r.t1, label: 'Shot ' + (j + 1) })), t => seekPlay(S.v)(t));
    let segFn = () => {};
    function memHTML(r) {
      const used = r.used || [];
      const roles = used.map((u, j) => u.role === 'G' ? 'G' : (used.filter(x => x.role === 'M').length > 1 && j === used.length - 2 ? 'M, older' : 'M, newest'));
      return used.map((u, j) => `<figure class="${u.role === 'G' ? 'ug' : 'um'}"><img src="${u.img}" alt="${u.role} memory image"><figcaption>${roles[j]}</figcaption></figure>`).join('')
        || '<p class="none">Only the cast description so far.</p>';
    }
    function renderShot(k) {
      const d = S.d, r = d.shots[k], n = k + 1, seq = S.live && S.k >= 0 && S.k === k - 1;
      $('.pk', ibox).textContent = 'Prompt ' + n; typewrite($('.tx', ibox), shotText(r, 200));
      $('.mh', ibox).textContent = r.wm >= 0 ? `written after prompt ${r.wm + 1}` : 'from the cast description';
      $('.iused', ibox).innerHTML = memHTML(r);
      const bank = r.bank || [];
      $('.en', ibox).textContent = bank.length ? bank.length + '' : '';
      $('.ibank', ibox).innerHTML = bank.map(b => `<div class="ie ${b.changed ? 'up' : ''} ${b.active ? '' : 'dim'}" data-n="${esc(b.name)}" title="${esc(b.name + (b.facts || []).map(f => '\n' + f.attr + ': ' + f.value).join(''))}"><img src="${b.img}" alt="" loading="lazy"><span>${esc(nick(b))}</span></div>`).join('');
      S.nx = {}; const nxs = $('.inext', ibox); nxs.hidden = true; nxs.style.opacity = ''; $('.inx', nxs).innerHTML = '';
      if (seq && (r.used || []).length) requestAnimationFrame(() => $$('.iused img', ibox).forEach(flash));
      task.textContent = k + 1 < d.shots.length ? `reads prompt ${n}, prepares the memory for shot ${n + 1}` : `reads prompt ${n}, the last one`;
      S.k = k; S.seen = -1; S.rows = {}; log.innerHTML = ''; log.scrollTop = 0; S.stick = true;
      const A = S.ag && S.ag.shots[k];
      S.ev = A ? [{ t: 0, k: 'read' }, { t: A.plan.lat, k: 'plan' }].concat(A.ev.map(e => Object.assign({}, e, { t: Math.max(e.t, A.plan.lat + .01) }))).sort((a, b) => a.t - b.t) : [];
      S.span = A ? Math.max(S.ag.slot, A.span + .4) : 1;
      if (!A) log.innerHTML = `<p class="none">${S.ag ? 'No agent record for this shot.' : 'Loading the agent log…'}</p>`;
    }
    const verdict = e => `<span class="avd ${e.ok ? 'aok' : 'ano'}">${e.ok ? 'passed' : 'rejected'} · score ${Math.round(e.score)}/10</span>` +
      (e.issues && e.issues.length ? `<ul class="ais">${e.issues.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '');
    function evHTML(e) {
      const A = S.ag.shots[S.k], n = S.k + 1, P = A.plan;
      const at = `<span class="at">+${e.t.toFixed(1)} s</span>`;
      if (e.k === 'read') return `<div class="ae read">${at}<div><b>Reads prompt ${n}.</b> ${n < S.d.shots.length ? `Prompt ${n + 1} is still hidden, so it plans from what it has seen.` : ''}</div></div>`;
      if (e.k === 'plan') {
        const ch = P.changes.map(c => c.k === 'revoked' ? `<li><em class="tg rv">removed</em><b>${esc(c.name)}</b></li>`
          : `<li><em class="tg ${c.k === 'new' ? 'nw' : 'cg'}">${c.k}</em><b>${esc(c.name)}</b>${c.kind ? ` <span class="ak">${esc(c.kind)}</span>` : ''}${c.facts.length ? `<span class="af">${c.facts.map(esc).join('<br>')}</span>` : c.desc ? `<span class="af">${esc(c.desc)}</span>` : ''}</li>`).join('');
        const chips = a => a.map(x => `<span class="chip">${esc(x)}</span>`).join('');
        return `<div class="ae plan">${at}<div><b>Planner: updates the story state</b> <span class="al">${P.lat.toFixed(1)} s</span>
          ${ch ? `<ul class="ach">${ch}</ul>` : '<p class="ap">No new facts in this prompt.</p>'}
          <p class="ap">Scene <span class="chip">${esc(P.scene)}</span>${P.retire.length ? ` · leaves the story ${chips(P.retire)}` : ''}</p>
          <p class="ap"><b>Plans G</b> ${chips(P.g)}${P.gcap ? `<span class="acap">${esc(P.gcap)}</span>` : ''}</p>
          <p class="ap"><b>Plans M</b> ${chips(P.m)}${P.mcap ? `<span class="acap">${esc(P.mcap)}</span>` : ''}</p>
          ${P.redraw.length ? `<p class="ap"><b>Asks for new entity images</b> ${chips(P.redraw)}</p>` : ''}
          <details><summary>Planner output (JSON)</summary><pre>${esc(A.raw)}</pre></details></div></div>`;
      }
      if (e.k === 'draw') {
        const what = e.role === 'entity' ? (e.edit ? `Painter: edits the image of ${esc(e.ent)}` : `Painter: draws ${esc(e.ent)}`) + (e.view && e.view !== 'front' ? ` (${esc(e.view)} view)` : '')
          : `Painter: ${e.edit ? 'composes' : 'draws'} ${ROLE[e.role]}${e.edit ? ' from the entity images' : ''}`;
        return `<div class="ae draw">${at}<div class="adr"><img src="${e.img}" alt="" loading="lazy"><div><b>${what}</b>${e.round !== '' ? ` <span class="al">round ${e.round}</span>` : ''} <span class="al">${e.lat.toFixed(1)} s</span>
          <p class="apr" title="Click to expand">${esc(e.prompt)}</p><div class="av"><span class="apend">waiting for the checker…</span></div></div></div></div>`;
      }
      if (e.k === 'check') return `<div class="ae check">${at}<div class="adr"><img src="${e.img}" alt="" loading="lazy"><div><b>Checker: ${e.role === 'entity' ? 'checks an entity image' : 'checks ' + e.role}</b> <span class="al">${e.lat.toFixed(1)} s</span><div class="av">${verdict(e)}</div></div></div></div>`;
      if (e.k === 'commit') return `<div class="ae commit">${at}<div><b>Commits round ${e.round}.</b> ${S.k + 1 < S.d.shots.length ? `Shot ${n + 1} will start from this memory.` : ''}</div></div>`;
      if (e.k === 'refine') return `<div class="ae note">${at}<div>Refines ${e.roles.join(' and ')}, round ${e.round}. A redraw replaces the committed image only if it scores higher.</div></div>`;
      if (e.k === 'stop') return `<div class="ae note">${at}<div>${/newer prompt/.test(e.why) ? `Prompt ${n + 1} has arrived, so it stops refining and moves on.` : esc(e.why)}</div></div>`;
      if (e.k === 'kept') return `<div class="ae note">${at}<div>No redraw of ${esc(e.ent)} passed, so entity memory keeps the previous image.</div></div>`;
      if (e.k === 'painted') return `<div class="ae note aok">${at}<div>Entity memory now holds the new image of ${esc(e.ent)}.</div></div>`;
      if (e.k === 'retry') return `<div class="ae note">${at}<div>Tries ${esc(e.ent)} again.</div></div>`;
      return '';
    }
    function flash(el) { if (!el) return; el.classList.remove('mflash'); void el.offsetWidth; el.classList.add('mflash'); }
    function fly(from, to, done) {
      if (!S.live || RM || !from || !to || !ibox.getBoundingClientRect().width) { done(); return; }
      const ir = ibox.getBoundingClientRect(), lr = log.getBoundingClientRect(), fr = from.getBoundingClientRect(), tr = to.getBoundingClientRect();
      const x0 = fr.left - ir.left + 22, y0 = clamp(fr.top + 14, lr.top + 12, lr.bottom - 12) - ir.top;
      const x1 = tr.left - ir.left + Math.min(tr.width / 2, 40), y1 = tr.top - ir.top + tr.height / 2;
      const mx = (x0 + x1) / 2, my = Math.min(y0, y1) - 46, dot = document.createElement('i');
      dot.className = 'mdot'; ibox.appendChild(dot);
      const an = dot.animate([{ transform: `translate(${x0}px,${y0}px) scale(.5)`, opacity: 0 }, { transform: `translate(${mx}px,${my}px) scale(1)`, opacity: 1, offset: .5 },
        { transform: `translate(${x1}px,${y1}px) scale(.8)`, opacity: 1 }], { duration: 760, easing: 'cubic-bezier(.4,0,.25,1)' });
      an.onfinish = an.oncancel = () => { dot.remove(); if (S) done(); };
    }
    function memFx(e, row) {
      const k = S.k, nx = S.d.shots[k + 1];
      if (e.k === 'painted') {
        const nb = nx && (nx.bank || []).find(b => b.name === e.ent);
        const dr = S.ev.filter(x => x.k === 'draw' && x.role === 'entity' && x.ent === e.ent && x.t <= e.t);
        const pick = dr.find(x => x.view === 'front' || x.view === 'plate') || dr[dr.length - 1], img = (nb && nb.img) || (pick && pick.img);
        if (!img) return;
        const bankEl = $('.ibank', ibox), none = $('.none', bankEl);
        let tile = $$('.ie', bankEl).find(x => x.dataset.n === e.ent);
        if (!tile) {
          if (none) none.remove();
          bankEl.insertAdjacentHTML('beforeend', `<div class="ie" data-n="${esc(e.ent)}" style="opacity:0"><img src="${img}" alt=""><span>${esc(nb ? nick(nb) : e.ent)}</span></div>`);
          tile = bankEl.lastElementChild; $('.en', ibox).textContent = $$('.ie', bankEl).length + '';
        }
        fly(row, tile, () => { const im = $('img', tile); im.src = img; tile.style.opacity = ''; tile.classList.add('up'); tile.classList.remove('dim'); flash(im); });
      } else if (e.k === 'commit' && nx) {
        const dr = S.ev.filter(x => x.k === 'draw' && (x.role === 'G' || x.role === 'M') && String(x.round) === String(e.round) && x.t <= e.t);
        ['G', 'M'].forEach(r => { const x = dr.filter(y => y.role === r).pop(); if (x) S.nx[r] = x.img; });
        let imgs = [S.nx.G, S.nx.M].filter(Boolean);
        if (!imgs.length) imgs = (nx.used || []).map(u => u.img);
        const nxs = $('.inext', ibox), go = !$('.inx', nxs).childElementCount;
        nxs.hidden = false; if (go) nxs.style.opacity = 0;
        fly(row, nxs, () => { $('.inx', nxs).innerHTML = imgs.map(s => `<img src="${s}" alt="">`).join(''); nxs.style.opacity = ''; flash(nxs); });
      }
    }
    function showUpTo(a) {
      if (!S.ag || !S.ag.shots[S.k]) return;
      let added = false;
      S.ev.forEach((e, j) => {
        if (j <= S.seen || e.t > a) return;
        S.seen = j; added = true;
        if (e.k === 'check') {
          const row = S.rows[e.img];
          if (row && !row.dataset.v) { row.dataset.v = 1; const av = $('.av', row); av.innerHTML = verdict(e); av.classList.add('ain'); return; }
        }
        log.insertAdjacentHTML('beforeend', evHTML(e));
        const row = log.lastElementChild; if (e.k === 'draw') S.rows[e.img] = row;
        if (e.k === 'painted' || e.k === 'commit') memFx(e, row);
      });
      if (added && S.stick) log.scrollTo({ top: log.scrollHeight, behavior: RM ? 'auto' : 'smooth' });
    }
    log.addEventListener('scroll', () => { if (S) S.stick = log.scrollTop + log.clientHeight > log.scrollHeight - 40; });
    log.addEventListener('click', e => { const p = e.target.closest('.apr'); if (p) p.classList.toggle('full'); });
    function lanesAt(a, live) {
      const A = S.ag && S.ag.shots[S.k];
      const busy = { plan: live && A && a < A.plan.lat ? 'reading the prompts' : '', draw: '', check: '' };
      if (A && live) A.ev.forEach(e => {
        if ((e.k === 'draw' || e.k === 'check') && e.t - e.lat <= a && a < e.t) busy[e.k] = (e.k === 'draw' ? 'drawing ' : 'checking ') + (e.role === 'entity' ? (e.ent || 'an entity') : e.role);
      });
      for (const l in lanes) { lanes[l].classList.toggle('on', !!busy[l]); $('span', lanes[l]).textContent = busy[l] || 'idle'; }
    }
    function onTime(t) {
      if (!S) return;
      const tm = timesOf(S.M), k = shotAt(tm, t);
      S.live = !S.v.paused && !RM;
      if (k !== S.k) renderShot(k);
      segFn(t);
      const r = tm[k], live = !S.v.paused && !RM;
      const a = live ? clamp((t - r.t0) / (r.t1 - r.t0), 0, 1) * S.span : Infinity;
      S.a = Math.max(S.a || 0, a);
      if (S.kA !== k) { S.kA = k; S.a = a; }
      showUpTo(S.a);
      clock.textContent = S.ag && S.ag.shots[k] ? (isFinite(S.a) ? `agent clock +${Math.min(S.a, S.span).toFixed(1)} s` : 'whole log') : '';
      lanesAt(a, live);
    }
    let raf = 0;
    const loop = () => { if (!S) { raf = 0; return; } onTime(S.v.currentTime); raf = S.v.paused ? 0 : requestAnimationFrame(loop); };
    const vplay = v => { const p = v.play(); if (p) p.catch(() => {}); };
    bPlay.addEventListener('click', e => { e.stopPropagation(); if (S) S.v.paused ? vplay(S.v) : S.v.pause(); });
    const NOSND = M => M && M.key === 'longlive_h3';
    function bindModel() {
      const T = S.T, M = S.M, id = (M && M.agent) || T.d.id;
      S.d = M && M.shots ? Object.assign({}, T.d, { shots: M.shots }) : T.d;
      if (NOSND(M)) S.v.muted = true;
      ibox.classList.toggle('nosnd', NOSND(M));
      bSnd.classList.toggle('alt', !S.v.muted);
      if (S.agId === id) return;
      S.agId = id; S.ag = null; $('.afoot', ibox).innerHTML = '';
      agentData(id).then(ag => {
        if (!S || S.T !== T || S.agId !== id) return;
        S.ag = ag;
        if (S.ag) $('.afoot', ibox).innerHTML = `Recorded in the live run that wrote this memory: <b>${esc(S.ag.models.plan)}</b>, <b>${esc(S.ag.models.image)}</b> as painter and <b>${esc(S.ag.models.check)}</b> as checker, on GPUs of their own. Times are the agent’s clock after the prompt arrived. That run gave each prompt ${S.ag.slot.toFixed(0)}&#8239;s, so the log is replayed faster to fit each 8&#8239;s shot. Pause the video to see the whole log.`;
        S.k = -1; onTime(S.v.currentTime);
      });
    }
    bSnd.addEventListener('click', e => { e.stopPropagation(); if (!S || NOSND(S.M)) return; S.v.muted = !S.v.muted; bSnd.classList.toggle('alt', !S.v.muted); });
    ivWrap.addEventListener('click', () => { if (S) S.v.paused ? vplay(S.v) : S.v.pause(); });
    function place() {
      const wide = WIDE();
      insp.classList.toggle('sheet', !wide);
      const pitch = tiles[2] ? tiles[2].el.offsetTop - tiles[0].el.offsetTop : wall.offsetHeight;
      const H = Math.max(Math.min(Math.max(wall.offsetHeight, 660), Math.max(660, 2 * pitch - 22)), Math.min(innerHeight - 76, 880));
      ibox.style.height = wide ? H + 'px' : '';
      if (wide && open) {
        const wr = wall.getBoundingClientRect(), tr = open.el.getBoundingClientRect();
        const want = clamp(tr.top, 64, Math.max(64, innerHeight - H - 12));
        insp.style.top = (want - wr.top) + 'px';
      } else insp.style.top = '';
      document.documentElement.classList.toggle('noscroll', !wide && !!open);
    }
    function show(T) {
      if (open === T) return;
      if (open) hide(true);
      open = T; clearTimeout(leaveT);
      tiles.forEach(x => x.el.classList.remove('arm'));
      const playing = inView && !RM;
      S = { T, v: T.v, M: T.M, k: -1, kA: -1, a: 0, ag: null };
      $('.ititle', ibox).textContent = T.d.title; $('.ihook', ibox).textContent = T.d.hook;
      $('.afoot', ibox).innerHTML = '';
      insp.classList.add('open'); insp.setAttribute('aria-hidden', 'false'); place();
      tiles.forEach(x => { if (x !== T) { x.v.pause(); if (x.v.networkState === 2) x.park(); } });
      ivWrap.insertBefore(T.v, ivWrap.firstChild);
      bindModel(); bPlay.classList.toggle('alt', !T.v.paused);
      if ((playing || !WIDE()) && T.v.paused && !RM) T.play();
      if (!T.v.paused && !raf) raf = requestAnimationFrame(loop);
      const t0 = T.v.currentTime;
      segFn = iseg();
      onTime(t0);
      if (WIDE() && !RM) {
        const bx = box.getBoundingClientRect(), tr = $('.wvid', T.el).getBoundingClientRect();
        ibox.style.transition = 'none'; ibox.style.transform = 'none';
        const vr = $('.vwrap', ibox).getBoundingClientRect(), ir = ibox.getBoundingClientRect();
        const s = tr.width / vr.width;
        const tx = (tr.left - bx.left) - (ir.left - bx.left) - s * (vr.left - ir.left), ty = (tr.top - bx.top) - (ir.top - bx.top) - s * (vr.top - ir.top);
        insp.classList.add('enter');
        ibox.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
        void ibox.offsetWidth;
        ibox.style.transition = ''; ibox.style.transform = '';
        requestAnimationFrame(() => insp.classList.remove('enter'));
      }
      if (!WIDE()) $('.iclose', ibox).focus({ preventScroll: true });
    }
    function hide(now) {
      if (!open) return;
      const T = open; open = null; clearTimeout(leaveT);
      const playing = !T.v.paused;
      const done = () => {
        $('.wvid', T.el).insertBefore(T.v, $('.wvid', T.el).firstChild); T.v.muted = true;
        insp.classList.remove('open', 'leave'); insp.setAttribute('aria-hidden', 'true'); ibox.style.transform = ''; place();
        if ((playing || inView) && !RM) playTiles();
      };
      if (!now && WIDE() && !RM) {
        const bx = box.getBoundingClientRect(), tr = $('.wvid', T.el).getBoundingClientRect(), vr = $('.vwrap', ibox).getBoundingClientRect(), ir = ibox.getBoundingClientRect();
        const s = tr.width / vr.width;
        const tx = (tr.left - bx.left) - (ir.left - bx.left) - s * (vr.left - ir.left), ty = (tr.top - bx.top) - (ir.top - bx.top) - s * (vr.top - ir.top);
        insp.classList.add('leave'); ibox.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
        setTimeout(done, 380);
      } else done();
      S = null;
      if (document.activeElement && ibox.contains(document.activeElement)) T.el.focus({ preventScroll: true });
    }
    insp.addEventListener('mouseleave', () => { if (WIDE()) { clearTimeout(leaveT); leaveT = setTimeout(() => hide(), 260); } });
    insp.addEventListener('mouseenter', () => clearTimeout(leaveT));
    $('.iclose', ibox).addEventListener('click', e => { e.stopPropagation(); hide(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && open) hide(); });
    addEventListener('resize', () => { if (open) place(); });
    new IntersectionObserver(es => {
      inView = es[0].isIntersecting;
      if (inView && api.onView) api.onView();
      if (inView && !RM && !open) playTiles();
      else if (!inView) pauseTiles();
    }, { threshold: .2 }).observe(wall);
    const api = {
      setModel(key) {
        tiles.forEach(T => T.setModel(key));
        box.classList.toggle('off', tiles.every(T => T.M.noAgent));
        box.classList.toggle('pend', tiles.every(T => T.M.pending));
        if (inView && !open && !RM) playTiles();
        tiles.forEach(T => T.el.setAttribute('aria-label', T.M.pending ? `${T.d.title}, Agentic LongLive (Offline), pending` : T.M.noAgent ? `${T.d.title}, Agentic LongLive (Offline)` : `${T.d.title}. Open the agent behind this video.`));
        if (open && open.M.noAgent) hide(true);
        else if (open) { S.M = open.M; bindModel(); segFn = iseg(); S.k = -1; S.kA = -1; onTime(S.v.currentTime); }
      },
      inView: () => inView,
      ctl: { play() { if (!open) playTiles(); else open.play(); }, pause() { pauseTiles(); } }
    };
    return api;
  }

  function swapSrc(v, src) {
    const t = v.currentTime, playing = !v.paused;
    v.src = src;
    v.addEventListener('loadedmetadata', function f() { v.removeEventListener('loadedmetadata', f); v.currentTime = t; if (playing) { const p = v.play(); if (p) p.catch(() => {}); } });
  }
  function toggleUI(pane, onPick) {
    const bs = $$('.vtoggle button', pane);
    bs.forEach(b => b.addEventListener('click', e => { e.stopPropagation(); if (b.getAttribute('aria-pressed') === 'true') return; bs.forEach(x => x.setAttribute('aria-pressed', String(x === b))); onPick(b.dataset.v); }));
  }

  function buildRobot(R, pane) {
    const parts = [{ a: 0, b: R.close_start, label: 'overview' }, { a: R.close_start, b: R.obs_end, label: 'close-up' }, { a: R.obs_end, b: R.duration, label: 'generated', cls: 'gen' }];
    pane.innerHTML = `
      <p class="hook">The wrist camera stays on the ${esc(R.object)} for longer than the model’s context window. When it moves back, the rest of the scene has to come back too.</p>
      <div class="dgrid">
        <div>
          <div class="vwrap"><video playsinline muted loop preload="none" poster="${R.poster}" src="${R.video_ours}"></video>
            <div class="vtoggle"><button data-v="ours" aria-pressed="true">with memory</button><button data-v="native" aria-pressed="false">original</button></div>
            <div class="vctl"><button class="vplay" aria-label="Play or pause">${ICON.play}</button></div></div>
          <div class="segs"></div><div class="seglab">${parts.map(p => `<span style="flex:${p.b - p.a}">${p.label}</span>`).join('')}</div>
          <div class="pline"><span class="pk">Task</span><div><span class="tx">${esc(R.task)}</span></div></div>
        </div>
        <aside class="mem">
          <h4>Memory</h4>
          <div class="pair"><figure><div class="ph g"><img src="${R.mem_scene}" alt="scene memory"></div><figcaption>scene</figcaption></figure>
            <figure><div class="ph g"><img src="${R.mem_object}" alt="object memory"></div><figcaption>${esc(R.object)}</figcaption></figure></div>
          <h4 class="s2">Context window</h4>
          <div class="ctxwin"></div><div class="ctxcap"><span>oldest</span><span>newest</span></div>
          <div class="status"></div>
        </aside>
      </div>`;
    const v = $('video', pane), win = $('.ctxwin', pane), status = $('.status', pane), tx = $('.tx', pane);
    let mode = 'ours';
    toggleUI(pane, m => { mode = m; swapSrc(v, m === 'ours' ? R.video_ours : R.video_native); onTime(v.currentTime); });
    const seg = segBar(pane, parts.map(p => Object.assign({ keep: true }, p)), seekPlay(v));
    const BLK = 32 / 24;
    function onTime(t) {
      seg(t);
      const gen = t >= R.obs_end;
      tx.textContent = gen ? R.rec_prompt : R.task;
      const nObs = Math.min(10, Math.floor(t / BLK) + 1), nGen = gen ? Math.min(4, Math.floor((t - R.obs_end) / BLK) + 1) : 0;
      const blocks = [];
      for (let j = 0; j < nObs; j++) blocks.push(j < 2 ? 'scene' : 'close');
      for (let j = 0; j < nGen; j++) blocks.push('gen');
      const cells = mode === 'ours' ? ['S', 'O'].concat(blocks.slice(-6)) : blocks.slice(-8);
      while (cells.length < 8) cells.push('');
      win.innerHTML = cells.map(c => c === 'S' ? `<div class="c sink"><img src="${R.mem_scene}" alt=""></div>` : c === 'O' ? `<div class="c sink"><img src="${R.mem_object}" alt=""></div>`
        : c === 'scene' ? `<div class="c"><img src="${R.mem_scene}" alt=""></div>` : c === 'close' ? `<div class="c"><img src="media/apps/robot_close.webp" alt=""></div>`
        : c === 'gen' ? '<div class="c gen"></div>' : '<div class="c"></div>').join('');
      const sceneIn = mode === 'ours' || cells.includes('scene');
      status.textContent = !gen ? (mode === 'ours' ? 'The scene and the kettle stay in the window as memory.' : sceneIn ? 'The overview is still in the window.' : 'The overview has left the window.')
        : (mode === 'ours' ? 'The camera moves back, and the table and bowl return from memory.' : 'The camera moves back, but the model has only close-ups and turns to a different scene.');
    }
    const ctl = setupPlayer(pane, v, onTime);
    onTime(0);
    return ctl;
  }

  function buildWorld(Wd, pane) {
    const its = Wd.its;
    pane.innerHTML = `
      <p class="hook">A world model keeps views, not what the user changed. The agent keeps a ledger of changes and edits stored views, so a change behind the camera is there when it turns around.</p>
      <div class="dgrid">
        <div>
          <div class="vwrap"><video playsinline muted loop preload="none" poster="${Wd.poster}" src="${Wd.video_ours}"></video>
            <div class="vtoggle"><button data-v="ours" aria-pressed="true">with our agent</button><button data-v="native" aria-pressed="false">original</button></div>
            <div class="vctl"><button class="vplay" aria-label="Play or pause">${ICON.play}</button></div></div>
          <div class="segs"></div>
          <div class="pline"><span class="pk">User</span><div><span class="tx">looking around the meadow</span></div></div>
        </div>
        <aside class="mem">
          <h4>Memory</h4>
          <div class="ledger">${Wd.ledger.map((l, j) => `<div class="${j === 2 ? 'bh' : ''}">${esc(l[1])} <i>${j === 2 ? 'behind the camera' : 'in view'}</i></div>`).join('')}</div>
          <h4 class="s2">Edited view</h4>
          <div class="pair"><figure><div class="ph"><img class="eb" alt="stored view"></div><figcaption>stored</figcaption></figure>
            <figure><div class="ph g"><img class="ee" alt="edited view"></div><figcaption>written back</figcaption></figure></div>
          <div class="status"></div>
        </aside>
      </div>`;
    const v = $('video', pane), tx = $('.tx', pane), lis = $$('.ledger div', pane), eb = $('.eb', pane), ee = $('.ee', pane), status = $('.status', pane);
    let mode = 'ours', k = -1;
    toggleUI(pane, m => { mode = m; swapSrc(v, m === 'ours' ? Wd.video_ours : Wd.video_native); k = -1; onTime(v.currentTime); });
    const seg = segBar(pane, its.map(r => ({ a: r.t0, b: r.t1, label: r.user || r.act, cls: r.user ? 'gen' : '' })), seekPlay(v));
    const writes = its.filter(r => r.write);
    function onTime(t) {
      seg(t);
      let kk = 0; while (kk + 1 < its.length && t >= its[kk + 1].t0) kk++;
      if (kk === k) return; k = kk;
      const r = its[kk], lastU = its.slice(0, kk + 1).filter(x => x.user).pop();
      if (lastU && tx.dataset.u !== String(lastU.it)) { tx.dataset.u = lastU.it; typewrite(tx, lastU.user); }
      else if (!lastU) { clearInterval(tx._t); tx.dataset.u = ''; tx.textContent = 'looking around the meadow'; }
      const done = writes.filter(x => x.it <= r.it), last = done[done.length - 1];
      lis.forEach((l, j) => l.classList.toggle('on', mode === 'ours' && j < done.length));
      if (mode !== 'ours') { eb.removeAttribute('src'); ee.removeAttribute('src'); status.textContent = 'The original model keeps past views but not the changes. A change behind the camera is never drawn.'; }
      else if (last) {
        eb.src = last.write.base; ee.src = last.write.edit;
        status.textContent = last.write.where === 'behind' ? 'Behind the camera: only memory changes now. When the camera turns, the campfire is there.' : 'In view: the edited view enters the model’s memory, so the change is drawn and stays.';
      } else { eb.removeAttribute('src'); ee.removeAttribute('src'); status.textContent = 'Nothing has changed yet.'; }
    }
    const ctl = setupPlayer(pane, v, onTime);
    onTime(0);
    return ctl;
  }

  (function demos() {
    const SHORT = { gpu_build_pc: 'Graphics card', kintsugi_robot_k7: 'Kintsugi robot', chef_mei_long_day: 'Chef Mei', detective_silver_key: 'Silver key' };
    const X = D.rt || { fasth3: {}, offline: {} };
    let mode = 'fasth3';
    const groups = [];
    function group(tabs, panes, items, first) {
      tabs.innerHTML = items.map((it, j) => (j && it.grp !== items[j - 1].grp ? '<span class="sep"></span>' : '') +
        `<button class="tab" role="tab" data-id="${it.id}" aria-selected="false">${esc(it.label)}<span class="tp"><i></i></span></button>`).join('');
      items.forEach(it => { const p = document.createElement('div'); p.className = 'demo'; p.setAttribute('role', 'tabpanel'); panes.appendChild(p); it.pane = p; });
      const G = { items, active: null, inView: false, onSelect: null };
      G.select = function (id, autoplay) {
        items.forEach(it => {
          const on = it.id === id;
          it.pane.classList.toggle('on', on);
          $(`.tab[data-id="${it.id}"]`, tabs).setAttribute('aria-selected', String(on));
          if (on) {
            if (!it.ctl) it.ctl = it.build(it.pane);
            G.active = it;
            const vid = $('video', it.pane); if (vid && vid.preload === 'none' && G.inView) vid.preload = 'auto';
            if (vid && !vid.dataset.tp) { vid.dataset.tp = 1; const bar = $(`.tab[data-id="${it.id}"] .tp i`, tabs); const upd = () => { if (vid.duration) bar.style.width = (100 * vid.currentTime / vid.duration) + '%'; }; vid.addEventListener('timeupdate', upd); vid.addEventListener('seeked', upd); }
            if (autoplay && G.inView) it.ctl.play();
          } else if (it.ctl) it.ctl.pause();
        });
        if (G.onSelect) G.onSelect(G.active);
      };
      tabs.addEventListener('click', e => { const bt = e.target.closest('.tab'); if (bt) G.select(bt.dataset.id, true); });
      G.select(first || items[0].id, false);
      new IntersectionObserver(es => {
        G.inView = es[0].isIntersecting;
        if (!G.active || !G.active.ctl) return;
        const av = $('video', G.active.pane); if (G.inView && av && av.preload === 'none') av.preload = 'auto';
        if (G.inView && !RM) { groups.forEach(o => { if (o !== G && o.active && o.active.ctl) o.active.ctl.pause(); }); G.active.ctl.play(); }
        else G.active.ctl.pause();
      }, { threshold: .35 }).observe(panes);
      groups.push(G);
      return G;
    }
    const models = d => {
      const out = [];
      const fh = X.fasth3[d.id];
      if (fh) out.push({ key: 'fasth3', video: fh.video, poster: fh.poster, times: fh.shots });
      out.push({ key: 'longlive_h3', video: d.video, poster: d.poster, times: d.shots.map(r => ({ t0: r.t0, t1: r.t1 })) });
      const off = X.offline[d.id];
      if (off) out.push({ key: 'offline', video: off.video, poster: off.poster, times: off.shots.map(r => ({ t0: r.t0, t1: r.t1 })), noAgent: true });
      return out;
    };
    const VH = '';
    const vh = (key, x, extra) => Object.assign({ key }, x, { video: VH + x.video }, extra || {});
    const wallModels = d => {
      const m = d.models, out = [];
      if (m.fasth3) out.push(vh('fasth3', m.fasth3));
      out.push(vh('longlive_h3', m.longlive_h3));
      out.push(m.offline ? vh('offline', m.offline, { noAgent: true })
        : { key: 'offline', noAgent: true, pending: true, video: '', poster: '', times: m.longlive_h3.times });
      return out;
    };
    const seg = $('#modelSeg'), segBtns = $$('button', seg);
    const TL_IDS = ['gpu_build_pc', 'painter_journey'], wall21 = (D.wall || []).filter(w => !TL_IDS.includes(w.id));
    const wallG = wall21.length ? buildWall($('#rtWall'), wall21, wallModels, () => mode) : buildWall($('#rtWall'), D.demos, models, () => mode);
    const wallGroup = { active: { ctl: wallG.ctl } };
    groups.push(wallGroup);
    wallG.onView = () => groups.forEach(o => { if (o !== wallGroup && o.active && o.active.ctl) o.active.ctl.pause(); });
    seg.addEventListener('click', e => {
      const bt = e.target.closest('button'); if (!bt || bt.disabled || bt.dataset.m === mode) return;
      mode = bt.dataset.m; segBtns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.m === mode))); wallG.setModel(mode);
    });
    const tlBox = $('#tlWall'), tlSeg = $('#tlSeg');
    if (tlBox && D.wall && D.wall.length) {
      const TL = TL_IDS.map(id => D.wall.find(w => w.id === id)).filter(Boolean);
      let tlMode = 'fasth3';
      const tlG = buildWall(tlBox, TL, wallModels, () => tlMode), tlGroup = { active: { ctl: tlG.ctl } };
      groups.push(tlGroup);
      tlG.onView = () => groups.forEach(o => { if (o !== tlGroup && o.active && o.active.ctl) o.active.ctl.pause(); });
      const tlBtns = $$('button', tlSeg);
      tlSeg.addEventListener('click', e => {
        const bt = e.target.closest('button'); if (!bt || bt.dataset.m === tlMode) return;
        tlMode = bt.dataset.m; tlBtns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.m === tlMode))); tlG.setModel(tlMode);
      });
    }
    if ($('#bvTabs')) group($('#bvTabs'), $('#bvPanes'), [{ id: 'robot', label: 'Wrist camera', build: p => buildRobot(D.apps.robot, p) }]);
  })();

  const GS = gpu.shots;
  function versions(id) {
    const out = [], seen = {};
    GS.forEach(r => (r.bank || []).forEach(b => { if (b.id === id && !seen[b.version]) { seen[b.version] = 1; out.push(b); } }));
    return out;
  }
  function dstage(host, W = 760, H = 400) {
    host.innerHTML = '<div class="stage"></div>';
    const st = host.firstChild, sz = () => fit(host, st, W, H);
    sz(); window.addEventListener('resize', sz);
    return st;
  }
  const blk = (cls, x, y, w, h, txt, extra = '') => `<div class="d-blk ${cls}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px" data-x="${x}" ${extra}>${txt}</div>`;

  function buildAsync(host) {
    const st = dstage(host);
    const X0 = 120, B = 520, E = 752, CW = 50;
    let h = '';
    [['prompts', 20], ['agent', 104], ['prefill', 184], ['DiT', 262]].forEach(([n, y]) => h += `<div class="d-lab" style="left:0;top:${y}px">${n}</div>`);
    h += blk('k', X0, 16, 46, 26, '<i style="font:italic 15px var(--serif)">p<sub>i−1</sub></i>');
    h += blk('k', B, 16, 40, 26, '<i style="font:italic 15px var(--serif)">p<sub>i</sub></i>');
    h += blk('soft', X0 + 4, 92, 60, 28, 'plan') + blk('soft', X0 + 70, 92, 90, 28, 'update');
    h += blk('g', X0 + 166, 92, 34, 28, 'G') + blk('m', X0 + 204, 92, 60, 28, 'M v0');
    h += blk('x', X0 + 268, 132, 26, 28, '×') + blk('m', X0 + 298, 132, 50, 28, 'v1') + blk('soft', X0 + 352, 132, 26, 28, '<span style="color:var(--g-ink)">✓</span>');
    h += `<div class="d-blk m v2" data-x="${X0 + 382}" style="left:${X0 + 382}px;top:132px;width:70px;height:28px;background:transparent">v2</div>`;
    h += blk('m', X0 + 266, 176, 40, 26, 'v0') + blk('m', X0 + 352, 176, 40, 26, 'v1');
    for (let j = 0; j < 8; j++) h += blk('c', X0 + j * CW + 2, 254, CW - 4, 34, '');
    h += blk('g', B + 2, 254, 22, 34, '') + blk('m', B + 26, 254, 22, 34, '');
    for (let j = 0; j < 4; j++) h += blk('c', B + 52 + j * CW, 254, CW - 4, 34, '');
    h += `<svg class="d-svg" width="760" height="400"><path class="dash" d="M ${X0 + 23} 44 V 252"/><path class="dash" d="M ${B + 20} 44 V 252"/><path class="dash mp" d="M ${X0 + 372} 204 C ${X0 + 372} 232, ${B + 24} 226, ${B + 26} 252"/></svg>`;
    h += `<div class="d-play gate" style="top:14px;height:290px"><i class="halo"></i><i class="beam"></i><span>now</span></div>`;
    h += `<div class="d-txt" style="left:${B - 40}px;top:310px;font:500 14px var(--mono);color:var(--g-ink)">wait 0.00 s</div>`;
    st.innerHTML = h;
    const play = $('.d-play', st), bl = $$('.d-blk', st), v2 = $('.v2', st);
    const P0 = X0 - 6, DUR = 9;
    return t => {
      const u = (t % DUR) / DUR, x = lerp(P0, E, clamp(u / .88, 0, 1));
      play.style.left = x + 'px';
      bl.forEach(b => { if (b !== v2) b.classList.toggle('fut', +b.dataset.x > x); });
      v2.style.opacity = x < X0 + 382 ? .18 : x < B ? .9 : .2;
      v2.style.textDecoration = x >= B ? 'line-through' : '';
    };
  }

  const LEO = 'media/story/';
  const LEO_P = {
    1: 'Leo and his bichon Mochi share a sunny picnic in the park.',
    4: 'A sudden rain soaks Leo, Mochi and the picnic blanket.',
    6: 'At home, Leo puts on a rain jacket and Mochi a yellow raincoat.',
    8: 'They walk down a rainy street under a red umbrella.'
  };
  let maskN = 0;
  function drawnPath(d, cls) {
    const id = 'dm' + (++maskN);
    return {
      svg: `<defs><mask id="${id}" maskUnits="userSpaceOnUse" x="-60" y="-60" width="900" height="540"><path d="${d}" pathLength="1" stroke="#fff" stroke-width="10" fill="none" stroke-dasharray="1 1" stroke-dashoffset="1" data-mk="${id}"/></mask></defs><path class="${cls}" d="${d}" mask="url(#${id})"/>`,
      id
    };
  }

  function buildLayers(host) {
    const st = dstage(host);
    const ROWS = [
      { n: 'Leo', w: 44, h: 70, dx: 60, v: [['leo_s1', 1], ['leo_s4', 4], ['leo_s6', 6]] },
      { n: 'Mochi', w: 44, h: 70, dx: 60, v: [['mochi_s1', 1], ['mochi_s4', 4], ['mochi_s6', 6]] },
      { n: 'Objects', w: 56, h: 56, dx: 70, v: [['blanket_s1', 1], ['blanket_s4', 4], ['umbrella_s6', 6], ['umbrella_s8', 8]] },
      { n: 'Places', w: 74, h: 42, dx: 84, v: [['sc_park1', 1], ['sc_park4', 4], ['sc_home6', 6], ['sc_street8', 8]] }];
    const IN_G = ['leo_s6', 'mochi_s6', 'umbrella_s8'], IN_M = ['sc_street8', 'leo_s6', 'mochi_s6'];
    let h = `<div class="d-box" style="left:0;top:0;width:470px;height:400px"></div><div class="d-lab m" style="left:16px;top:10px">entity memory</div>`;
    let y = 36;
    ROWS.forEach(r => {
      h += `<div class="d-lab" style="left:16px;top:${y + r.h / 2 - 9}px">${r.n}</div>`;
      r.v.forEach(([f, p], j) => {
        const x = 104 + j * r.dx;
        h += `<div class="d-img lv" data-p="${p}" data-f="${f}" style="left:${x}px;top:${y}px;width:${r.w}px;height:${r.h}px"><img src="${LEO + f}.webp" alt=""></div>`;
        h += `<div class="d-lab m lvt" data-p="${p}" style="left:${x}px;top:${y + r.h + 2}px;font-size:11.5px">@p${p}</div>`;
      });
      y += r.h + 24;
    });
    h += `<div class="d-txt lp" style="left:16px;top:372px;width:440px"></div>`;
    h += `<div class="d-lab m" style="left:520px;top:10px">memory images for the next shot</div>`;
    h += `<div class="d-img lg" style="left:520px;top:36px;width:224px;height:126px"><img src="${LEO}G.webp" alt=""></div><div class="d-lab lgl" style="left:520px;top:168px">global image <b>G</b></div>`;
    h += `<div class="d-img lm" style="left:520px;top:214px;width:224px;height:126px"><img src="${LEO}M.webp" alt=""></div><div class="d-lab lml" style="left:520px;top:346px">shot memory <b>M</b></div>`;
    const pg = drawnPath('M 472 120 C 496 120, 496 99, 516 99', 'dash gp'), pm = drawnPath('M 472 250 C 496 250, 496 277, 516 277', 'dash mp');
    h += `<svg class="d-svg" width="760" height="400">${pg.svg}${pm.svg}</svg>`;
    st.innerHTML = h;
    const lv = $$('.lv', st), lvt = $$('.lvt', st), lp = $('.lp', st), lg = $('.lg', st), lm = $('.lm', st), mk = $$('[data-mk]', st);
    const STEPS = [1, 4, 6, 8, 9], STEP = 1.9, DUR = STEPS.length * STEP + 1.6;
    let last = -1;
    return t => {
      const u = t % DUR, j = Math.min(STEPS.length - 1, Math.floor(u / STEP)), p = STEPS[j], comp = p === 9;
      const a = comp ? clamp((u - j * STEP) / .7, 0, 1) : 0;
      mk.forEach(m => m.setAttribute('stroke-dashoffset', (1 - a).toFixed(3)));
      lg.style.opacity = lm.style.opacity = comp ? (.2 + .8 * a).toFixed(2) : .14;
      if (j === last) return; last = j;
      lv.forEach(e => {
        const q = +e.dataset.p, f = e.dataset.f;
        e.style.opacity = q <= Math.min(p, 8) ? 1 : .12;
        e.style.boxShadow = comp ? (IN_G.includes(f) && IN_M.includes(f) ? '0 0 0 2px var(--g), 0 0 0 4px var(--m)' : IN_G.includes(f) ? '0 0 0 2px var(--g)' : IN_M.includes(f) ? '0 0 0 2px var(--m)' : '') : q === p ? '0 0 0 2px var(--ink)' : '';
      });
      lvt.forEach(e => e.style.opacity = +e.dataset.p <= Math.min(p, 8) ? 1 : .3);
      lp.innerHTML = comp ? 'Before shot 9: G and M are composed from the latest entity images.' : `<span class="pq">Prompt ${p}</span>${esc(LEO_P[p])}`;
    };
  }

  function buildSink(host) {
    const st = dstage(host);
    const GX = 162, SX = 230, S = 52, TY = 84, D = 6.6, LOOP = 3 * D;
    const slotP = k => k === 'stage' ? [640, 8] : [SX + k * S, TY];
    const ROUTE = [2, 1, 0];
    const SHOT = ['shot <i>i</i>', 'shot <i>i</i>+1', 'shot <i>i</i>+2'];
    const B = [];
    const mk = (cls, label, tin, slot, from) => { const b = { cls, label, tin, ev: [[tin, slot]], from, tout: 1e9, exit: 'drop' }; B.push(b); return b; };
    const go = (b, t, slot) => b.ev.push([t, slot]);
    let ms = [mk('m', 'M<sub><i>i</i>−2</sub>', 0, 0, 'fade'), mk('m', 'M<sub><i>i</i>−1</sub>', 0, 1, 'fade')];
    const TOP = [];
    for (let c = 0; c < 3; c++) {
      const t0 = c * D, nM = ms.length;
      const ch = [0, 1, 2].map(k => mk('c', k ? '' : '1st', t0 + .35 + k * .55, nM + .5 + k, 'right'));
      TOP.push([t0, t0 + 2, 'mem', nM]);
      ms.forEach(b => b.tout = t0 + 2);
      ch.forEach((b, k) => go(b, t0 + 2, k));
      ch[0].pin = t0 + 2;
      TOP.push([t0 + 2, t0 + 5.7, 'roll', 0]);
      const recent = [ch[1], ch[2]], all = [...ch];
      for (let k = 0; k < 8; k++) {
        const t = t0 + 2.5 + k * .38;
        if (recent.length === 8) { const old = recent.shift(); old.tout = t; recent.forEach((b, q) => go(b, t, q + 1)); }
        const b = mk('c', '', t, recent.length + 1, 'right'); recent.push(b); all.push(b);
      }
      const lab = ['M<sub><i>i</i></sub>', 'M<sub><i>i</i>+1</sub>', 'M<sub><i>i</i>+2</sub>'][c];
      const nm = mk('m', lab, t0 + 4.8, 'stage', 'above');
      const tb = t0 + 5.7;
      all.forEach((b, q) => { if (b.tout > tb) b.tout = tb + q * .03; });
      const r = ROUTE[c];
      if (r === 2) { nm.tout = tb; nm.exit = 'store'; ms = []; }
      else if (r === 1) { nm.tout = tb; nm.exit = 'store'; ms = [mk('m', 'M<sub>old</sub>', tb + .2, 0, 'fade')]; }
      else { const prev = mk('m', 'M<sub><i>i</i>+1</sub>', tb + .2, 0, 'fade'); go(nm, tb + .2, 1); ms = [prev, nm]; }
    }
    ms.forEach(b => b.tout = LOOP - .05);
    let h = `<div class="d-lab" style="left:0;top:${TY + 4}px">attention window</div><div class="d-lab m sh" style="left:0;top:${TY + 24}px"></div>`;
    h += `<div class="sk-frame" style="left:150px;top:${TY - 10}px;width:592px;height:62px"></div><div class="sk-div" style="left:218px;top:${TY - 4}px;height:50px"></div>`;
    h += `<div class="d-blk g" style="left:${GX}px;top:${TY}px;width:46px;height:42px">G</div>`;
    h += `<div class="sk-top" style="left:${GX - 6}px;top:${TY - 32}px">global sink</div>`;
    h += `<div class="sk-top tm" style="top:${TY - 32}px">memory</div><div class="sk-top ts" style="left:${SX - 6}px;top:${TY - 32}px">shot sink</div><div class="sk-top tr" style="left:${SX + S - 2}px;top:${TY - 32}px">recent chunks</div>`;
    h += `<div class="sk-top ta" style="left:696px;top:20px">from agent</div>`;
    B.forEach((b, j) => h += `<div class="d-blk ${b.cls} sk-b" data-b="${j}" style="width:46px;height:42px;opacity:0">${b.label}</div>`);
    h += `<div class="d-blk k sk-p" style="left:0;top:224px;width:120px;height:36px;font-size:12.5px">next prompt</div>`;
    const rows = [['same place', [['g', 'G'], ['m', 'M<sub><i>i</i>−1</sub>'], ['m', 'M<sub><i>i</i></sub>']]], ['earlier place', [['g', 'G'], ['e x', '×'], ['m', 'M<sub>old</sub>']]], ['new place', [['g', 'G'], ['e x', '×'], ['e x', '×']]]];
    rows.forEach(([n, bl], q) => {
      const y = 170 + q * 72;
      h += `<div class="d-lab rl" data-q="${q}" style="left:172px;top:${y + 12}px;font-size:14px">${n}</div>`;
      bl.forEach(([c, tx], j) => h += `<div class="d-blk ${c} rb" data-q="${q}" style="left:${300 + j * S}px;top:${y}px;width:46px;height:42px">${tx}</div>`);
      for (let j = 0; j < 3; j++) h += `<div class="d-blk e rb" data-q="${q}" style="left:${300 + 3 * S + 14 + j * S}px;top:${y}px;width:46px;height:42px"></div>`;
    });
    h += `<svg class="d-svg" width="760" height="400"><path class="dash rpath" d="M 120 242 L 120 242"/></svg>`;
    st.innerHTML = h;
    const els = $$('.sk-b', st), sh = $('.sh', st), tm = $('.tm', st), ts = $('.ts', st), tr = $('.tr', st), ta = $('.ta', st), pr = $('.sk-p', st);
    const rl = $$('.rl', st), rb = $$('.rb', st), rpath = $('.rpath', st);
    const ease = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2, sg = (x, a, b) => clamp((x - a) / (b - a), 0, 1);
    return t => {
      const u = t % LOOP, c = Math.min(2, Math.floor(u / D)), cu = u - c * D;
      const fadeAll = 1 - sg(u, LOOP - .35, LOOP);
      B.forEach((b, j) => {
        const el = els[j];
        if (u < b.tin || u > b.tout + .7) { el.style.opacity = 0; return; }
        let k = 0;
        while (k + 1 < b.ev.length && b.ev[k + 1][0] <= u) k++;
        const cur = slotP(b.ev[k][1]);
        let prev;
        if (k) prev = slotP(b.ev[k - 1][1]);
        else prev = b.from === 'right' ? [cur[0] + 60, cur[1]] : b.from === 'above' ? [cur[0], cur[1] - 30] : cur;
        const e = ease(sg(u, b.ev[k][0], b.ev[k][0] + .45));
        let x = lerp(prev[0], cur[0], e), y = lerp(prev[1], cur[1], e), op = sg(u, b.tin, b.tin + .3), rot = 0;
        if (u > b.tout) {
          const q = ease(sg(u, b.tout, b.tout + .6));
          if (b.exit === 'store') { x += 50 * q; y -= 26 * q; }
          else { y += 46 * q; rot = 10 * q; }
          op *= 1 - q;
        }
        el.style.left = x.toFixed(1) + 'px'; el.style.top = y.toFixed(1) + 'px';
        el.style.transform = rot ? `rotate(${rot.toFixed(1)}deg)` : '';
        el.style.opacity = (op * fadeAll).toFixed(2);
        el.classList.toggle('pin', b.pin != null && u >= b.pin);
      });
      sh.innerHTML = SHOT[c];
      const nM = c === 0 ? 2 : c === 1 ? 0 : 1;
      const memOn = cu < 2 && nM > 0, rollOn = cu >= 2.15 && cu < 5.7;
      tm.style.left = (SX - 6) + 'px'; tm.style.opacity = memOn ? 1 : 0;
      ts.style.opacity = rollOn ? 1 : 0; tr.style.opacity = rollOn ? 1 : 0;
      ta.style.opacity = cu >= 4.9 && cu < 5.7 ? 1 : 0;
      const q = cu >= 4.35 || cu < .5 && u > D ? ROUTE[cu >= 4.35 ? c : c - 1] : -1;
      pr.style.opacity = cu >= 4.2 || (cu < .5 && u > D) ? 1 : .18;
      rl.forEach(e => { e.style.color = +e.dataset.q === q ? 'var(--ink)' : 'var(--faint)'; });
      rb.forEach(e => e.style.opacity = +e.dataset.q === q ? 1 : .16);
      rpath.setAttribute('d', q >= 0 ? `M 120 242 C 146 242, 140 ${191 + q * 72}, 166 ${191 + q * 72}` : 'M 120 242 L 120 242');
    };
  }

  function buildRL(host) {
    const st = dstage(host);
    const json = [['"Leo"', '"rain gear @p6"'], ['"umbrella"', '"wet @p8"'], ['"G"', '["Leo", "Mochi", "umbrella"]'], ['"M"', '["street", "Leo", "Mochi"]']];
    let h = `<div class="d-blk pol" style="left:0;top:0;width:760px;height:42px;justify-content:flex-start;padding-left:16px;font-size:13.5px">policy · Qwen3-8B + LoRA<span class="gr" style="margin-left:auto;margin-right:16px;font:500 12px var(--mono)">GRPO update ×8</span></div>`;
    h += `<div class="d-box" style="left:0;top:62px;width:340px;height:250px"></div><div class="d-lab" style="left:14px;top:72px">memory update after shot 8</div><div class="d-json js" style="left:16px;top:100px"></div>`;
    h += `<div class="d-box" style="left:360px;top:62px;width:400px;height:250px"></div><div class="d-lab" style="left:374px;top:72px">hindsight reward</div>`;
    h += `<div class="d-lab m" style="left:374px;top:110px">text judge</div><div class="d-score ok sc" style="left:520px;top:106px">✓ +1</div><div class="d-score z sc" style="left:580px;top:106px">0</div><div class="d-score no sc" style="left:616px;top:106px">✗ −1.5</div>`;
    h += `<div class="d-lab m" style="left:374px;top:170px">image judge</div><div class="d-img g ig" style="left:520px;top:150px;width:110px;height:62px"><img src="${LEO}G.webp" alt=""></div><div class="d-img m ig" style="left:640px;top:150px;width:110px;height:62px"><img src="${LEO}M.webp" alt=""></div>`;
    h += `<div class="d-lab m wt" style="left:374px;top:232px">real-video weights</div>`;
    h += `<div class="d-score ok wt" style="left:374px;top:262px">G .51</div><div class="d-score mm wt" style="left:438px;top:262px">M .49</div><div class="d-score no wt" style="left:502px;top:262px">off −.11</div>`;
    h += `<div class="d-blk soft rw" style="left:600px;top:256px;width:146px;height:34px;font-size:13px">reward</div>`;
    [1, 4, 6, 8].forEach((n, j) => h += `<div class="d-blk k" style="left:${j * 74}px;top:344px;width:60px;height:26px;border-radius:13px;font-size:12px">p${n}</div>`);
    ['p9', 'p10', '…', 'p14'].forEach((n, j) => h += `<div class="d-blk e" style="left:${400 + j * 74}px;top:344px;width:60px;height:26px;border-radius:13px;font-size:12px">${n}</div>`);
    h += `<div class="d-lab m" style="left:0;top:376px">revealed</div><div class="d-lab m" style="left:400px;top:376px">hidden, used only for scoring</div>`;
    const pa = drawnPath('M 140 340 V 314', 'dash'), pb = drawnPath('M 560 340 V 314', 'dash'), pc = drawnPath('M 746 272 C 790 272, 792 60, 730 42', 'dash rwp up');
    h += `<svg class="d-svg" width="760" height="400">${pa.svg}${pb.svg}${pc.svg}</svg>`;
    h += `<div class="d-dot" style="background:var(--chunk);box-shadow:0 0 0 3px rgba(225,182,75,.25)"></div>`;
    st.innerHTML = h;
    const pol = $('.pol', st), js = $('.js', st), sc = $$('.sc', st), ig = $$('.ig', st), wt = $$('.wt', st), rw = $('.rw', st), dot = $('.d-dot', st), gr = $('.gr', st);
    const mA = $(`[data-mk="${pa.id}"]`, st), mB = $(`[data-mk="${pb.id}"]`, st), mC = $(`[data-mk="${pc.id}"]`, st);
    const up = $('.up', st), L = up.getTotalLength();
    const DUR = 10, T0 = .7;
    return t => {
      const u = t % DUR, n = Math.floor(clamp((u - T0) / 2.6, 0, 1) * json.length * 10);
      mA.setAttribute('stroke-dashoffset', (1 - clamp(u / .6, 0, 1)).toFixed(3));
      let out = '{\n';
      json.forEach((kv, j) => {
        const k0 = j * 10; if (n <= k0) return;
        const s = kv[0] + ': ' + kv[1], c = clamp((n - k0) / 10, 0, 1), shown = s.slice(0, Math.ceil(s.length * c)), sp = shown.indexOf(':');
        out += '  ' + (sp > 0 ? `<span class="k">${esc(shown.slice(0, sp + 1))}</span><span class="v">${esc(shown.slice(sp + 1))}</span>` : `<span class="k">${esc(shown)}</span>`) + (j < json.length - 1 && c >= 1 ? ',' : '') + '\n';
      });
      js.innerHTML = out + (u < T0 + 2.6 ? '<span class="cur"></span>' : '}');
      mB.setAttribute('stroke-dashoffset', (1 - clamp((u - 3.4) / .5, 0, 1)).toFixed(3));
      sc.forEach((s, j) => s.style.opacity = u > 3.9 + j * .3 ? 1 : .12);
      ig.forEach(s => s.style.opacity = u > 4.9 ? 1 : .15);
      wt.forEach((s, j) => s.style.opacity = u > 5.4 + j * .25 ? 1 : .12);
      rw.classList.toggle('won', u > 6.4);
      const a = clamp((u - 6.6) / 1.3, 0, 1);
      mC.setAttribute('stroke-dashoffset', (1 - a).toFixed(3));
      const p = up.getPointAtLength(L * a);
      dot.style.left = p.x + 'px'; dot.style.top = p.y + 'px'; dot.style.opacity = a > 0 && a < 1 ? 1 : 0;
      gr.style.opacity = u > 8 ? 1 : .25;
      pol.classList.toggle('on', u > 7.9);
    };
  }

  (function methodBlocks() {
    const DIAS = { async: buildAsync, layers: buildLayers, sink: buildSink, rl: buildRL };
    $$('[data-dia]').forEach(host => {
      const tick = DIAS[host.dataset.dia](host);
      tick(30);
      let vis = false, t0 = 0;
      function loop(now) { if (!vis) return; tick(RM ? 30 : (now - t0) / 1000); if (!RM) requestAnimationFrame(loop); }
      new IntersectionObserver(es => {
        const was = vis; vis = es[0].isIntersecting;
        if (vis && !was) { t0 = performance.now(); requestAnimationFrame(loop); }
      }, { threshold: .3 }).observe(host);
    });
  })();

  (function gpus() {
    const host = $('#gpuDia'); if (!host) return;
    host.innerHTML = '<div class="stage"></div>';
    const st = host.firstChild;
    const isPh = () => host.clientWidth < 560;
    let PH = isPh(), L = null;
    const ov = (D.wall || []).find(w => w.id === 'gpu_build_pc');
    const SRC = ov ? {
      fh: { video: ov.models.fasth3.video, poster: ov.models.fasth3.poster, shots: ov.shots },
      ar: { video: ov.models.longlive_h3.video, poster: ov.models.longlive_h3.poster, shots: ov.models.longlive_h3.shots || ov.shots } }
      : { fh: gpu, ar: gpu };
    const AGENT = [['Planner', 'Qwen3-8B, RL'], ['Checker', 'Qwen3.6-35B'], ['Drawer', 'FLUX.2-klein']];
    const FH = { TXT: 1.28, DIT: 5.41, VAE: 6.57, GENE: 6.57, VIEW: 7.7, PLAN: 2.2, COMMIT: 5.2 };
    const AR = { ENC: 0.5, ROLL: 5.30, FIRST: 1.5, GENE: 5.53, PLAN: 2.3, COMMIT: 5.8, NCH: 11 };
    const slots = (p, c) => [[[0, p]], [[3.0, 3.5], [4.1, c], [6.3, 7.2]], [[p, 3.0], [3.5, 4.1], [5.3, 6.3]]];
    const chunkAt = q => AR.FIRST + q * (AR.GENE - AR.FIRST) / (AR.NCH - 1);
    const dia = v => v.replace(/\.mp4$/, '_dia.mp4');
    const card = (cls, x, y, id, r, i) => `<div class="gp ${cls}" style="left:${x}px;top:${y}px"><span class="id">GPU ${id}</span><div class="bars">${'<i></i>'.repeat(6)}</div><div class="rl">${r}<i>${i || '&nbsp;'}</i></div></div>`;
    const rows = (list, y0, sc) => list.map(([n, y, c, e, lab], q) => `<div class="gp-tll" style="left:14px;top:${y - 6}px">${n}</div><div class="gp-tl" style="left:96px;top:${y}px;width:${sc * e / 8}px"><div class="f" data-q="${q}" style="background:${c}"></div></div><div class="gp-tll" style="left:${104 + sc * e / 8}px;top:${y - 6}px;color:var(--muted)">${lab}</div>`).join('');
    const screen = (S, b) => `<div class="gp-screen" style="left:${b.x}px;top:${b.y}px;width:${b.w}px;height:${b.h}px"><video class="gv" src="${dia(S.video)}" poster="${S.poster}" muted playsinline loop preload="${near ? 'auto' : 'metadata'}"></video></div>` +
      `<div class="gp-tll gp-stat" style="left:${b.x}px;top:${b.y + b.h + 9}px"></div><div class="gp-tll" style="left:${b.x}px;top:${b.y + b.h + 33}px;color:var(--g-ink);font-family:var(--mono)">wait 0.00 s</div>`;
    const thumbs = ({ a, b }) => `<div class="gp-thumb t1" style="left:${a[0]}px;top:${a[1]}px"><img src="" alt=""></div><div class="gp-thumb t2" style="left:${b[0]}px;top:${b[1]}px"><img src="" alt=""></div>` +
      `<div class="gp-tll" style="left:${a[0] + 58}px;top:${a[1] + 6}px;color:var(--muted)">entity</div><div class="gp-tll" style="left:${b[0] + 58}px;top:${b[1] + 6}px;color:var(--muted)">memory</div>`;
    const nodes = L => L.nodes.map(([x, y, w, h]) => `<div class="gp-node" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px"></div>`).join('');
    const pills = L => `<div class="gp-in i1" style="left:${L.pill[0]}px;top:${L.pill[1]}px">p1</div><div class="gp-in i2" style="left:${L.pill[0]}px;top:${L.pill[1]}px">p1</div>`;
    const KEY = (x, y, t) => `<div class="gp-key" style="left:${x}px;top:${y}px"><span><i style="background:var(--base)"></i>${t}</span><span><i style="background:var(--chunk)"></i>DiT</span><span><i style="background:var(--base-2)"></i>VAE decoder</span></div>`;

    function layFH(ph) {
      const GF = [0, 1, 2, 3, 4, 5, 6, 7].map(j => ({ x: ph ? 10 + (j % 4) * 101 : 14 + j * 123, y: ph ? (j < 4 ? 54 : 156) : 36,
        i: j < 2 ? '+ text, VAE' : j < 4 ? '+ text' : '' }));
      const AF = [0, 1, 2].map(j => ph ? { x: 10 + j * 101, y: 318 } : { x: [94, 204, 314][j], y: 226 });
      return ph ? { W: 420, H: 822, GF, AF, nodes: [[0, 0, 420, 262], [0, 286, 420, 138]], g1: [10, 8], key: [10, 30], g2: [10, 294],
          th: { a: [304, 324], b: [304, 366] }, pill: [10, 263], scr: { x: 0, y: 444, w: 420, h: 236 }, rows: [750, 776, 802], sc: 240,
          i1: [10, 263, 24, 346], i2: [10, 263, 24, 120], clip: [[111, 414, 210], [150, 300, 540]], wait: [396, 300], dst: q => [GF[q].x + 48, GF[q].y + 80] }
        : { W: 1000, H: 452, GF, AF, nodes: [[0, 0, 1000, 172], [0, 192, 430, 166]], g1: [14, 10], key: [600, 11], g2: [94, 202],
          th: { a: [446, 236], b: [446, 274] }, pill: [14, 274], scr: { x: 640, y: 192, w: 360, h: 203 }, rows: [384, 410, 436], sc: 280,
          i1: [14, 274, 78, 274], i2: [14, 274, 14, 160], clip: [[187, 520, 652], [160, 176, 296]], wait: [300, 196], dst: q => [GF[q].x + 50, 156] };
    }
    function htmlFH() {
      let h = nodes(L);
      h += `<div class="gp-grp" style="left:${L.g1[0]}px;top:${L.g1[1]}px">Video generation: FastH3-Omni on one 8×B200 node</div>` + KEY(L.key[0], L.key[1], 'text encoder');
      h += `<div class="gp-grp" style="left:${L.g2[0]}px;top:${L.g2[1]}px">Memory agent: three B200s of a second node</div>`;
      L.GF.forEach((g, j) => h += card('gen', g.x, g.y, j, 'DiT', g.i));
      L.AF.forEach((g, j) => h += card('ag', g.x, g.y, j, AGENT[j][0], AGENT[j][1]));
      h += thumbs(L.th) + pills(L) + screen(SRC.fh, L.scr);
      h += rows([['playback', L.rows[0], 'var(--ink)', 8, '8 s'], ['next shot', L.rows[1], 'var(--chunk)', FH.GENE, '6.6 s'], ['memory', L.rows[2], 'var(--g)', FH.COMMIT, '5.2 s']], 0, L.sc);
      h += `<div class="gp-tok clip"></div>`;
      for (let q = 0; q < 8; q++) h += `<div class="gp-tok g" data-m="${q}"></div>`;
      return h;
    }
    function layAR(ph) {
      const at = j => ph ? { x: 10 + (j % 4) * 101, y: j < 4 ? 54 : 176 } : { x: 14 + j * 123, y: 36 };
      const GA = [0, 1, 2, 3, 4].map(at), AA = [5, 6, 7].map(at);
      return ph ? { W: 420, H: 724, GA, AA, nodes: [[0, 0, 420, 282]], g1: [10, 8], g2: [111, 156], key: [10, 30],
          th: { a: [10, 296], b: [150, 296] }, pill: [330, 298], scr: { x: 0, y: 340, w: 420, h: 236 }, rows: [648, 674, 700], sc: 196,
          i1q: [[330, 300, 150], [298, 290, 250]], i2: [330, 298, 40, 120], chunk: [58, 268, 380], wait: [280, 311], dst: q => [GA[q].x + 48, GA[q].y + 80] }
        : { W: 1000, H: 452, GA, AA, nodes: [[0, 0, 1000, 172]], g1: [14, 10], g2: [629, 10], key: [14, 186],
          th: { a: [840, 236], b: [840, 274] }, pill: [14, 232], scr: { x: 460, y: 192, w: 360, h: 203 }, rows: [318, 344, 370], sc: 240,
          i1q: [[14, 200, 640], [232, 160, 160]], i2: [14, 232, 14, 160], chunk: [556, 158, 236], wait: [812, 204], dst: q => [GA[q].x + 50, 156] };
    }
    function htmlAR() {
      let h = nodes(L);
      h += `<div class="gp-grp" style="left:${L.g1[0]}px;top:${L.g1[1]}px">Video generation: LongLive-H3 on five GPUs of one 8×B200 node</div>`;
      h += `<div class="gp-grp" style="left:${L.g2[0]}px;top:${L.g2[1]}px">Memory agent: the other three GPUs</div>` + KEY(L.key[0], L.key[1], 'text and memory pictures');
      L.GA.forEach((g, j) => h += j < 4 ? card('gen', g.x, g.y, j, 'DiT', '+ text') : card('gen', g.x, g.y, j, 'VAE decoder', 'streaming'));
      L.AA.forEach((g, j) => h += card('ag', g.x, g.y, j + 5, AGENT[j][0], AGENT[j][1]));
      h += thumbs(L.th) + screen(SRC.ar, L.scr);
      h += rows([['next prompt', L.rows[0], 'var(--ink)', 8, '8 s'], ['this shot', L.rows[1], 'var(--chunk)', AR.GENE, '5.5 s, first frames at 1.5 s'], ['memory', L.rows[2], 'var(--g)', AR.COMMIT, '5.8 s']], 0, L.sc);
      h += `<div class="gp-mk" style="left:${96 + L.sc * AR.FIRST / 8}px;top:${L.rows[1] - 4}px;height:16px"></div>`;
      h += pills(L);
      for (let q = 0; q < AR.NCH; q++) h += `<div class="gp-tok c" data-c="${q}"></div>`;
      for (let q = 0; q < 4; q++) h += `<div class="gp-tok g" data-m="${q}"></div>`;
      return h;
    }

    let mode = 'fh', vis = false, near = false, t0 = performance.now(), lastK = -1, lastStat = '', E = {}, vid = null;
    const bar = (b, on, t, q, j) => b.style.height = (on ? 35 + 60 * Math.abs(Math.sin(t * (3.1 + q * .7) + q * 1.7 + j)) : 10) + '%';
    const qb2 = (p0, c, p1, u) => (1 - u) * (1 - u) * p0 + 2 * (1 - u) * u * c + u * u * p1;
    const setStat = s => { if (s !== lastStat) { lastStat = s; E.stat.textContent = s; } };
    const setThumbs = r => {
      const e = (r.bank || []).find(b => b.changed && b.kind !== 'scene') || (r.bank || [])[0], m = (r.used || []).filter(u => u.role === 'M').pop();
      if (e) E.ti1.src = e.img; if (m) E.ti2.src = m.img;
    };
    function busy(cards, bars, stg, t, off) {
      cards.forEach((c, j) => {
        const sg = stg(j);
        if (c.dataset.st !== sg) { c.dataset.st = sg; c.classList.toggle('busy', !!sg); }
        bars[j].forEach((b, q) => bar(b, !!sg, t, q, j + off));
      });
    }
    function agents(c, p, cm, t) {
      const ab = slots(p, cm);
      E.acards.forEach((a, j) => { const on = ab[j].some(([x, y]) => c >= x && c < y); a.classList.toggle('busy', on); E.abars[j].forEach((b, q) => bar(b, on, t, q, j + 8)); });
    }

    function tickFH(t) {
      const G = SRC.fh.shots, n = G.length, k = Math.floor(t / 8), s = t - k * 8, sh = G[k % n];
      if (k !== lastK) {
        lastK = k; setThumbs(G[(k + 2) % n]);
        setStat(`on screen: shot ${sh.i + 1} · generating shot ${sh.i + 2} · memory for shot ${sh.i + 3}`);
        E.i1.textContent = E.i2.textContent = 'p' + (sh.i + 2);
      }
      busy(E.gcards, E.gbars, j => s >= FH.TXT && s < FH.DIT ? 'dit' : j < 4 && s < FH.TXT ? 'text' : j < 2 && s >= FH.DIT && s < FH.VAE ? 'vae' : '', t, 0);
      agents(s, FH.PLAN, FH.COMMIT, t);
      E.fills.forEach(f => { const e = [8, FH.GENE, FH.COMMIT][+f.dataset.q]; f.style.width = (100 * clamp(s / e, 0, 1)) + '%'; });
      const pa = clamp(s / .7, 0, 1), po = s < .9 ? 1 : 0;
      E.i1.style.left = lerp(L.i1[0], L.i1[2], pa) + 'px'; E.i1.style.top = lerp(L.i1[1], L.i1[3], pa) + 'px'; E.i1.style.opacity = po;
      E.i2.style.left = lerp(L.i2[0], L.i2[2], pa) + 'px'; E.i2.style.top = lerp(L.i2[1], L.i2[3], pa) + 'px'; E.i2.style.opacity = po;
      const a = clamp((s - FH.GENE) / (FH.VIEW - FH.GENE), 0, 1), on = s >= FH.GENE && s < FH.VIEW + .25;
      E.clip.style.left = qb2(...L.clip[0], a) + 'px'; E.clip.style.top = qb2(...L.clip[1], a) + 'px'; E.clip.style.opacity = on ? 1 : 0;
      E.t1.style.opacity = s > 3.0 ? 1 : 0; E.t2.style.opacity = s > 4.1 ? 1 : 0;
      E.mems.forEach((m, q) => {
        const [tx, ty] = L.dst(q), [wx, wy] = L.wait;
        if (s >= FH.COMMIT) { m.style.left = wx + 'px'; m.style.top = wy + 'px'; m.style.opacity = q === 0 ? 1 : 0; }
        else if (s < .7) { const u = s / .7; m.style.left = lerp(wx, tx, u) + 'px'; m.style.top = lerp(wy, ty, u) + 'px'; m.style.opacity = 1 - u * .3; }
        else m.style.opacity = 0;
      });
    }

    function tickAR(t) {
      const G = SRC.ar.shots, n = G.length, LF = AR.FIRST, k = Math.floor(t / 8), s = t - k * 8;
      const late = s >= 8 - LF, x = late ? k + 1 : k, c = late ? s - (8 - LF) : s + LF;
      if (x !== lastK) { lastK = x; setThumbs(G[(x + 1) % n]); E.i1.textContent = E.i2.textContent = 'p' + (x % n + 1); }
      setStat(c < AR.GENE ? `on screen: shot ${k % n + 1} · generating shot ${x % n + 1} · memory for shot ${x % n + 2}`
        : `on screen: shot ${k % n + 1} · waiting for prompt ${x % n + 2} · memory for shot ${x % n + 2}`);
      busy(E.gcards, E.gbars, j => j < 4 ? (c < AR.ENC ? 'text' : c < AR.ROLL ? 'dit' : '')
        : ([...Array(AR.NCH).keys()].some(q => c >= chunkAt(q) - .3 && c < chunkAt(q)) ? 'vae' : ''), t, 0);
      agents(c, AR.PLAN, AR.COMMIT, t);
      E.fills.forEach(f => { const e = [8, AR.GENE, AR.COMMIT][+f.dataset.q]; f.style.width = (100 * clamp(c / e, 0, 1)) + '%'; });
      const pa = clamp(c / .7, 0, 1), po = c < .9 ? 1 : 0;
      E.i2.style.left = lerp(L.i2[0], L.i2[2], pa) + 'px'; E.i2.style.top = lerp(L.i2[1], L.i2[3], pa) + 'px'; E.i2.style.opacity = po;
      E.i1.style.left = qb2(...L.i1q[0], pa) + 'px'; E.i1.style.top = qb2(...L.i1q[1], pa) + 'px'; E.i1.style.opacity = po;
      E.chunks.forEach((m, q) => {
        const u = (c - chunkAt(q)) / .45;
        if (u >= 0 && u < 1) { m.style.left = L.chunk[0] + 'px'; m.style.top = lerp(L.chunk[1], L.chunk[2], u) + 'px'; m.style.opacity = 1 - u * .4; }
        else m.style.opacity = 0;
      });
      E.t1.style.opacity = c > 3.0 ? 1 : 0; E.t2.style.opacity = c > 4.1 ? 1 : 0;
      E.mems.forEach((m, q) => {
        const [tx, ty] = L.dst(q), [wx, wy] = L.wait;
        if (c >= AR.COMMIT) { m.style.left = wx + 'px'; m.style.top = wy + 'px'; m.style.opacity = q === 0 ? 1 : 0; }
        else if (c < .7) { const u = c / .7; m.style.left = lerp(wx, tx, u) + 'px'; m.style.top = lerp(wy, ty, u) + 'px'; m.style.opacity = 1 - u * .3; }
        else m.style.opacity = 0;
      });
    }

    function step(now) {
      const playing = vid.readyState >= 2 && !vid.paused;
      const t = RM ? (mode === 'fh' ? 6.9 : 25.2) : playing ? vid.currentTime : (now - t0) / 1000;
      (mode === 'fh' ? tickFH : tickAR)(t);
    }
    let running = false;
    function loop(now) { step(now); if (!RM && vis) requestAnimationFrame(loop); else running = false; }
    const start = () => { if (!running && !RM && vis) { running = true; requestAnimationFrame(loop); } };
    let lastTry = 0;
    const tryPlay = () => { if (!vis || RM || !vid.paused) return; lastTry = performance.now(); const pr = vid.play(); if (pr) pr.catch(() => {}); };
    function render(m, at) {
      mode = m; lastK = -1; lastStat = '';
      L = (m === 'fh' ? layFH : layAR)(PH);
      st.classList.toggle('ph', PH);
      fit(host, st, L.W, L.H);
      st.innerHTML = m === 'fh' ? htmlFH() : htmlAR();
      const gcards = $$('.gp.gen', st), acards = $$('.gp.ag', st);
      E = { gcards, acards, gbars: gcards.map(c => $$('.bars i', c)), abars: acards.map(c => $$('.bars i', c)),
        fills: $$('.gp-tl .f', st), mems: $$('[data-m]', st), chunks: $$('[data-c]', st), clip: $('.clip', st), stat: $('.gp-stat', st),
        i1: $('.i1', st), i2: $('.i2', st), t1: $('.t1', st), t2: $('.t2', st), ti1: $('.t1 img', st), ti2: $('.t2 img', st) };
      vid = $('.gv', st);
      ['loadeddata', 'canplay', 'stalled', 'suspend'].forEach(ev => vid.addEventListener(ev, tryPlay));
      if (at) vid.addEventListener('loadedmetadata', () => { try { vid.currentTime = at; } catch (e) { } tryPlay(); }, { once: true });
      step(performance.now());
    }
    render('fh', 0);
    window.addEventListener('resize', () => {
      if (isPh() === PH) { fit(host, st, L.W, L.H); return; }
      PH = isPh(); const at = vid.currentTime || 0; vid.pause(); render(mode, at);
    });
    setInterval(() => { if (vis && !RM && vid.paused && performance.now() - lastTry > 1500) { if (vid.networkState === 3 || vid.error) vid.load(); tryPlay(); } }, 1600);
    new IntersectionObserver(es => { if (es[0].isIntersecting && !near) { near = true; vid.preload = 'auto'; } }, { rootMargin: '700px 0px' }).observe(host);
    new IntersectionObserver(es => {
      const was = vis; vis = es[0].isIntersecting;
      if (vis) tryPlay(); else vid.pause();
      if (vis && !was) start();
    }, { threshold: .15 }).observe(host);
    const seg = $('#gpSeg');
    if (seg) seg.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b || b.dataset.k === mode) return;
      $$('button', seg).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      $$('[data-gk]').forEach(x => x.hidden = x.dataset.gk !== b.dataset.k);
      const at = vid.currentTime || 0; vid.pause();
      render(b.dataset.k, at);
    });
  })();

  (function diaZoom() {
    const ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
    $$('.mfig').forEach(fig => {
      const d = $('.dia', fig), cap = $('figcaption', fig); if (!d || !cap) return;
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'dia-zoom'; b.innerHTML = ICON + 'Enlarge the diagram';
      fig.insertBefore(b, cap);
      b.addEventListener('click', () => {
        const ratio = d.offsetHeight / Math.max(1, d.offsetWidth);
        const mark = document.createComment('dia'); d.parentNode.insertBefore(mark, d);
        const ov = document.createElement('div'); ov.className = 'dz-ov';
        ov.innerHTML = '<div class="dz-in"></div><button type="button" class="dz-close" aria-label="Close">×</button>';
        document.body.appendChild(ov);
        const inn = $('.dz-in', ov); inn.appendChild(d); d.classList.add('dz');
        let busy = false;
        const size = () => {
          const land = innerWidth > innerHeight, w = land ? innerWidth : innerHeight, h = land ? innerHeight : innerWidth;
          inn.style.width = w + 'px'; inn.style.height = h + 'px';
          inn.style.transform = 'translate(-50%,-50%)' + (land ? '' : ' rotate(90deg)');
          d.style.width = Math.min(w - 56, (h - 40) / ratio) + 'px';
          busy = true; window.dispatchEvent(new Event('resize')); busy = false;
        };
        const onR = () => { if (!busy) size(); };
        size(); window.addEventListener('resize', onR);
        document.documentElement.classList.add('dz-lock');
        const close = () => {
          window.removeEventListener('resize', onR);
          d.classList.toggle('dz', false); d.style.width = '';
          mark.parentNode.insertBefore(d, mark); mark.parentNode.removeChild(mark); ov.parentNode.removeChild(ov);
          document.documentElement.classList.toggle('dz-lock', false);
          window.dispatchEvent(new Event('resize'));
          b.focus({ preventScroll: true });
        };
        $('.dz-close', ov).addEventListener('click', close);
        ov.addEventListener('click', e => { if (e.target === ov || e.target === inn) close(); });
        document.addEventListener('keydown', function esc(e) { if (e.key === 'Escape' && ov.isConnected) { close(); document.removeEventListener('keydown', esc); } });
        $('.dz-close', ov).focus({ preventScroll: true });
      });
    });
  })();

  (function fpsChart() {
    const host = $('#fpsChart');
    const rows = [
      { g: 'LongLive-2.0', s: '6 B200 (2 without the agent)', bars: [['generator alone', 34.1, 'var(--base)', '2 B200, 7.43 s per 10.5 s shot'], ['with our agent', 34.3, 'var(--g)', '6 B200 (1 generator, 2 decoding and prefill, 3 agent), live, 7.37 s per 10.5 s shot']] },
      { g: 'LongLive-H3', s: '8 B200 (5 without the agent)', bars: [['generator alone', 38.4, 'var(--base)', '5 B200, 5.00 s per 8 s shot'], ['with our agent', 34.7, 'var(--g)', '8 B200 (4 generator, 1 decoder, 3 agent), 5.53 s per 8 s shot (paper)']] },
      { g: 'FastH3-Omni', s: '8 + 3 B200 (8 without the agent)', bars: [['generator alone', 33.4, 'var(--base)', '8 B200, no memory pictures, 5.75 s per 8 s shot'], ['with our agent', 29.0, 'var(--g)', '8 B200 generator + 3 B200 agent on a second node, live, 6.63 s per 8 s shot']] },
      { g: 'Agentic LongLive (Offline)', s: '4 H100 + Gemini, GPT Image APIs', bars: [['with the API agent', 0.33, 'var(--p)', '4 H100 for MiniMax-H3 Ref2VA (50 steps), 556 s + 25 s prompt rewrite per 8 s shot']] }];
    const isPh = () => host.clientWidth < 560;
    let PH = isPh();
    function draw() {
    const W = PH ? 400 : 760, BH = 16, GAP = 4, left = PH ? 10 : 210, right = PH ? 74 : 50, max = 40, TOP = PH ? 38 : 0;
    let y = 6; const pos = rows.map(r => { const y0 = y; y += TOP + (PH ? r.bars.length : Math.max(2, r.bars.length)) * (BH + GAP) + (PH ? 14 : 16); return y0; });
    const H = y + 22, x = v => left + (W - left - right) * v / max;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Frames per second of four systems with the memory agent">`;
    [0, 10, 20, 30, 40].forEach(v => s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="0" y2="${H - 22}"/><text class="ax" x="${x(v)}" y="${H - 6}" text-anchor="middle">${v}</text>`);
    rows.forEach((r, j) => {
      const y0 = pos[j];
      s += `<text class="lbl" x="${PH ? left : 0}" y="${y0 + 13}">${r.g}</text><text class="lbl2" x="${PH ? left : 0}" y="${y0 + (PH ? 29 : 30)}">${r.s}</text>`;
      r.bars.forEach(([lab, v, col, det], q) => {
        const yy = y0 + TOP + q * (BH + GAP), xe = Math.max(x(v), x(0) + 5);
        s += `<path class="bar" data-tip="${esc(r.g)}|${esc(lab)}: ${v < 1 ? v.toFixed(2) : v.toFixed(1)} FPS|${esc(det)}" d="M ${x(0)} ${yy} H ${xe - 4} a4 4 0 0 1 4 4 V ${yy + BH - 4} a4 4 0 0 1 -4 4 H ${x(0)} Z" fill="${col}"/><text class="axl" x="${xe + 6}" y="${yy + 12}">${v < 1 ? v.toFixed(2) + ' · not real time' : v.toFixed(1)}</text>`;
      });
    });
    s += `<line x1="${x(24)}" x2="${x(24)}" y1="0" y2="${H - 22}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="4 3"/><text class="axl" x="${x(24) + 5}" y="${H - 26}">real time (24)</text>`;
    s += `</svg>`;
    host.innerHTML = s;
    }
    draw();
    window.addEventListener('resize', () => { if (isPh() !== PH) { PH = isPh(); draw(); } });
    const tip = tipFor(host);
    const show = e => {
      const p = e.target.closest('[data-tip]'); if (!p) { tip.style.opacity = 0; return; }
      const [a, b, c] = p.dataset.tip.split('|'), rc = host.getBoundingClientRect();
      tip.innerHTML = `<b>${a}</b><br><span class="sw" style="background:${p.getAttribute('fill')}"></span>${b}<br>${c}`;
      tip.style.left = Math.max(0, Math.min(rc.width - 270, e.clientX - rc.left + 12)) + 'px'; tip.style.top = (e.clientY - rc.top - 10) + 'px'; tip.style.opacity = 1;
    };
    host.addEventListener('mousemove', show);
    host.addEventListener('pointerdown', show);
    host.addEventListener('mouseleave', () => tip.style.opacity = 0);
  })();

  function lineChart(host, key) {
    const cfg = { identity: { lo: 0, hi: 100, ticks: [0, 25, 50, 75, 100], f: v => v.toFixed(0), unit: 'protagonist identity (%)' }, imaging: { lo: .56, hi: .74, ticks: [.56, .62, .68, .74], f: v => v.toFixed(2), unit: 'imaging quality (VBench)' } }[key];
    const data = D.longhorizon[key];
    const PH = host.clientWidth < 560, W = PH ? 400 : 520, H = PH ? 260 : 250, L = 36, R = 44, T = 18, B = 28;
    const x = m => L + (W - L - R) * m / 14, y = v => T + (H - T - B) * (1 - (v - cfg.lo) / (cfg.hi - cfg.lo));
    const ser = [['longlive2', 'var(--base)', 'LongLive-2.0'], ['ours', 'var(--g)', 'with our memory']];
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${cfg.unit} over 14 minutes">`;
    cfg.ticks.forEach(v => s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${cfg.f(v)}</text>`);
    [0, 2, 4, 6, 8, 10, 12, 14].forEach(m => s += `<text class="ax" x="${x(m)}" y="${H - 8}" text-anchor="middle">${m}</text>`);
    s += `<text class="ax" x="${W - R + 30}" y="${H - 8}" text-anchor="end">min</text><text class="ax" x="${L}" y="10">${cfg.unit}</text>`;
    ser.forEach(([k, col]) => { const rows = data[k]; s += `<polygon class="band" points="${rows.map(r => `${x(r[0])},${y(r[3])}`).join(' ')} ${rows.slice().reverse().map(r => `${x(r[0])},${y(r[2])}`).join(' ')}" fill="${col}"/>`; });
    ser.forEach(([k, col]) => {
      const rows = data[k], last = rows[rows.length - 1];
      s += `<polyline class="ln" points="${rows.map(r => `${x(r[0])},${y(r[1])}`).join(' ')}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
      s += `<circle class="endp" cx="${x(last[0])}" cy="${y(last[1])}" r="4" fill="${col}" stroke="var(--bg)" stroke-width="2"/><text class="axl endp" x="${x(last[0]) + 8}" y="${y(last[1]) + 4}">${cfg.f(last[1])}</text>`;
    });
    s += `<line class="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--ink)" stroke-width="1" opacity="0"/>`;
    ser.forEach(([k, col]) => s += `<circle class="hd" data-k="${k}" r="4.5" fill="${col}" stroke="var(--bg)" stroke-width="2" opacity="0"/>`);
    s += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg>`;
    host.innerHTML = s;
    const lns = $$('.ln', host);
    lns.forEach(l => { const n = l.getTotalLength(); l.style.strokeDasharray = n; l.style.strokeDashoffset = host.classList.contains('in') ? 0 : n; });
    if (host.classList.contains('in')) { lns.forEach(l => { const n = l.getTotalLength(); l.style.transition = 'none'; l.style.strokeDashoffset = n; l.getBoundingClientRect(); l.style.transition = ''; l.style.strokeDashoffset = 0; }); }
    const svg = $('svg', host), xh = $('.xh', svg), hd = $$('.hd', svg), tip = tipFor(host);
    const show = e => {
      const rc = svg.getBoundingClientRect(), sx = (e.clientX - rc.left) * W / rc.width;
      const m = clamp(Math.round(((sx - L) / (W - L - R)) * 14 - .5), 0, 13), xs = x(m + .5);
      xh.setAttribute('x1', xs); xh.setAttribute('x2', xs); xh.setAttribute('opacity', .35);
      let html = `<b>minute ${m}–${m + 1}</b>`;
      ser.slice().reverse().forEach(([k, col, name]) => {
        const r = data[k][m], c = hd.find(q => q.dataset.k === k);
        c.setAttribute('cx', xs); c.setAttribute('cy', y(r[1])); c.setAttribute('opacity', 1);
        html += `<br><span class="sw" style="background:${col}"></span>${name}: ${cfg.f(r[1])}`;
      });
      tip.innerHTML = html;
      tip.style.left = Math.max(0, Math.min(rc.width - 180, (xs / W) * rc.width + 12)) + 'px'; tip.style.top = '20px'; tip.style.opacity = 1;
    };
    svg.addEventListener('mousemove', show);
    svg.addEventListener('pointerdown', show);
    svg.addEventListener('mouseleave', () => { xh.setAttribute('opacity', 0); hd.forEach(q => q.setAttribute('opacity', 0)); tip.style.opacity = 0; });
  }
  lineChart($('#lhChart'), 'identity');
  {
    let ph = $('#lhChart').clientWidth < 560;
    window.addEventListener('resize', () => {
      const now = $('#lhChart').clientWidth < 560; if (now === ph) return; ph = now;
      const b = $('#lhSeg button[aria-pressed="true"]'); lineChart($('#lhChart'), b ? b.dataset.k : 'identity');
    });
  }
  $('#lhSeg').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b || b.getAttribute('aria-pressed') === 'true') return;
    $$('#lhSeg button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    lineChart($('#lhChart'), b.dataset.k);
  });

  (function chartReveal() {
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return; io.unobserve(e.target);
      e.target.classList.add('in'); $$('.ln', e.target).forEach(l => l.style.strokeDashoffset = 0);
    }), { threshold: .35 });
    $$('.chart').forEach(c => io.observe(c));
  })();

  $('#copyBib').addEventListener('click', () => {
    const txt = $('#bibtex').textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(() => { $('#copyBib').textContent = 'Copied'; setTimeout(() => $('#copyBib').textContent = 'Copy', 1500); }).catch(() => {});
  });
})();
