(function () {
  'use strict';
  const RL = window.AGL_RL, hosts = Array.from(document.querySelectorAll('#rlList, [data-rl]'));
  if (!RL || !hosts.length) return;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const RM = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const HOVER = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  const VH = '';
  const mmss = t => { t = Math.max(0, Math.floor(t)); return Math.floor(t / 60) + ':' + String(t % 60).padStart(2, '0'); };
  const PLAY = '<svg class="i1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><svg class="i2" viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
  const RATES = [1, 1.5, 2];
  let RATE = 2;
  try { const r = parseFloat(localStorage.getItem('agl-rl-rate')); if (RATES.includes(r)) RATE = r; } catch (e) {  }
  const cards = [];
  function setRate(r) {
    RATE = r;
    try { localStorage.setItem('agl-rl-rate', String(r)); } catch (e) {  }
    cards.forEach(c => c.rate(r));
  }

  function card(R, host) {
    const T = R.t, N = R.instr.length, D = T[N];
    const pct = x => (100 * x / D).toFixed(3) + '%';
    const el = document.createElement('article');
    el.className = 'rl-card reveal';
    el.id = document.getElementById('rl-' + R.id) ? 'rl-' + R.id + '-' + host.id : 'rl-' + R.id;
    const vbox = a => `<div class="vwrap rl-v ${a}"><video playsinline muted preload="none" poster="${R['poster_' + a]}" data-src="${VH + R['video_' + a]}"></video>
        <span class="rl-vl">${a === 'ours' ? esc(R.sysl || 'Agentic LongLive (Offline)') : 'Video model alone'}</span><span class="rl-gen">Generated</span>
        ${a === 'ours' ? '<div class="rl-in" aria-hidden="true"><span>memory in</span><div class="ims"></div></div>' : ''}
        <span class="rl-chip"></span>
        ${a === 'ours' ? `<div class="vctl"><button class="vplay" aria-label="Play or pause">${PLAY}</button></div>` : ''}</div>`;
    el.innerHTML = `
      <div class="rl-head">
        <div class="rl-ttl"><h4>${esc(R.title)}</h4><p>${esc(R.hook)}</p></div>
        <div class="seg2 rl-speed" role="group" aria-label="Playback speed">${RATES.map(r => `<button data-r="${r}" aria-pressed="${r === RATE}">${r}×</button>`).join('')}</div>
      </div>
      <div class="rl-vids">${vbox('base')}${vbox('ours')}</div>
      <div class="rl-tl">
        <div class="rl-gap" style="left:${pct(R.gap.a)};width:${pct(R.gap.b - R.gap.a)}"><span>${esc(R.gap.label)}</span></div>
        <div class="segs rl-segs">${R.instr.map((s, k) => `<button class="${R.ret.includes(k) ? 'ret' : ''}" style="flex:${T[k + 1] - T[k]}" aria-label="Instruction ${k + 1}: ${esc(s)}"><span class="f"></span></button>`).join('')}</div>
        <div class="seglab rl-lab">${R.short.map((s, k) => `<span class="${R.ret.includes(k) ? 'ret' : R.hide.includes(k) ? 'hid' : ''}" style="flex:${T[k + 1] - T[k]}">${k + 1}&ensp;${esc(s)}</span>`).join('')}</div>
      </div>
      <div class="rl-now"><span class="pk"></span><span class="tx"></span></div>
      <div class="rl-tabs"><div class="rl-tabb" role="tablist"><button data-v="hl" role="tab" aria-selected="true">What comes back</button><button data-v="mem" role="tab" aria-selected="false">Agent memory</button></div>
        <span class="rl-hint">${HOVER ? 'Hover over the videos to watch the agent’s memory.' : 'Tap Agent memory to see what the agent keeps.'}</span></div>
      <div class="rl-bottom">
        <div class="rl-hl">${R.detail.map((d, j) => `<figure class="rl-d ${j === 0 ? 'before' : d.src}" data-t="${d.t}"><div class="ph"><img src="${d.img}" alt="${esc(d.sub)}" loading="lazy" decoding="async"><em>${d.when}</em></div>
          <figcaption><b>${esc(d.cap)}</b>${esc(d.sub)}</figcaption></figure>`).join('')}</div>
        <div class="rl-mem">
          <div class="rl-c notes"><h5>What it wrote down <span class="n"></span></h5><div class="rows"></div></div>
          <div class="rl-c pics"><h5>Scene pictures <span class="n1"></span></h5><div class="strip scene"></div><h5>Close-ups <span class="n2"></span></h5><div class="strip close"></div></div>
          <div class="rl-c plan"><h5 class="ph5"></h5><p class="q"></p><div class="pins"></div><p class="pr"></p></div>
        </div>
      </div>`;
    host.appendChild(el);

    const vb = $('.base video', el), vo = $('.ours video', el), vids = [vb, vo];
    const wb = $('.rl-v.base', el), wo = $('.rl-v.ours', el), cb = $('.base .rl-chip', el), co = $('.ours .rl-chip', el);
    const inBox = $('.rl-in', el), inIms = $('.rl-in .ims', el), bp = $('.vplay', el);
    const segBtns = $$('.rl-segs button', el), pk = $('.rl-now .pk', el), tx = $('.rl-now .tx', el);
    const rows = $('.notes .rows', el), nN = $('.notes .n', el), sc = $('.strip.scene', el), cl = $('.strip.close', el), n1 = $('.n1', el), n2 = $('.n2', el);
    const ph5 = $('.ph5', el), q = $('.plan .q', el), pins = $('.plan .pins', el), pr = $('.plan .pr', el);
    const tabs = $$('.rl-tabb button', el);
    const P = R.pics;

    let pinnedTab = 'hl', hov = false;
    function showMem() {
      const on = hov || pinnedTab === 'mem';
      el.classList.toggle('rl-on', on);
      tabs.forEach(b => b.setAttribute('aria-selected', String((b.dataset.v === 'mem') === on)));
    }
    tabs.forEach(b => b.addEventListener('click', () => { pinnedTab = b.dataset.v; hov = false; showMem(); }));
    if (HOVER) {
      $('.rl-vids', el).addEventListener('mouseenter', () => { hov = true; showMem(); });
      el.addEventListener('mouseleave', () => { hov = false; showMem(); });
    }

    const thumb = (pid, cls) => { const p = P[pid]; return `<figure class="th ${cls || ''}" title="${esc(p.label + (p.sub ? ' · ' + p.sub : ''))}"><img src="${p.img}" alt="${esc(p.label)}" loading="lazy" decoding="async"></figure>`; };
    function renderMem(k) {
      const S = R.segs[k];
      const order = S.notes.slice().sort((a, b) => (a.inview - b.inview) || (a.seen - b.seen));
      const top = order.slice(0, 5);
      const upto = S.ob == null ? '' : S.ob < 0 ? 'from the input image' : `has watched up to instruction ${S.ob + 1}`;
      nN.textContent = [S.notes.length ? S.notes.length + (S.notes.length === 1 ? ' thing' : ' things') : '', upto].filter(Boolean).join(' · ');
      rows.innerHTML = top.map(r => `<div class="row${r.inview ? '' : ' away'}">
          <div class="l1"><b>${esc(r.name)}</b><span class="st" data-seen="${r.seen_t}" data-in="${r.inview ? 1 : 0}" data-hid="${r.hidden ? 1 : 0}"></span></div>
          <div class="l2" title="${esc([r.where, r.look, r.cont].filter(Boolean).join(' · '))}">${r.inview || !r.where ? '' : '<i>last seen</i> '}${esc([r.where.replace(' (hidden)', ''), r.look, r.cont].filter(Boolean).join(' · '))}</div></div>`).join('')
        + (order.length > top.length ? `<div class="more">+${order.length - top.length} more in view</div>` : '')
        + (S.notes.length ? '' : '<div class="more">It looks at the input image before the first instruction.</div>');
      const pinSet = new Set(S.pins), fresh = pid => P[pid].seg === (S.ob == null ? k - 1 : S.ob);
      n1.textContent = S.scene.length || '';
      sc.innerHTML = S.scene.map(pid => thumb(pid, (pinSet.has(pid) ? 'pin' : '') + (fresh(pid) ? ' new' : ''))).join('');
      const pinnedClose = S.close.filter(pid => pinSet.has(pid)), others = S.close.filter(pid => !pinSet.has(pid));
      const keep = new Set(pinnedClose.concat(others.slice(Math.max(0, others.length - (5 - pinnedClose.length)))));
      const closeShown = S.close.filter(pid => keep.has(pid));
      n2.textContent = S.close.length || '';
      cl.innerHTML = closeShown.length ? closeShown.map(pid => thumb(pid, (pinSet.has(pid) ? 'pin' : '') + (fresh(pid) ? ' new' : ''))).join('') : '<span class="none">None yet</span>';
      ph5.textContent = 'Instruction ' + (k + 1);
      const shT = Math.round(R.shared_t || 8);
      if (!S.planned) {
        q.innerHTML = '<b>The first instruction goes in as written.</b> Both videos share ' + (shT > 8 ? `the first ${shT}&#8239;s.` : 'these first 8&#8239;s.');
        pins.innerHTML = '';
      } else if (S.ret) {
        q.innerHTML = `<b>Does anything come back? Yes:</b> ${esc(S.what)}. It hands ${S.pins.length === 1 ? 'this picture' : 'these pictures'} to the generator.`;
        pins.innerHTML = S.pins.map(pid => `<figure><img src="${P[pid].img}" alt="${esc(P[pid].label)}"><figcaption>${esc(P[pid].label)}</figcaption></figure>`).join('');
      } else {
        q.innerHTML = '<b>Does anything come back? No.</b> No pictures go in. It only rewrites the instruction with the names and looks it wrote down.';
        pins.innerHTML = '';
      }
      if (S.planned && S.shared) q.innerHTML += ` Both videos share the first ${shT}&#8239;s, so the video model alone shows the same part.`;
      pr.innerHTML = `<span>Prompt it sends</span>${esc(S.prompt)}`;
      pr.title = S.prompt;
      inIms.innerHTML = S.pins.map(pid => `<img src="${P[pid].img}" alt="">`).join('');
      inBox.classList.toggle('on', S.pins.length > 0);
    }
    function tickAgo(t) {
      $$('.st', rows).forEach(s => {
        if (s.dataset.in === '1') { s.textContent = 'in view'; return; }
        const ago = Math.max(1, Math.round(t - parseFloat(s.dataset.seen)));
        s.textContent = (s.dataset.hid === '1' ? 'hidden, ' : 'out of sight, ') + 'seen ' + ago + ' s ago';
      });
    }

    let k0 = -1, raf = 0, inView = false, userPaused = false, loaded = false;
    const load = () => { if (loaded) return; loaded = true; vids.forEach(v => { v.src = v.dataset.src; v.preload = 'auto'; v.playbackRate = RATE; }); };
    const play = () => { load(); vids.forEach(v => { v.playbackRate = RATE; const p = v.play(); if (p) p.catch(() => {}); }); };
    const pause = () => vids.forEach(v => v.pause());
    const seek = t => { load(); vids.forEach(v => { v.currentTime = t; }); onTime(t); };
    function onTime(t) {
      segBtns.forEach((b, j) => {
        const u = Math.max(0, Math.min(1, (t - T[j]) / (T[j + 1] - T[j])));
        b.classList.toggle('done', u >= 1); b.firstChild.style.width = u >= 1 ? '' : (u * 100) + '%';
      });
      if (!vo.paused && !vb.seeking && loaded && Math.abs(vb.currentTime - t) > .15) vb.currentTime = t;
      if (!vo.paused && vb.paused && !vb.ended && loaded) { const p = vb.play(); if (p) p.catch(() => {}); }
      const c = R.callouts.find(x => t >= x.a && t < x.b);
      wo.classList.toggle('ok', !!c); wb.classList.toggle('bad', !!c);
      if (c) { if (co.dataset.k !== c.ours) { co.dataset.k = c.ours; co.textContent = '✓ ' + c.ours; cb.textContent = '✕ ' + c.base; } }
      let k = 0; while (k + 1 < N && t >= T[k + 1]) k++;
      if (k !== k0) {
        k0 = k;
        pk.textContent = 'Instruction ' + (k + 1);
        tx.textContent = R.instr[k];
        renderMem(k);
      }
      tickAgo(t);
    }
    const loop = () => { onTime(vo.currentTime); raf = vo.paused ? 0 : requestAnimationFrame(loop); };
    const upd = () => bp.classList.toggle('alt', !vo.paused);
    vo.addEventListener('play', () => { upd(); if (!raf) raf = requestAnimationFrame(loop); });
    vo.addEventListener('pause', () => { upd(); if (!vb.paused) vb.pause(); });
    vo.addEventListener('seeked', () => onTime(vo.currentTime));
    vo.addEventListener('timeupdate', () => { if (!raf) onTime(vo.currentTime); });
    vo.addEventListener('ended', () => { seek(0); if (inView && !userPaused) play(); });
    const toggle = () => { if (vo.paused) { userPaused = false; play(); } else { userPaused = true; pause(); } };
    bp.addEventListener('click', e => { e.stopPropagation(); toggle(); });
    $$('.rl-v', el).forEach(w => w.addEventListener('click', toggle));
    segBtns.forEach((b, j) => b.addEventListener('click', () => { seek(T[j] + .05); userPaused = false; play(); }));
    $$('.rl-d', el).forEach(f => f.addEventListener('click', () => { seek(Math.max(0, parseFloat(f.dataset.t) - 2)); userPaused = false; play(); }));
    $$('.rl-speed button', el).forEach(b => b.addEventListener('click', () => setRate(parseFloat(b.dataset.r))));

    new IntersectionObserver(es => { if (es[0].isIntersecting) load(); }, { rootMargin: '700px 0px' }).observe(el);
    new IntersectionObserver(es => {
      inView = es[0].isIntersecting;
      if (inView && !RM && !userPaused) play(); else if (!inView) pause();
    }, { threshold: .45 }).observe($('.rl-vids', el));
    onTime(0);
    return {
      rate(r) { vids.forEach(v => { v.playbackRate = r; }); $$('.rl-speed button', el).forEach(b => b.setAttribute('aria-pressed', String(parseFloat(b.dataset.r) === r))); }
    };
  }

  const claimed = new Set(hosts.flatMap(h => h.dataset.rl ? h.dataset.rl.split(',') : []));
  hosts.forEach(host => {
    const only = host.dataset.rl ? host.dataset.rl.split(',') : null;
    RL.filter(R => only ? only.includes(R.id) : !claimed.has(R.id)).forEach(R => cards.push(card(R, host)));
  });
})();
