'use strict';
const C = window.CATALOG, S = window.STORY;
const $ = id => document.getElementById(id);
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const state = { paused: reduced.matches, query: '', limits: {wan21:6,wan22:6,h3:6} };
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const familyName = id => ({wan21:'Wan2.1 · 14B',wan22:'Wan2.2 · TI2V-5B',h3:'MiniMax-H3'}[id]);
const groups = new Set();
const modal = $('case-dialog');
let currentDialog = null, dialogGroup = null, opener = null;
const canPlay = () => !state.paused && !document.hidden && !modal.open;
function video(a, extra='') { return `<video muted playsinline preload="none" poster="${esc(a.poster)}" data-src="${esc(a.src)}" ${extra}></video>`; }
function tile(a,label,detail='',featured=false,caption='') {return `<div class="video-tile ${featured?'featured':''}"><div class="tile-label"><span>${esc(label)}</span><small>${esc(detail)}</small></div>${video(a)}${caption?`<p class="tile-caption">${esc(caption)}</p>`:''}</div>`;}
function media(c,ours=true){return {src:ours?c.ours:c.native,poster:ours?c.poster:c.nativePoster};}
function fmt(s){return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;}
function tabbar(root,items,select,initial=0){root.innerHTML='';items.forEach((item,i)=>{const b=document.createElement('button');b.type='button';b.textContent=item.title;b.setAttribute('aria-pressed',i===initial);b.onclick=()=>{root.querySelectorAll('button').forEach((x,j)=>x.setAttribute('aria-pressed',j===i));select(i);};root.append(b);});}
function attachGroup(root, {sync=true,dialog=false}={}){
 const g={root,sync,dialog,visible:dialog,paused:false,videos:[...root.querySelectorAll('video')],lastCheck:0,coverage:root.classList.contains('coverage-card'),revealRemaining:1200,cycle:0};groups.add(g);root._group=g;
 for(const v of g.videos){v.muted=true;v.loop=!sync;v.addEventListener('loadeddata',()=>refreshGroup(g));v.addEventListener('error',()=>{if(dialog)$('pair-status').textContent='This recording could not load. Extract the complete ZIP before opening index.html.';});}
 if(dialog){loadGroup(g);refreshGroup(g);}else groupObserver.observe(root);
 return g;
}
const groupObserver=new IntersectionObserver(entries=>entries.forEach(({target,isIntersecting})=>{const g=target._group;if(!g)return;g.visible=isIntersecting;if(isIntersecting){loadGroup(g);if(target.classList.contains('transfer-stage'))target.classList.add('transfer-ready');}refreshGroup(g);if(isIntersecting&&target.classList.contains('coverage-card'))revealCard(target);}),{threshold:.12});
function loadGroup(g){g.videos.forEach(v=>{if(!v.getAttribute('src')){v.src=v.dataset.src;v.load();}});}
function playingAllowed(g){return g.visible&&!g.paused&&!document.hidden&&(g.dialog||canPlay());}
function refreshGroup(g){
 const allowed=playingAllowed(g);const ready=!g.sync||g.videos.every(v=>v.readyState>=2);
 g.videos.forEach(v=>{if(allowed&&ready&&v.paused)v.play().catch(()=>{});else if(!allowed||!ready)v.pause();});
}
function destroyGroup(root){const g=root?._group;if(!g)return;groupObserver.unobserve(root);g.videos.forEach(v=>{v.pause();v.removeAttribute('src');v.load();});groups.delete(g);delete root._group;}
function destroyWithin(root){root.querySelectorAll('.cfg-case-stage').forEach(stage=>stage._releaseGuidanceFlow?.());[...groups].filter(g=>g.root===root||root.contains(g.root)).forEach(g=>destroyGroup(g.root));}
function refreshAll(){groups.forEach(refreshGroup);document.body.classList.toggle('paused',state.paused);updateMotion();}
function switchVideo(v,a,t=0){v.pause();v.poster=a.poster;v.dataset.src=a.src;v.src=a.src;v.onloadedmetadata=()=>{try{v.currentTime=Math.min(t,Math.max(0,v.duration-.1));}catch{}};v.load();}
let previousFrameTime=0;
function syncGroups(now){
 const dt=Math.min(100,Math.max(0,now-(previousFrameTime||now)));previousFrameTime=now;
 for(const g of groups){
  if(!g.sync||!playingAllowed(g)||g.videos.some(v=>v.readyState<2))continue;
  const [first,...rest]=g.videos,duration=Math.min(...g.videos.map(v=>v.duration));
  if(g.coverage&&g.revealRemaining>0){g.revealRemaining=Math.max(0,g.revealRemaining-dt);if(!g.revealRemaining)g.root.classList.add('revealed');}
  if(first.currentTime>=duration-.055||g.videos.some(v=>v.ended)){
   if(g.coverage)restartCardInsertion(g);
   g.videos.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});
  }else if(now-g.lastCheck>350){rest.forEach(v=>{if(!v.seeking&&!first.seeking&&Math.abs(v.currentTime-first.currentTime)>.16)v.currentTime=first.currentTime;});g.lastCheck=now;}
 }
 if(dialogGroup&&modal.open){const v=dialogGroup.videos[0],d=Math.min(...dialogGroup.videos.map(v=>v.duration));if(Number.isFinite(d)){if(!$('pair-seek').matches(':active'))$('pair-seek').value=String(v.currentTime/d*1000);$('pair-time').textContent=`${fmt(v.currentTime)} / ${fmt(d)}`;}}
 updateOverview(now);requestAnimationFrame(syncGroups);
}
function updateMotion(){const label=state.paused?'▷ Resume motion':'Ⅱ Pause motion';$('motion-toggle').innerHTML=`${state.paused?'▷':'Ⅱ'} <span>${state.paused?'Resume motion':'Pause motion'}</span>`;$('motion-toggle').setAttribute('aria-pressed',state.paused);const m=$('mobile-motion');if(m){m.textContent=state.paused?'▷':'Ⅱ';m.setAttribute('aria-label',label);m.setAttribute('aria-pressed',state.paused);}}
$('motion-toggle').onclick=()=>{state.paused=!state.paused;refreshAll();};
const mobileMotion=document.createElement('button');mobileMotion.id='mobile-motion';mobileMotion.className='mobile-motion';mobileMotion.onclick=$('motion-toggle').onclick;document.querySelector('.sidebar').append(mobileMotion);

// Figure 1: each task has its own data collection and training run.
const ovTaskTraining=[{start:4.6,duration:1.6},{start:4.9,duration:3},{start:5.2,duration:5.6}];
const ovTiming={developed:2.8,dataReady:4.6,trainingDuration:1.6,adapterHold:.85,flightDuration:1.05,transferStagger:.16,playWindow:6.6,hold:1.2};
ovTiming.adapters=ovTiming.trainingDuration;
ovTiming.flightStart=ovTiming.adapters+ovTiming.adapterHold;
ovTiming.transfer=ovTiming.flightStart+ovTiming.flightDuration;
ovTiming.distilled=Math.max(...ovTaskTraining.map(task=>task.start+task.duration));
const overview={visible:false,time:0,last:0,cycle:Math.max(ovTiming.distilled,ovTiming.transfer+ovTiming.transferStagger+ovTiming.playWindow)+ovTiming.hold,sound:false,flights:[]};
const ovMedia=()=>[...$('overview-animation').querySelectorAll('video')];
const ovProgress=(bar,progress)=>{const p=Math.max(0,Math.min(1,progress));bar.style.setProperty('--training-progress',p);bar.setAttribute('aria-valuenow',Math.floor(p*100));return p;};
function overviewAux(kind,arm){
 if(kind==='control')return `<div class="ov-conditioning" aria-label="Depth input">${video(S.overviewDepth,'data-depth="true"')}</div>`;
 if(kind==='nava')return `<div class="ov-audio" aria-label="Generated audio"><img src="story/nava-waveform-${arm}.svg" alt="Generated speech waveform (${arm})"></div>`;
 if(kind==='alpha')return '';
 return '';
}
function resetOverview(){
 overview.time=0;overview.last=performance.now();
 ovMedia().forEach(v=>{v.pause();if(v.readyState)v.currentTime=v.dataset.native?Math.min(1,v.duration-.1):0;});
}
function initOverviewFlights(){
 const lane=document.querySelector('#overview-animation .plug-route'),layer=$('overview-flight-layer');
 const sources=[...$('overview-plugs').children],targets=[...$('parallel-tasks').children];
 const types=['step','cfg','long'];
 targets.forEach((target,i)=>sources.forEach((source,j)=>{
  const destination=target.querySelector(`.ov-adapter-dock .${types[j]}`);
  if(!destination)return;
  const token=document.createElement('span');token.className=`lora-flight ${types[j]}`;token.textContent='+';layer.append(token);
  overview.flights.push({token,source,target:destination,delay:Math.floor(i/3)*ovTiming.transferStagger});
 }));
 const measure=()=>{
  const box=lane.getBoundingClientRect();
  overview.flights.forEach(f=>{const a=f.source.getBoundingClientRect(),b=f.target.getBoundingClientRect();f.from={x:a.left+a.width/2-box.left,y:a.top+a.height/2-box.top};f.to={x:b.left+b.width/2-box.left,y:b.top+b.height/2-box.top};});
 };
 const observer=new ResizeObserver(measure);[lane,$('overview-plugs'),$('parallel-tasks')].forEach(el=>observer.observe(el));measure();
}
function initOverview(){
 const alpha=C.models.find(m=>m.id==='wan21-wan-alpha-v1-v2'),nava=C.models.find(m=>m.id==='wan22-nava'),control=S.comparison.find(c=>c.id==='control');
 const picks=[{title:'Transparent video',task:'RGBA output',kind:'alpha',native:media(alpha.cases[alpha.selectedCase],false),ours:media(alpha.cases[alpha.selectedCase])},{title:'Audio-video',task:'Video + speech',kind:'nava',native:media(nava.cases[nava.selectedCase],false),ours:media(nava.cases[nava.selectedCase])},{title:'Video control',task:'Depth conditioning',kind:'control',native:control.variants[0],ours:control.variants[3]}];
 picks.forEach((p,i)=>{
  const heading=`<div class="ov-task-heading"><strong>${p.title}</strong></div>`;
  $('serial-tasks').insertAdjacentHTML('beforeend',`<div class="serial-row" data-index="${i}">
   <div class="ov-downstream-node">${heading}<div class="serial-output"><div class="serial-pending"><span class="pending-number">↓</span></div><div class="ov-task-media">${video(p.native,'data-native="true" aria-label="Native downstream task preview"'+(p.kind==='nava'?' data-nava="true"':''))}${overviewAux(p.kind,'native')}</div></div></div>
   <div class="task-training"><div class="ov-branch-arrow" aria-hidden="true">↓</div><div class="data-operation"><strong>Data</strong><em aria-hidden="true"></em></div><div class="train-operation"><strong>Training</strong><em aria-hidden="true"></em></div><div class="ov-training-progress task-progress" role="progressbar" aria-label="${esc(p.title)} distillation progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span></span></div><div class="ov-branch-arrow end-arrow" aria-hidden="true">↓</div></div>
   <div class="serial-result"><strong>Distilled model</strong></div>
  </div>`);
 });
 const robot=C.models.find(m=>m.id==='wan21-abot-physworld'),motion=C.models.find(m=>m.id==='wan22-flashmotion'),matrix=C.models.find(m=>m.id==='wan22-matrix-game-3-0'),autumn=S.scope.find(c=>c.title==='Autumn Temple');
 const downstream=[picks[0],{title:'Interactive world',kind:'matrix',longContext:true,ours:media(matrix.cases[matrix.selectedCase])},picks[2],{title:'Robot manipulation',task:'Video prediction',kind:'robot',ours:media(robot.cases[robot.selectedCase])},{title:'FPS world model',task:'Autumn Temple',kind:'world',longContext:true,ours:autumn.variants[1]},{title:'Motion control',task:'Guided trajectories',kind:'motion',ours:media(motion.cases[motion.selectedCase])}];
 downstream.forEach((p,i)=>{
  $('parallel-tasks').insertAdjacentHTML('beforeend',`<div class="parallel-task" data-kind="${p.kind}" data-index="${i}"><div class="ov-task-heading"><strong>${p.title}</strong></div><div class="ov-adapter-dock" aria-label="Transferred LoRA capabilities"><span class="step" title="Few-step LoRA"></span><span class="cfg" title="CFG LoRA"></span>${p.longContext?'<span class="long" title="Long-context LoRA"></span>':''}<b>✓</b></div><div class="parallel-output"><div class="output-wait" aria-label="Awaiting LoRA transfer"><span>+</span></div><div class="output-recording">${video(p.ours,p.kind==='nava'?'data-nava="true"':'')}${overviewAux(p.kind,'ours')}</div></div></div>`);
 });
 initOverviewFlights();
 const observer=new IntersectionObserver(entries=>{
  overview.visible=entries[0].isIntersecting;
  if(overview.visible){ovMedia().forEach(v=>{if(!v.getAttribute('src')){
   v.muted=true;if(v.dataset.depth)v.playbackRate=.8;
   if(v.dataset.native)v.addEventListener('loadeddata',()=>{if(v.paused&&overview.time<ovTiming.developed)try{v.currentTime=Math.min(1,v.duration-.1);}catch{}},{once:true});
   v.src=v.dataset.src;v.load();
  }});}else ovMedia().forEach(v=>v.pause());
 },{threshold:.06});observer.observe($('overview-animation'));
 $('overview-replay').onclick=()=>{resetOverview();if(state.paused){state.paused=false;refreshAll();}};
 $('overview-sound').onclick=()=>{
  overview.sound=!overview.sound;$('overview-sound').setAttribute('aria-pressed',overview.sound);$('overview-sound').textContent=overview.sound?'♪ Speech sound on':'♪ Speech sound off';
  document.querySelector('#serial-tasks [data-nava]').muted=!overview.sound;
  if(overview.sound&&(overview.time>=ovTiming.developed+ovTiming.playWindow||state.paused)){resetOverview();state.paused=false;refreshAll();}
 };
 if(reduced.matches)overview.time=overview.cycle;
}
function updateOverview(now){
 const dt=Math.min(.1,(now-(overview.last||now))/1000);overview.last=now;if(!overview.visible)return;const run=canPlay();
 if(run){overview.time+=dt;if(overview.time>overview.cycle)resetOverview();}
 const t=overview.time,developed=t>=ovTiming.developed,dataReady=t>=ovTiming.dataReady,done=t>=ovTiming.distilled;
 const collecting=developed&&!dataReady,training=dataReady&&!done;
 const trained=t>=ovTiming.adapters,outputs=t>=ovTiming.transfer+ovTiming.transferStagger;
 const root=$('overview-animation');
 const backboneProgress=ovProgress($('backbone-progress'),t/ovTiming.trainingDuration);
 root.classList.toggle('downstream-developed',developed);root.classList.toggle('adapters-trained',trained);root.classList.toggle('adapters-transferred',outputs);
 root.classList.toggle('adapters-in-flight',trained&&!outputs&&t>=ovTiming.flightStart);
 root.dataset.serialStage=done?'ready':training?'distilling':collecting?'collecting-data':'developing';
 root.dataset.parallelStage=outputs?'transferred':trained?'transferring':'distilling';
 $('overview-development').classList.toggle('complete',developed);
 $('development-status').textContent=developed?'✓ Downstream models ready':'Developing downstream models…';
 let finished=0;
 document.querySelectorAll('.serial-row').forEach((row,i)=>{
  const task=ovTaskTraining[i],ready=t>=task.start+task.duration,started=t>=task.start;
  const progress=ovProgress(row.querySelector('.task-progress'),(t-task.start)/task.duration);
  if(ready)finished++;
  row.classList.toggle('active',developed&&!ready);row.classList.toggle('complete',ready);
  row.dataset.stage=ready?'ready':started?'training':developed?'data':'developing';
  row.querySelector('.data-operation em').textContent=started?'✓':'';
  row.querySelector('.train-operation em').textContent=ready?'✓':'';
  row.querySelectorAll('video').forEach(v=>{
   if(v.dataset.nava)v.muted=!overview.sound;
   const play=run&&developed&&t<ovTiming.developed+ovTiming.playWindow;
   if(play&&!v.ended){if(v.paused)v.play().catch(()=>{});}else v.pause();
  });
 });
 $('overview-plugs').classList.toggle('available',trained);
 $('backbone-training').textContent=trained?'✓ Three LoRAs. Ready to reuse.':`Train reusable capabilities · ${Math.floor(backboneProgress*100)}%`;
 let received=0;
 document.querySelectorAll('.parallel-task').forEach((row,i)=>{
  const arrival=ovTiming.transfer+Math.floor(i/3)*ovTiming.transferStagger,ready=t>=arrival;
  if(ready)received++;row.classList.toggle('ready',ready);
  row.querySelectorAll('video').forEach(v=>{
   if(v.dataset.nava)v.muted=!overview.sound;
   const play=run&&ready&&t<arrival+ovTiming.playWindow;
   if(play&&!v.ended){if(v.paused)v.play().catch(()=>{});}else v.pause();
  });
 });
 overview.flights.forEach(f=>{
  const progress=(t-ovTiming.flightStart-f.delay)/ovTiming.flightDuration;
  const visible=progress>=0&&progress<1&&!!f.from;f.token.style.visibility=visible?'visible':'hidden';
  if(visible){
   const p=progress,x=f.from.x+(f.to.x-f.from.x)*(p*p*(3-2*p));
   const y=f.from.y+(f.to.y-f.from.y)*p*p-24*Math.sin(p*Math.PI);
   f.token.style.transform=`translate(${x}px,${y}px) translate(-50%,-50%) scale(${1-.26*p})`;
  }
 });
 $('parallel-count').textContent=`${received} / 6 ready`;$('serial-count').textContent=`${finished} / 3 ready`;
 root.dataset.phase=!trained?'backbone-training':t<ovTiming.flightStart?'adapters-ready':!outputs?'adapter-flight':t<ovTiming.transfer+ovTiming.transferStagger+ovTiming.playWindow?'parallel-output':!done?'parallel-wait':'hold';
}

// Connect the rendered nodes rather than relying on fixed CSS offsets.
function attachGuidanceFlow(stage){
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
 svg.classList.add('guidance-flow');svg.setAttribute('aria-hidden','true');stage.append(svg);
 let frame=0;
 function draw(){
  frame=0;if(!stage.isConnected)return;
  const box=stage.getBoundingClientRect();if(!box.width||!box.height)return;
  const relative=el=>{const r=el.getBoundingClientRect();return {x:r.left-box.left,y:r.top-box.top,w:r.width,h:r.height};};
  const source=relative(stage.querySelector('.cfg-insertion .transfer-adapters')||stage.querySelector('.cfg-insertion .lora-chip'));
  const native=relative(stage.querySelector('.cfg-native')),branches=relative(stage.querySelector('.cfg-branches'));
  const rows=[...stage.querySelectorAll('.cfg-branch')];
  const paths=[],heads=[];
  const rightArrow=(x1,y,x2)=>{paths.push(`M${x1},${y} H${x2}`);heads.push(`M${x2-7},${y-5} L${x2},${y} L${x2-7},${y+5}`);};
  if(branches.y>=native.y+native.h-1){
   const x=box.width/2,y1=native.y+native.h+3,y2=branches.y-6;
   paths.push(`M${x},${y1} V${y2}`);heads.push(`M${x-5},${y2-7} L${x},${y2} L${x+5},${y2-7}`);
   rows.forEach(row=>{const label=relative(row.querySelector('.weight-route')),tile=relative(row.querySelector('.video-tile'));rightArrow(label.x+label.w+3,label.y+label.h/2,tile.x-5);});
  }else{
   const targets=rows.map(row=>relative(row.querySelector('.weight-route > strong')));
   const sourceX=source.x+source.w+5,sourceY=source.y+source.h/2;
   const targetX=Math.min(...targets.map(r=>r.x))-8,railX=(sourceX+targetX)/2;
   const ys=targets.map(r=>r.y+r.h/2);
   paths.push(`M${sourceX},${sourceY} H${railX}`,`M${railX},${Math.min(sourceY,...ys)} V${Math.max(sourceY,...ys)}`);
   ys.forEach(y=>rightArrow(railX,y,targetX));
  }
  svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
  svg.innerHTML=`<path class="flow-lines" d="${paths.join(' ')}"/><path class="flow-heads" d="${heads.join(' ')}"/>`;
 }
 const schedule=()=>{if(!frame)frame=requestAnimationFrame(draw);};
 const observer=new ResizeObserver(schedule);
 [stage,stage.querySelector('.cfg-native'),stage.querySelector('.cfg-insertion'),...stage.querySelectorAll('.cfg-branch')].forEach(el=>observer.observe(el));
 stage._releaseGuidanceFlow=()=>{observer.disconnect();cancelAnimationFrame(frame);};
 schedule();
}

// Native multi-step CFG and three adapter weights are visible together for both cases.
function renderBaseCases(){
 S.base.forEach(c=>{
  const article=document.createElement('article');article.className='cfg-case';article.dataset.case=c.id;
  article.innerHTML=`<div class="cfg-case-heading"><h3>${esc(c.title)}</h3><span>Same prompt · same seed · 50 sampling steps</span></div><div class="cfg-case-stage"><div class="cfg-native"><span class="mini-label">ORIGINAL BASE MODEL</span>${tile(c.native,'Default multi-step + CFG','Native CFG = 5',false,'50 steps · 2 model forwards per step')}<p class="native-guidance-note">The standard guided result.</p><div class="cfg-insertion"><span class="lora-chip cfg">Insert CFG LoRA</span><span>→</span></div></div><div class="cfg-branches">${c.variants.map((a,i)=>`<div class="cfg-branch"><div class="weight-route"><span class="branch-arrow">→</span><strong>${['Low','Medium','High'][i]} weight</strong><div class="fixed-scale" role="img" aria-label="Fixed CFG LoRA weight ${c.weights[i]} of three illustrated weights"><span style="left:${i*50}%"></span></div><span class="weight-number">λ<sub>cfg</sub> = ${c.weights[i]}</span></div>${tile(a,'+ CFG LoRA',`λ = ${c.weights[i]}`,true,'1 conditional forward per step')}</div>`).join('')}</div></div>`;
  $('base-cases').append(article);attachGroup(article.querySelector('.cfg-case-stage'));attachGuidanceFlow(article.querySelector('.cfg-case-stage'));
 });
}
renderBaseCases();
const methodObserver=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('animate');methodObserver.unobserve(e.target);}}),{threshold:.3});methodObserver.observe($('training-map'));

function makeTransfer(type,title,subtitle){
 const list=S[type],root=document.createElement('article');root.className='cfg-case transfer-demo';root.dataset.transfer=type;
 root.innerHTML=`<div class="cfg-case-heading transfer-heading"><span class="mini-label">${subtitle}</span><h3>${title}</h3></div><div class="transfer-content"></div>`;
 $('transfer-demos').append(root);
 const content=root.querySelector('.transfer-content');
 function render(selected){
  destroyWithin(content);const c=list[selected];content.dataset.case=c.id;
  const isScope=type==='scope';
  content.innerHTML=`<div class="transfer-case-caption"><strong>${esc(c.title)}</strong><span>Shared adapters · 4-step outputs</span></div>
   <div class="cfg-case-stage transfer-guidance-stage">
    <div class="cfg-native transfer-native"><span class="mini-label">${isScope?'ORIGINAL DOWNSTREAM MODEL':'VIDEO CONTINUATION INPUT'}</span>
     ${tile(c.original,isScope?'Native SCOPE':'Original video',isScope?'Original prompt':'Conditioning clip',false,isScope?'Original downstream output':'The supplied continuation input')}
     <div class="cfg-insertion transfer-insertion"><div class="transfer-adapters"><span class="lora-chip step">Few-step <small>λ<sub>step</sub> = 1</small></span><span class="lora-chip cfg">CFG LoRA</span></div><span>→</span></div>
     <p class="native-guidance-note">Adapters reused from the base model.</p>
    </div>
    <div class="cfg-branches transfer-guidance-results" role="group" aria-label="${title}: three recorded CFG LoRA weights">
     ${c.variants.map((a,i)=>`<div class="cfg-branch"><div class="weight-route"><span class="branch-arrow">→</span><strong>${['Low','Medium','High'][i]} weight</strong><div class="fixed-scale" role="img" aria-label="Fixed CFG LoRA weight ${c.weights[i]}"><span style="left:${i*50}%"></span></div><span class="weight-number">λ<sub>cfg</sub> = ${c.weights[i]}</span></div>${tile(a,'+ LongLive-Plug',`λ = ${c.weights[i]}`,true,'4 steps · 1 forward / step')}</div>`).join('')}
    </div>
   </div>
   <p class="transfer-footnote">${esc(c.originalNote)}</p>
   <details><summary>Prompt & sampling settings</summary><p>${esc(c.prompt)}</p><p>Wan2.2-TI2V-5B family · 4 steps · runtime CFG = 1 · λ<sub>step</sub> = 1. The three recorded outputs vary CFG LoRA weight (${c.weights.join(', ')}); the scales show adapter weights, not native CFG settings. Increasing the adapter weight can also change layout and appearance.</p></details>`;
  // Keep the three generated results synchronized. The original/input clip has its own duration.
  attachGroup(content.querySelector('.transfer-native .video-tile'),{sync:false});
  attachGroup(content.querySelector('.transfer-guidance-results'));
  attachGuidanceFlow(content.querySelector('.cfg-case-stage'));
 }
 const initial=type==='scope'?Math.max(0,list.findIndex(c=>c.title==='Autumn Temple')):0;
 render(initial);
}
makeTransfer('scope','SCOPE','INTERACTIVE WORLD MODEL / WAN2.2');
makeTransfer('continuation','Video continuation','EXTEND THE STORY / WAN2.2');

function chart(exp){
 const colors={base:'#aeb8a0',distilled:'#a79a84',long:'#78a72f'},all=exp.curves.flatMap(c=>c.points.map(p=>p[1]));const min=Math.floor(Math.min(...all)-1),max=Math.ceil(Math.max(...all)+.5),x=t=>40+(t-15)/49*355,y=v=>150-(v-min)/(max-min)*118;
 let svg=`<svg class="quality-chart" viewBox="0 0 430 182" role="img" aria-label="Mean of seven VBench dimensions versus rollout duration for ${esc(exp.title)}"><title>${esc(exp.title)}: quality over rollout duration</title>`;
 for(let i=0;i<4;i++){const v=min+(max-min)*i/3;svg+=`<line x1="40" x2="397" y1="${y(v)}" y2="${y(v)}" stroke="#e5eadc"/><text x="30" y="${y(v)+3}" text-anchor="end">${v.toFixed(0)}</text>`;}
 for(const t of [16,32,48,64])svg+=`<text x="${x(t)}" y="173" text-anchor="middle">${t}s</text>`;
 exp.curves.forEach(c=>{const color=colors[c.style]||colors.base;svg+=`<polyline points="${c.points.map(p=>x(p[0])+','+y(p[1])).join(' ')}" fill="none" stroke="${color}" stroke-width="${c.style==='long'?2.6:1.7}" ${c.style==='distilled'?'stroke-dasharray="4 3"':''}/>`;c.points.forEach(p=>svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${color}"><title>${esc(c.label)} · ${p[0].toFixed(1)} seconds · ${p[1].toFixed(2)}</title></circle>`);});
 svg+='</svg><div class="chart-legend">'+exp.curves.map(c=>`<span><i style="--legend:${colors[c.style]||colors.base}"></i>${esc(c.label)}</span>`).join('')+'</div>';return svg;
}
function renderLong(i){const e=S.long[i],root=$('long-demo');destroyWithin(root);const before=e.curves[0].points.at(-1)[1],after=e.curves.at(-1).points.at(-1)[1];root.innerHTML=`<div class="long-meta"><h3>${esc(e.case)}</h3><span>Late rollout · ${e.start.toFixed(1)}–${e.end.toFixed(1)} s</span></div><div class="long-videos">${e.variants.map((a,j)=>tile(a,a.label,'',j===2)).join('')}</div><div class="timeline-context"><span>0 s</span><div class="line"><span></span></div><span>Only the final ~8 seconds shown</span><span>${e.end.toFixed(1)} s</span></div><div class="long-chart-row"><div class="score-uplift"><strong>+${e.gain.toFixed(2)}<small> pts</small></strong><h4>Quality at the longest rollout</h4><p>${before.toFixed(2)} → ${after.toFixed(2)} vs. native<br>Mean of seven VBench dimensions</p></div><div>${chart(e)}</div></div>`;attachGroup(root.querySelector('.long-videos'));}
tabbar($('long-tabs'),S.long,i=>renderLong(i));renderLong(0);

function restartCardInsertion(g){
 g.cycle++;g.root.dataset.insertionCycle=String(g.cycle);g.revealRemaining=reduced.matches?0:1200;g.root.classList.toggle('revealed',!g.revealRemaining);
}
function revealCard(card){const g=card._group;if(!g)return;if(reduced.matches){g.revealRemaining=0;card.classList.add('revealed');}else if(!g.cycle)restartCardInsertion(g);}

function makeCard(m){const c=m.cases[m.selectedCase],card=document.createElement('article');card.className='coverage-card';card.dataset.model=m.id;card.innerHTML=`<button class="pair-open" aria-label="Compare ${esc(m.name)}"><span class="card-pair"><span>${video(media(c,false))}<span class="card-video-label">Multi-step inference,<br>2-forward-passes CFG</span></span><span class="card-right">${video(media(c))}<span class="card-video-label ours">Four-step inference,<br>single-pass CFG</span></span><span class="insert-plug">↳ Insert LongLive-Plug</span></span></button><div class="card-body"><div><h4>${esc(m.name)}</h4><button class="card-open" aria-label="Open ${esc(m.name)} comparison">↗</button></div><p>${esc(m.task)}</p><div class="card-count">${m.cases.length} paired example${m.cases.length>1?'s':''} · ${esc(m.categories[0])}</div></div>`;card.querySelectorAll('button').forEach(b=>b.onclick=()=>openModel(m.id));attachGroup(card);return card;}
function renderCoverage(){destroyWithin($('coverage-families'));$('coverage-families').innerHTML='';C.families.forEach(f=>{const models=C.models.filter(m=>m.family===f.id&&(!state.query||[m.name,m.task,...m.categories,...m.cases.map(c=>c.prompt)].join(' ').toLowerCase().includes(state.query)));const show=state.query?models:models.slice(0,state.limits[f.id]);const block=document.createElement('div');block.className='family-section';block.id='family-'+f.id;block.innerHTML=`<div class="family-heading"><div><h3>${f.id==='h3'?'MiniMax-H3':esc(f.name)} ${f.id==='h3'?'':`<small>${esc(f.detail)}</small>`}</h3><p>${models.length} downstream models · ${models.reduce((n,m)=>n+m.cases.length,0)} paired examples</p></div><span>Train here. Reuse here.</span></div><div class="coverage-grid"></div>`;const grid=block.querySelector('.coverage-grid');show.forEach(m=>grid.append(makeCard(m)));if(!models.length)grid.innerHTML='<p class="empty-family">No matching models in this family.</p>';if(show.length<models.length){const b=document.createElement('button');b.className='more-button';b.textContent=`Show all ${models.length} ${f.id==='h3'?'MiniMax-H3':f.name} models ↓`;b.onclick=()=>{const oldTop=block.getBoundingClientRect().top;state.limits[f.id]=models.length;renderCoverage();const newBlock=$('family-'+f.id);window.scrollBy(0,newBlock.getBoundingClientRect().top-oldTop);};block.append(b);}$('coverage-families').append(block);});}
let searchTimeout;$('search').oninput=()=>{clearTimeout(searchTimeout);searchTimeout=setTimeout(()=>{state.query=$('search').value.trim().toLowerCase();renderCoverage();},130);};renderCoverage();

const comparisonMetrics={scope:[{name:'FVD',direction:'↓ lower is better',values:[805.5,502.1,478.7],decimals:1},{name:'JEPA',direction:'↑ higher is better',values:[.732,.782,.792],decimals:3},{name:'Photometric error',direction:'↓ lower is better',values:[8.468,5.695,4.246],decimals:3}],control:[{name:'DOVER',direction:'↑ higher is better',values:[8.90,10.25,10.11],decimals:2},{name:'SSIM',direction:'↑ higher is better',values:[.560,.544,.566],decimals:3},{name:'mIoU',direction:'↑ higher is better',values:[.582,.595,.612],decimals:3}]};
function metric(m){const max=Math.max(...m.values)*1.1;return `<div class="metric-card"><h4>${m.name}<span>${m.direction}</span></h4>${m.values.map((v,i)=>`<div class="metric-line ${i===2?'ours':''}"><span>${['Naive 4-step','Task-specific','LongLive-Plug'][i]}</span><span class="metric-bar"><i style="width:${v/max*100}%"></i></span><strong>${v.toFixed(m.decimals)}</strong></div>`).join('')}</div>`;}
S.comparison.forEach(c=>{const root=document.createElement('article');root.className='comparison-block';root.dataset.comparison=c.id;const scope=c.id==='scope';root.innerHTML=`<div class="comparison-heading"><div><h3>${c.title}</h3><p>${scope?'WorldCam example · 1,378 CrossFPS clips evaluated':'Task 0309 · 600 PAI-Bench-C cases evaluated'}</p></div><span class="zero-badge">Our downstream training cost: 0</span></div><div class="comparison-videos">${c.variants.map((v,i)=>tile(v,[`Native · ${scope?'30':'40'} steps`,'Naive · 4 steps','Task-specific distillation','+ LongLive-Plug'][i],i>=2?'4 steps':'',i===3)).join('')}</div><div class="metrics-row">${comparisonMetrics[c.id].map(metric).join('')}</div><p class="caption">${scope?'All accelerated methods use four steps. LongLive-Plug: λstep = 1, λcfg = 3. Lower FVD and photometric error than task-specific distillation in this evaluation.':'All quantitative bars use the same 600 cases at 720P and four steps with runtime CFG disabled. LongLive-Plug: λstep = 1, λcfg = 1. DOVER is close to task-specific distillation; control fidelity varies by metric.'}</p>`;$('comparison-demos').append(root);attachGroup(root.querySelector('.comparison-videos'));});

function openModel(id,caseId,updateHash=true){const m=C.models.find(m=>m.id===id);if(!m)return;if(!modal.open)opener=document.activeElement;currentDialog=m;$('dialog-family').textContent=familyName(m.family);$('dialog-title').textContent=m.name;$('dialog-task').textContent=m.task;if(!modal.open)modal.showModal();document.body.classList.add('modal-open');refreshAll();loadCase(m.cases.find(c=>c.id===caseId)||m.cases[m.selectedCase],updateHash);}
function loadCase(c,updateHash=true){destroyGroup($('dialog-pair'));const m=currentDialog;const steps=c.nativeSteps?`${c.nativeSteps} ${c.unit}`:'Default schedule';$('dialog-pair').innerHTML=tile(media(c,false),'Native inference',steps)+tile(media(c),'+ LongLive-Plug',`${c.steps} ${c.unit}`,true);$('case-prompt').textContent=c.prompt||'The source task’s driving input and conditioning.';$('case-protocol').textContent=[c.fps?`${c.fps} fps`:null,c.duration?`${c.duration.toFixed(2)} s`:null,c.cfg!=null?`Native CFG ${c.cfg}`:null,c.oursCfg!=null?`Plug runtime CFG ${c.oursCfg}`:null].filter(Boolean).join(' · ');$('case-note').textContent=c.note||'';$('pair-rate').value='1';$('pair-seek').value='0';$('pair-status').textContent='Synchronized playback · paired recordings';dialogGroup=attachGroup($('dialog-pair'),{dialog:true});dialogGroup.paused=state.paused;refreshGroup(dialogGroup);$('pair-play').textContent=dialogGroup.paused?'▷ Play':'Ⅱ Pause';if(updateHash)history.replaceState(null,'','#model='+encodeURIComponent(m.id)+'&case='+encodeURIComponent(c.id));}
function closeModel(){destroyGroup($('dialog-pair'));dialogGroup=null;currentDialog=null;modal.close();document.body.classList.remove('modal-open');if(location.hash.startsWith('#model='))history.replaceState(null,'','#coverage');refreshAll();opener?.focus({preventScroll:true});}
$('close-case').onclick=closeModel;modal.addEventListener('cancel',e=>{e.preventDefault();closeModel();});$('pair-play').onclick=()=>{if(!dialogGroup)return;dialogGroup.paused=!dialogGroup.paused;$('pair-play').textContent=dialogGroup.paused?'▷ Play':'Ⅱ Pause';refreshGroup(dialogGroup);};$('pair-seek').oninput=()=>{if(!dialogGroup)return;const d=Math.min(...dialogGroup.videos.map(v=>v.duration));if(Number.isFinite(d))dialogGroup.videos.forEach(v=>v.currentTime=Number($('pair-seek').value)/1000*d);};$('pair-rate').onchange=()=>dialogGroup?.videos.forEach(v=>v.playbackRate=Number($('pair-rate').value));$('pair-fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('dialog-pair').requestFullscreen();}catch{$('pair-status').textContent='Fullscreen is unavailable here; the comparison remains playable.';}};
function fromHash(){if(location.hash.startsWith('#model=')){const p=new URLSearchParams(location.hash.slice(1));openModel(p.get('model'),p.get('case'),false);}}
window.addEventListener('hashchange',fromHash);window.addEventListener('popstate',fromHash);
modal.addEventListener('click',e=>{if(e.target!==modal)return;const r=modal.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeModel();});
document.addEventListener('visibilitychange',()=>{refreshAll();});reduced.addEventListener('change',e=>{state.paused=e.matches;refreshAll();});
const sections=[...document.querySelectorAll('main>.chapter')];let navTick=false;
window.addEventListener('scroll',()=>{if(navTick)return;navTick=true;requestAnimationFrame(()=>{navTick=false;let active=sections[0];for(const s of sections)if(s.getBoundingClientRect().top<innerHeight*.35)active=s;document.querySelectorAll('.sidebar nav a').forEach(a=>{const on=a.hash==='#'+active.id;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});});},{passive:true});
initOverview();refreshAll();fromHash();requestAnimationFrame(syncGroups);
