/* One camera, one workcell. Scroll position is the animation playhead. */
(()=>{
 const flight=document.querySelector('.flight');if(!flight)return;
 const nav=document.querySelector('.nav'),plane=flight.querySelector('[data-plane="0"]'),copies=[...flight.querySelectorAll('.flight-copy')],intro=flight.querySelector('.flight-intro'),outro=flight.querySelector('.flight-outro'),opening=flight.querySelector('.flight-opening-image'),track=flight.querySelector('.flight-track>span'),counter=document.querySelector('#flight-counter'),mode=document.querySelector('#flight-mode');
 const scene=plane.querySelector('.motion-scene'),left=plane.querySelector('.left-arm'),right=plane.querySelector('.right-arm'),rear=plane.querySelector('.rear-arm'),part=plane.querySelector('.housing-part'),message=plane.querySelector('#motion-message'),route=plane.querySelector('#motion-route'),status=plane.querySelector('#motion-state'),playhead=plane.querySelector('.scene-playhead>span'),detection=plane.querySelector('.detect.housing');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
 const ease=(a,b,v)=>{const t=clamp((v-a)/(b-a));return t*t*(3-2*t)};
 const mix=(a,b,t)=>a+(b-a)*t;
 let current=0,target=0,raf=0,lastTime=0,lockedZ=0;
 const poses=t=>{
  const reach=ease(.01,.18,t),release=ease(.32,.47,t),lift=ease(.18,.26,t)*(1-ease(.27,.33,t)),contact=ease(.56,.76,t),collect=ease(.79,.98,t);
  const angle=mix(-25,0,collect)*contact;
  let dx=4.7*ease(.18,.3,t),dy=-6*lift;
  if(t>=.79){const pivot=[1178,365],original=[850,573],goal=[850+1672*.047,573],a=25*Math.PI/180,vec=[goal[0]-pivot[0],goal[1]-pivot[1]],local=[vec[0]*Math.cos(a)-vec[1]*Math.sin(a),vec[0]*Math.sin(a)+vec[1]*Math.cos(a)],r=angle*Math.PI/180;dx=(pivot[0]+local[0]*Math.cos(r)-local[1]*Math.sin(r)-original[0])/1672*100;dy=(pivot[1]+local[0]*Math.sin(r)+local[1]*Math.cos(r)-original[1])/941*100;}
  return {leftX:0,leftY:-10*reach*(1-release)+4*release,leftA:0,leftSX:1+.55*reach*(1-release)-.06*release,leftSY:1-.35*reach*(1-release),rightX:0,rightY:0,rightS:1-.035*ease(.12,.4,t),rearA:angle,partX:dx,partY:dy};
 };
 function animateWorkcell(t){
  const pose=poses(t);left.style.transform=`translate(${pose.leftX}%,${pose.leftY}%) scale(${pose.leftSX},${pose.leftSY})`;right.style.transform=`scale(${pose.rightS})`;rear.style.transform=`rotate(${pose.rearA}deg)`;part.style.transform=`translate(${pose.partX}%,${pose.partY}%)`;
  detection.style.transform=`translate(${pose.partX/.14}%,${pose.partY/.2}%)`;
  playhead.style.width=`${t*100}%`;
  const phases=t<.18?['ATLAS / POSITIONING','Atlas moves into the transfer tray.','ATLAS HAS THE ZONE']:t<.33?['ATLAS / PLACEMENT','Housing positioned for pickup.','OBJECT: HOUSING_01']:t<.48?['ATLAS → RELAY','Atlas withdraws from the shared zone.','CLEARANCE PENDING']:t<.56?['ATLAS → NOVA','Handoff ready. The transfer zone is clear.','HANDOFF_READY / ILLUSTRATION']:t<.76?['NOVA → ATLAS','Handoff accepted. Nova reaches for the housing.','NOVA HAS THE ZONE']:t<.8?['NOVA / GRASP','Gripper closes around the housing.','PICKUP IN PROGRESS']:['NOVA → RELAY','Housing collected. Nova lifts clear.','HANDOFF COMPLETE / ILLUSTRATION'];
  route.textContent=phases[0];message.textContent=phases[1];status.textContent=phases[2];
  scene.dataset.phase=t<.33?'positioning':t<.56?'clearance':'pickup';
 }
 function measure(){const rect=flight.getBoundingClientRect();target=clamp(-rect.top/Math.max(1,rect.height-innerHeight));const perspective=innerWidth<600?650:1000;const scale=Math.min(innerWidth*.9/plane.offsetWidth,innerHeight*.61/plane.offsetHeight);lockedZ=perspective*(1-1/Math.max(.4,scale));if(!raf)raf=requestAnimationFrame(update);}
 function paint(p){
  const perspective=innerWidth<600?650:1000,entry=ease(.035,.2,p),motion=ease(.22,.91,p);
  const introOpacity=1-ease(.015,.085,p);intro.style.opacity=introOpacity;intro.style.transform=`translate(-50%,-50%) scale(${1+Math.min(p,.15)*1.8})`;intro.style.pointerEvents=introOpacity>.45?'auto':'none';intro.inert=introOpacity<.1;
  opening.style.opacity=(1-ease(.01,.09,p))*.15;
  const pastFlight=flight.getBoundingClientRect().bottom<innerHeight*.25;nav.style.transform=`translateY(${pastFlight?0:-ease(0,.018,p)*100}%)`;nav.inert=p>.025&&!pastFlight;
  plane.style.opacity=ease(.035,.09,p);plane.style.visibility=p>.035?'visible':'hidden';plane.style.transform=`translate3d(${mix(45,0,entry)}px,0,${mix(-perspective*1.95,lockedZ,entry)}px) rotateY(${mix(-18,0,entry)}deg) rotateX(${mix(5,0,entry)}deg)`;
  plane.querySelectorAll('[data-depth]').forEach(layer=>{const d=Number(layer.dataset.depth),spread=mix(85*d,8*d,ease(.11,.22,p));layer.style.transform=`translate3d(0,0,${spread}px)`;layer.style.opacity=layer.classList.contains('contour-layer')?1-ease(.17,.23,p):layer.classList.contains('detection-layer')?mix(.65,.22,ease(.18,.23,p)):.4;});
  animateWorkcell(motion);
  const chapter=p<.46?0:p<.7?1:2;
  copies.forEach((copy,i)=>{copy.style.opacity=i===chapter?ease(.105,.16,p)*(1-ease(.92,.965,p)):0;copy.style.visibility=i===chapter?'visible':'hidden';copy.style.transform='translate(-50%,0)';});
  const finish=ease(.94,.99,p);outro.style.opacity=finish;outro.style.transform='translate(-50%,-50%)';outro.style.pointerEvents=finish>.7?'auto':'none';outro.inert=finish<.1;plane.querySelector('.plane-image').style.filter=`brightness(${1-finish*.75})`;plane.querySelector('.in-frame-message').style.opacity=ease(.18,.24,p)*(1-finish);
  track.style.width=`${p*100}%`;const active=p<.105?0:p<.46?1:p<.7?2:3;counter.textContent=`0${active} / 03`;mode.textContent=['OPERATOR VIEW','ATLAS / POSITIONING','SHARED ZONE / CLEARANCE','NOVA / PICKUP'][active];flight.querySelector('.flight-scroll').style.opacity=1-ease(.92,.98,p);
 }
 function update(time){raf=0;if(reduced.matches){nav.inert=false;intro.inert=false;outro.inert=false;animateWorkcell(0);return;}const dt=Math.min(time-(lastTime||time-16),64);lastTime=time;current+=(target-current)*(1-Math.exp(-dt/55));if(Math.abs(target-current)<.00003)current=target;paint(current);if(current!==target)raf=requestAnimationFrame(update);}
 addEventListener('scroll',measure,{passive:true});addEventListener('resize',measure,{passive:true});reduced.addEventListener('change',measure);measure();
})();
