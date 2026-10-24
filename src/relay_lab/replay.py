"""Write a self-contained viewer with one video/decision playback timeline."""

import html
import json
from pathlib import Path


def write_replay(
    path: Path, summary: dict, records: list[dict], video_name: str | None = None
) -> None:
    """Save a portable spectator view; recorded strings never become HTML."""
    data = json.dumps({"summary": summary, "records": records}, ensure_ascii=True)
    # Neither recorded messages nor model names can close the embedded script tag.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    video = ""
    if video_name:
        source = html.escape(video_name, quote=True)
        video = (
            f'<video id="video" preload="auto" playsinline src="{source}" '
            'aria-label="Recorded MuJoCo simulation, controlled by the timeline below"></video>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML.replace("__VIDEO__", video).replace("__DATA__", data), encoding="utf-8")


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relay Lab · Workcell replay</title>
<style>
:root{color-scheme:light;--bg:#eff1f2;--paper:#fff;--ink:#1c2a32;--muted:#68767e;
--line:#dce2e5;--teal:#087b85;--amber:#b86a13;--green:#22744c;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit}button,select{color:var(--ink)}button{cursor:pointer}button:disabled{cursor:default;opacity:.4}
button:focus-visible,summary:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #4ba6b5;outline-offset:3px}
main{max-width:1512px;margin:auto;padding:26px 32px 28px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:12px}.brand-icon{width:38px;height:38px;border:1px solid #b9c7cd;border-radius:9px;background:white;display:grid;place-items:center}
h1{font-size:23px;letter-spacing:-.7px;margin:0;line-height:1.15;font-weight:700}.subheading{font-size:11px;color:var(--muted);margin-top:4px;letter-spacing:1px;text-transform:uppercase}
.header-right{display:flex;align-items:center;gap:10px}.badge{font-size:11px;padding:5px 9px;border:1px solid var(--line);border-radius:5px;white-space:nowrap;background:white;line-height:1.4}
.badge.complete{color:var(--green);background:#edf8f0;border-color:#cce4d3}.badge.incomplete{color:#8b5e1d;background:#fcf5e8;border-color:#ebdbc0}
.badge.source{font-family:var(--mono);background:#f7f9f9}.runline{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:17px;font-size:12px;color:var(--muted)}
.runline strong{color:var(--ink);font-weight:550}.run-meta{display:flex;gap:21px;flex-wrap:wrap}.kicker{font-size:10px;letter-spacing:1.3px;text-transform:uppercase;color:var(--muted);font-weight:650}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px;align-items:start}.panel{border:1px solid var(--line);background:var(--paper);border-radius:11px;overflow:hidden;box-shadow:0 2px 4px #1d2b3410}
.panel-head{padding:14px 18px;display:flex;align-items:center;justify-content:space-between;gap:15px;border-bottom:1px solid var(--line)}h2{font-size:13px;font-weight:650;margin:0}
.panel-head .minor{font-size:11px;color:var(--muted)}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--teal);margin-right:6px}
.viewport{position:relative;background:#c5cdd2;aspect-ratio:3/2;min-height:0;max-height:650px;overflow:hidden}video{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:contain}.viewport canvas{display:block;width:100%;height:100%;object-fit:contain}
.viewport.has-video #scene{display:none}.view-label{position:absolute;top:14px;left:16px;padding:5px 9px;border:1px solid #ffffff8a;border-radius:5px;background:#ffffffd9;color:#34454d;font:10px var(--mono);letter-spacing:.6px;pointer-events:none}
.frame-label{position:absolute;right:16px;bottom:13px;padding:4px 7px;background:#ffffffd9;border-radius:4px;font:10px var(--mono);color:#34454d;pointer-events:none}
.transport{padding:15px 18px 14px}.transport-row{display:flex;align-items:center;gap:9px}.icon-button{display:grid;place-items:center;border:1px solid var(--line);border-radius:6px;background:white;min-width:30px;height:30px;font-size:15px}
.play-button{background:#243b45;color:white;border:0;border-radius:6px;height:32px;padding:0 14px;font-size:12px;font-weight:600;min-width:76px}
.transport select{margin-left:2px;background:#f5f7f8;border:1px solid var(--line);height:28px;border-radius:5px;font-size:11px;padding:0 6px}.timecode{margin-left:auto;font:11px var(--mono);color:var(--muted)}
.timeline-wrap{position:relative;margin:12px 0 0}.timeline-wrap input{display:block;width:100%;height:16px;accent-color:var(--teal);margin:0;cursor:pointer}.timeline-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-top:4px;gap:16px}
.side{display:grid;gap:15px}.side .body{padding:16px 18px}.task-title{font-size:18px;line-height:1.35;letter-spacing:-.35px;font-weight:650;margin:8px 0}
p{margin:6px 0}.muted{color:var(--muted);font-size:12px}.task-description{font-size:12px;color:var(--muted);margin-bottom:17px}
.order-row{display:grid;grid-template-columns:28px 1fr auto;gap:9px;align-items:center;padding:10px 0;border-top:1px solid #edf0f2}.number{font:10px var(--mono);color:#809098}.object{display:flex;gap:9px;align-items:center;font-size:12px;text-transform:capitalize}.swatch{width:13px;height:13px;border-radius:3px;box-shadow:inset 0 0 0 1px #00000012}.slot-state{font-size:10px;color:#859198}.slot-state.done{color:var(--green)}
.progress-track{height:4px;border-radius:5px;background:#e6ecee;overflow:hidden;margin:15px 0 8px}.progress-track>div{height:100%;background:var(--teal);width:0;transition:width .15s}.progress-copy{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.event-list{max-height:182px;overflow:auto}.event{font-size:12px;display:flex;gap:10px;line-height:1.45;padding:8px 0;border-top:1px solid #edf0f2}.event:first-child{border:0;padding-top:0}.event-time{font:10px var(--mono);color:#839099;padding-top:3px;min-width:31px}.event-text{overflow-wrap:anywhere}.empty{color:#839099;font-size:12px;padding:2px 0}
.agents{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.agent-card{border-top:3px solid var(--teal)}.agent-card.packer{border-top-color:#d58b3a}.agent-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:8px}.agent-head strong{font-size:13px}.agent-id{font:10px var(--mono);color:var(--teal);margin-right:7px}.packer .agent-id{color:var(--amber)}
.agent-body{padding:15px 18px}.context-tag{font-size:10px;color:#7c8990}.instruction{font-size:14px;font-weight:550;min-height:42px;line-height:1.45;margin:6px 0 14px;overflow-wrap:anywhere}.agent-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.message{font-size:11px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;min-height:36px;margin-top:6px;color:#50616a}.message.empty{color:#8b969c}
.feedback-loop{margin-top:16px;padding:14px;border:1px solid #d6e7e8;border-radius:8px;background:#f4f9f9}.feedback-loop .message{max-height:150px;overflow:auto;margin-bottom:12px}.feedback-loop .message:last-child{margin-bottom:0}.feedback-loop .kicker{color:#397681}.telemetry{display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:12px;margin-top:12px;font:10px var(--mono);color:#74838b}.telemetry strong{font-weight:500;color:#3b515c}
.model-details{display:grid;grid-template-columns:auto 1fr;gap:8px 13px;font-size:11px}.model-details dt{color:#7b8991}.model-details dd{margin:0;overflow-wrap:anywhere;color:#344b56}.model-details .wide{grid-column:1/-1}
details{margin-top:12px}summary{cursor:pointer;font-size:11px;color:var(--muted)}pre{font:10px/1.5 var(--mono);background:#f3f6f7;padding:11px;border-radius:5px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto;margin-bottom:0}
footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;color:#859198;font-size:10px;margin-top:17px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(min-width:1380px){.workspace{grid-template-columns:minmax(0,1fr) 312px}}
@media(max-width:1000px){main{padding:20px}.workspace{grid-template-columns:minmax(0,1fr) 260px}.agent-grid{grid-template-columns:1fr;gap:10px}.runline{align-items:start}.run-meta{gap:8px 18px}}
@media(max-width:760px){main{padding:16px 12px}.workspace,.agents{grid-template-columns:1fr}.side{grid-template-columns:1fr 1fr;align-items:start}.side .panel:last-child{grid-column:1/-1}.header-right{gap:5px;flex-direction:column;align-items:end}.subheading{font-size:9px}.runline{display:block}.runline>.muted{display:none}.viewport{height:auto;aspect-ratio:3/2}.agents{margin-top:15px;gap:15px}.agent-grid{grid-template-columns:1fr 1fr}.task-title{font-size:16px}}
@media(max-width:460px){.side{grid-template-columns:1fr}.agent-grid{grid-template-columns:1fr}.timecode{font-size:10px}.transport-row{gap:6px}.transport{padding:12px}.brand-icon{display:none}.badge{font-size:9px}.panel-head{padding:12px 14px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><main>
<header><div class="brand"><div class="brand-icon" aria-hidden="true"><svg width="23" height="23" viewBox="0 0 24 24" fill="none"><path d="M4 20V4h16v16M4 8h16M8 8v6m8-6v6M6 14h4m4 0h4M6 14v3m4-3v3m4-3v3m4-3v3" stroke="#365b68" stroke-width="1.6" stroke-linecap="round"/></svg></div><div><h1>Relay Lab</h1><div class="subheading">Multi-agent robotics workcell</div></div></div>
<div class="header-right"><span id="source-badge" class="badge source"></span><span id="result" class="badge"></span></div></header>
<div class="runline"><div id="stats" class="run-meta"></div><span class="muted">Recorded experiment / local replay</span></div>
<div class="workspace"><div class="primary"><section class="panel" aria-labelledby="camera-title"><div class="panel-head"><h2 id="camera-title"><span class="dot"></span>Workcell overview</h2><span id="view-type" class="minor">MuJoCo simulation</span></div>
<div class="viewport" id="viewport">__VIDEO__<canvas id="scene" width="1000" height="660" role="img" aria-label="Recorded top-down state: source blocks, shared relay, packing slots, and gripper positions"></canvas><span class="view-label" id="view-label">CAM 01 / SPECTATOR</span><span class="frame-label" id="frame-label"></span></div>
<div class="transport"><div class="transport-row"><button id="play" class="play-button" type="button" aria-label="Play replay">▶ Play</button><button id="back" class="icon-button" type="button" aria-label="Previous decision">‹</button><button id="next" class="icon-button" type="button" aria-label="Next decision">›</button><label class="sr-only" for="speed">Playback speed</label><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><span id="time" class="timecode"></span></div>
<div class="timeline-wrap"><label class="sr-only" for="timeline">Synchronized simulation and decision timeline</label><input id="timeline" type="range" min="0" value="0" step="1"></div><div class="timeline-labels"><span id="phase"></span><span id="step-count"></span></div></div></section>
<div class="agents" id="agents"></div></div>
<aside class="side"><section class="panel"><div class="panel-head"><h2>Task</h2><span class="minor">01 / Parcel relay</span></div><div class="body"><div class="kicker">Coordinate · Transfer · Pack</div><p class="task-title">One relay.<br>Three deliveries.</p><p class="task-description" id="task-description">Picker supplies the blocks. Packer knows the required order.</p><div id="order"></div><div class="progress-track"><div id="progress"></div></div><div class="progress-copy"><span id="packed-count"></span><span id="task-state"></span></div><p id="visibility-note" class="muted" style="font-size:10px;margin-top:12px">Spectator view · Packing order is private to the packer.</p></div></section>
<section id="mediator-panel" class="panel" hidden><div class="panel-head"><h2>Team mediator</h2><span id="mediator-call" class="minor"></span></div><div class="body"><div class="kicker">Qwen · Verified feedback → next subgoal</div><p id="mediator-plan" class="task-description"></p><details><summary>Model rationale · slot IDs 0–2</summary><p id="mediator-reason" class="task-description"></p></details><p id="mediator-trigger" class="muted"></p></div></section>
<section class="panel"><div class="panel-head"><h2>Activity</h2><span id="event-count" class="minor"></span></div><div class="body"><div id="events" class="event-list" aria-live="off"></div><details><summary>Current decision details</summary><pre id="event-raw"></pre></details></div></section>
<section class="panel"><div class="panel-head"><h2>Run configuration</h2><span class="minor">Recorded provenance</span></div><div class="body"><dl id="model-details" class="model-details"></dl><details id="transport-details" hidden><summary>Agent identities & transport record</summary><pre id="transport-raw"></pre></details><details><summary>Simulation assumptions</summary><div id="limitations" class="muted"></div></details></div></section></aside></div>
<footer><span>Portable replay · No external page assets · Separate robot policy contexts</span><span>Cartesian apparatus · Idealized proximity grasps</span></footer>
</main><script type="application/json" id="rollout">__DATA__</script><script>
'use strict';
const {summary:run,records}=JSON.parse(document.getElementById('rollout').textContent);
const $=id=>document.getElementById(id), cfg=run.config||run, setting=key=>cfg[key]??run[key];
const text=(id,value)=>{$(id).textContent=String(value??'—')};
const names=['picker','packer'], colors={coral:'#e87868',mint:'#49b993',gold:'#d9ad33'};
const policy=setting('policy')||'unknown';
const policyNames={scripted:'Scripted controller','relay-vla':'Relay VLA · Task-adapted policy',smolvla:'SmolVLA · Unadapted checkpoint',http:'HTTP policy endpoint'};
const policyName=policyNames[policy]||policy;
const transportName=setting('planner')==='llm'?'Local LLM mediator':'Local message channel';
const coordination=run.coordination_info||{},isSpacetime=coordination.backend==='spacetimedb';
text('task-description',setting('planner')==='llm'?'Picker supplies the blocks. Packer fills the tray. A local Qwen mediator reads verified results and assigns their next subgoals.':'Picker supplies the blocks. Packer knows the required order. They coordinate through a local message channel.');
text('visibility-note',setting('planner')==='llm'?'Local camera inputs · Central mediator sees team state and packing order.':'Spectator view · Packing order is private to the packer.');
const initial=run.initial_state||records[0]?.state||{},order=run.order||initial.order||[];
const fps=Number(run.video_fps||run.fps||(run.control_dt?1/run.control_dt:10)), total=records.length/fps;
const canvas=$('scene'),ctx=canvas.getContext('2d'),video=$('video');
let index=0,playing=false,lastClock=0,accumulator=0,videoReady=false;
const clock=seconds=>`${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(1).padStart(4,'0')}`;
text('source-badge',policyName);text('result',run.success?'✓ Task complete':'◌ Task incomplete');
$('result').classList.add(run.success?'complete':'incomplete');
for(const [label,value]of [['Seed',setting('seed')],['Communication',setting('communication')?'On':'Off'],['Transport',transportName],['Decisions',run.steps??records.length],['Simulated',`${total.toFixed(1)} s`]]){
 const el=document.createElement('span'),strong=document.createElement('strong');el.append(`${label} `);strong.textContent=String(value??'—');el.append(strong);$('stats').append(el);
}
function detail(label,value){if(value==null||value==='')return;const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=typeof value==='object'?JSON.stringify(value):String(value);$('model-details').append(dt,dd)}
const model=run.model||run.policy_metadata||run.model_info||{};
detail('Controller',policyName);detail('Model',setting('model_id')||setting('checkpoint')||model.model_id||model.checkpoint||(typeof model==='string'?model:null));
if(policy!=='scripted')detail('Device',model.device||run.device||setting('device'));
if(policy==='smolvla'){detail('Adaptation','Pretrained LIBERO checkpoint · no Relay Lab fine-tuning');detail('Parameters',model.parameters?`${(model.parameters/1e6).toFixed(0)}M`:null);detail('Model calls',model.inference_calls);detail('Inference time',model.total_inference_seconds!=null?`${Number(model.total_inference_seconds).toFixed(2)} s`:null)}
if(setting('planner')!=='llm'){detail('Message delay',`${setting('message_latency')??1} decision(s)`);detail('Messages',`${run.messages?.delivered??0} delivered`);}detail('Step budget',setting('max_steps'));
detail('Transport',transportName);
detail('Coordination',isSpacetime?'SpacetimeDB · durable tick barriers':'In-memory reference');
if(isSpacetime){detail('Database',coordination.database);detail('Task state',coordination.task?.status);detail('Committed ticks',coordination.task?.next_tick);detail('Synchronization','Instructions → actions → both observed reports');}
if(policy==='relay-vla'){detail('Adaptation','Learned on original Relay Lab simulator examples');detail('Parameters',model.parameters);detail('Model calls',model.inference_calls);detail('Inference time',model.total_inference_seconds!=null?`${Number(model.total_inference_seconds).toFixed(2)} s`:null)}
if(run.mediator){detail('LLM generations',run.mediator.generation_calls);detail('Mean LLM latency',`${Number(run.mediator.mean_generation_seconds).toFixed(2)} s`);detail('Judge','Observed simulator state · no model call');detail('LLM fallback',run.mediator.fallback?'Enabled':'None');}
const inference=run.inference_stats||run.inference||model.inference;
if(inference)for(const [key,value]of Object.entries(inference))detail(key.replaceAll('_',' '),typeof value==='number'?Number(value.toFixed(3)):value);
detail('Orchestration',run.orchestrator||'Rule-based');detail('Grasp model','Proximity attachment');
const assumptions=run.limitations||['Cartesian end effectors with idealized proximity attachments.','Original parcel-relay task, not a benchmark evaluation.'];
for(const item of assumptions){const p=document.createElement('p');p.textContent=item;$('limitations').append(p)}
for(const [i,color]of order.entries()){
 const row=document.createElement('div');row.className='order-row';
 const number=document.createElement('span');number.className='number';number.textContent=String(i+1).padStart(2,'0');
 const label=document.createElement('span');label.className='object';const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=colors[color]||'#a0aeb5';label.append(swatch,String(color));
 const status=document.createElement('span');status.className='slot-state';status.id=`slot-${i}`;row.append(number,label,status);$('order').append(row);
}
for(const [i,name]of names.entries()){
 const card=document.createElement('section');card.className=`panel agent-card ${name}`;
 // All interpolated identifiers here are fixed internal strings, never log content.
 card.innerHTML=`<div class="agent-body"><div class="agent-head"><strong><span class="agent-id">0${i+1}</span>${name==='picker'?'Picker':'Packer'}</strong><span class="context-tag">Independent context</span></div><div class="kicker">Local instruction</div><div id="${name}-instruction" class="instruction"></div><div class="agent-grid"><div><div class="kicker">Received request</div><div id="${name}-inbox" class="message"></div></div><div><div class="kicker">Sent to peer</div><div id="${name}-outgoing" class="message"></div></div><div><div class="kicker">Observed execution</div><div id="${name}-execution" class="message"></div></div><div><div class="kicker">Received peer status</div><div id="${name}-peer-execution" class="message"></div></div></div><div id="${name}-feedback-loop" class="feedback-loop"><div class="kicker">01 / Context used for this action</div><div id="${name}-model-context" class="message"></div><div class="kicker">02 / Observed result → next decision</div><div id="${name}-action-feedback" class="message"></div></div><div id="${name}-telemetry" class="telemetry"></div><details><summary>Observation, action & coordination payload</summary><pre id="${name}-raw"></pre></details></div>`;$('agents').append(card);
}
$('timeline').max=records.length;
if(setting('planner')==='llm'){for(const name of names){for(const key of ['inbox','outgoing','peer-execution'])$(`${name}-${key}`).parentElement.hidden=true;} $('mediator-panel').hidden=false;}
function message(mail){if(!mail)return 'No message this decision';return `${mail.text}\n${mail.sender} → ${mail.recipient} · delivery ${mail.deliver_tick}`}
function humanEvent(event){const who=event.agent==='picker'?'Picker':event.agent==='packer'?'Packer':event.agent||'Robot';const object=event.object||'block';switch(event.type){case'grasp':return `${who} picked up ${object}`;case'release':return `${who} released ${object}`;case'grasp_missed':return `${who} closed its gripper; no block attached`;case'packed':return `${object[0].toUpperCase()+object.slice(1)} placed in slot ${event.slot+1}`;case'unpacked':return `${object} removed from slot ${event.slot+1}`;case'success':return 'All three blocks packed in the correct order';default:return event.text||event.type||JSON.stringify(event)}}
const agentName=name=>name==='picker'?'Picker':name==='packer'?'Packer':name||'Robot';
function humanExecution(event){
 const who=agentName(event.agent_id),object=event.object_id?` — ${event.object_id}`:'',target=event.target_id==='relay'?' at relay':String(event.target_id||'').startsWith('slot_')?` in slot ${Number(event.target_id.slice(5))+1}`:'';
 const reasons={wrong_object_held:'holding the wrong object',motion_limited:'requested motion was limited',no_observed_progress:'no observed progress',grasp_missed:'grasp missed'};
 switch(event.kind){
  case'started':return `${who}: instruction started${object}`;
  case'holding':return `${who}: holding${object} (observed)`;
  case'transfer_ready':return `${who}: transfer ready${object} (observed)`;
  case'completed':return `${who}: instruction completed${object}${target} (verified)`;
  case'blocked':return `${who}: blocked${object} · ${reasons[event.reason]||String(event.reason||'execution needs attention').replaceAll('_',' ')}`;
  default:return `${who}: ${String(event.kind||'execution update').replaceAll('_',' ')}${object}`;
 }
}
const feedbackAgents=new Set(names.filter(name=>records.some(record=>Array.isArray(record.agents?.[name]?.execution_events))));
const activity=[];
for(const [i,record]of records.entries()){
 for(const event of record.events||[]){if(event.type==='grasp_missed'&&feedbackAgents.has(event.agent))continue;activity.push({index:i+1,text:humanEvent(event)})}
 for(const name of names){
  const agent=record.agents?.[name],mail=agent?.outgoing;if(mail)activity.push({index:i+1,text:`${agentName(name)} → peer: ${mail.text}`});
  for(const event of agent?.execution_events||[])activity.push({index:i+1,text:humanExecution(event)});
  for(const event of agent?.peer_execution_events||[])activity.push({index:i+1,text:`${agentName(name)} received ${humanExecution(event)}`});
 }
}
function showActivity(){const entries=activity.filter(event=>event.index<=index).slice(-5).reverse();$('events').replaceChildren();text('event-count',entries.length?'Latest events':'Awaiting action');if(!entries.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='The workcell is ready. Play to follow the agents.';$('events').append(empty)}for(const event of entries){const line=document.createElement('div');line.className='event';const time=document.createElement('span');time.className='event-time';time.textContent=String(event.index).padStart(3,'0');const content=document.createElement('span');content.className='event-text';content.textContent=event.text;line.append(time,content);$('events').append(line)}}
function draw(state){
 const project=p=>[500+p[0]*480,335-p[1]*480],label=(value,x,y,color='#6b7e89')=>{ctx.font='13px system-ui';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(value,x,y)};
 ctx.fillStyle='#e5ecef';ctx.fillRect(0,0,1000,660);ctx.fillStyle='#f9fbfc';ctx.fillRect(92,108,816,450);ctx.strokeStyle='#c1ced4';ctx.strokeRect(92,108,816,450);
 ctx.fillStyle='#dcebec';ctx.fillRect(160,155,180,360);ctx.fillStyle='#efeadf';ctx.fillRect(650,155,175,360);
 label('01 / PICKER',252,85,'#087b85');label('SHARED RELAY',500,85);label('02 / PACKER',730,85,'#b86a13');
 for(const[id,p]of Object.entries(state.targets||{})){const[x,y]=project(p);ctx.strokeStyle='#8e9fa8';ctx.lineWidth=1.5;ctx.beginPath();if(id==='relay')ctx.arc(x,y,44,0,Math.PI*2);else ctx.rect(x-31,y-31,62,62);ctx.stroke();label(id==='relay'?'Relay':`Slot ${Number(id.slice(5))+1}`,x,y+56)}
 for(const[color,p]of Object.entries(state.objects||{})){const[x,y]=project(p);ctx.fillStyle=colors[color]||'#a0aeb5';ctx.fillRect(x-16,y-16,32,32);ctx.strokeStyle='#ffffff';ctx.strokeRect(x-16,y-16,32,32)}
 for(const[id,robot]of Object.entries(state.robots||{})){const[x,y]=project(robot.position);ctx.strokeStyle=id==='picker'?'#087b85':'#b86a13';ctx.lineWidth=3;ctx.strokeRect(x-25,y-22,50,44);ctx.beginPath();ctx.moveTo(x-9,y-13);ctx.lineTo(x-9,y+13);ctx.moveTo(x+9,y-13);ctx.lineTo(x+9,y+13);ctx.stroke();label(`${id}${robot.holding?' · '+robot.holding:''}`,x,y+75,ctx.strokeStyle)}
}
function render(){
 const record=index?records[index-1]:null,state=record?.state||initial;const packed=state.packed||[];
 $('timeline').value=index;text('time',`${clock(index/fps)} / ${clock(total)}`);text('step-count',`Decision ${index} / ${records.length}`);text('frame-label',`FRAME ${String(index).padStart(4,'0')}`);
 text('phase',record?`Decision ${record.tick} → state ${state.tick??index}`:'Initial state · agents ready');
 let correct=0;for(const[i,color]of order.entries()){const match=packed[i]===color;if(match)correct++;text(`slot-${i}`,match?'✓ Packed':packed[i]?`Contains ${packed[i]}`:'Waiting');$(`slot-${i}`).classList.toggle('done',match)}
 text('packed-count',`${correct} / ${order.length||3} correct`);text('task-state',state.success?'Complete':records.length&&index===records.length?'Step limit reached':'In progress');$('progress').style.width=`${100*correct/(order.length||3)}%`;
 for(const name of names){const agent=record?.agents?.[name],robot=state.robots?.[name];
 text(`${name}-instruction`,agent?.instruction?.text||'Ready for the first instruction');text(`${name}-inbox`,agent?.inbox?.length?agent.inbox.map(message).join('\n\n'):'No message this decision');text(`${name}-outgoing`,agent?.outgoing?message(agent.outgoing):agent?.attempted_message?`${agent.attempted_message}\nChannel disabled · not delivered`:'No message this decision');
 $(`${name}-inbox`).classList.toggle('empty',!agent?.inbox?.length);$(`${name}-outgoing`).classList.toggle('empty',!agent?.outgoing&&!agent?.attempted_message);
 text(`${name}-execution`,agent?.execution_events?.length?agent.execution_events.map(humanExecution).join('\n'):'No new execution status');text(`${name}-peer-execution`,agent?.peer_execution_events?.length?agent.peer_execution_events.map(humanExecution).join('\n'):'No peer status this decision');
 $(`${name}-execution`).classList.toggle('empty',!agent?.execution_events?.length);$(`${name}-peer-execution`).classList.toggle('empty',!agent?.peer_execution_events?.length);
 $(`${name}-feedback-loop`).hidden=!run.feedback;
 text(`${name}-model-context`,agent?.prediction?.model_prompt||[agent?.instruction?.text,agent?.instruction?.feedback].filter(Boolean).join('\n')||'Awaiting the first decision');
 text(`${name}-action-feedback`,agent?.action_feedback?.report?.feedback||'Awaiting an observed action result');
 const telemetry=$(`${name}-telemetry`);telemetry.replaceChildren();for(const[label,value]of [['EE',(robot?.position||[]).map(n=>Number(n).toFixed(2)).join(', ')||'—'],['Holding',robot?.holding||'—']]){const span=document.createElement('span'),strong=document.createElement('strong');span.append(`${label} `);strong.textContent=value;span.append(strong);telemetry.append(span)}
 text(`${name}-raw`,agent?JSON.stringify({policy:agent.policy||policy,observation:agent.observation,instruction:agent.instruction,action:agent.action,executed_delta:agent.executed_delta,workspace_clipped:agent.workspace_clipped,prediction:agent.prediction,action_feedback:agent.action_feedback,execution_events:agent.execution_events||[],peer_execution_events:agent.peer_execution_events||[]},null,2):'No decision recorded yet');
 }
 if(setting('planner')==='llm'){const m=record?.mediator;text('mediator-call',m?`Decision ${m.call_id}`:'Ready');text('mediator-plan',m?names.map(name=>`${agentName(name)}: ${record.agents[name].instruction.text}`).join(' '):'Awaiting the first model decision.');text('mediator-reason',m?.reason||'Awaiting the first model decision.');text('mediator-trigger',m?`Triggered by ${String(m.trigger).replaceAll('_',' ')} at tick ${m.tick}.`:'Plans remain stable while the robots execute.');}
 text('event-raw',record?JSON.stringify({tick:record.tick,events:record.events,execution_events:names.flatMap(name=>record.agents?.[name]?.execution_events||[]),received_peer_events:Object.fromEntries(names.map(name=>[name,record.agents?.[name]?.peer_execution_events||[]]))},null,2):'Initial state');showActivity();if(!videoReady)draw(state);
 $('back').disabled=index===0;$('next').disabled=index===records.length;$('play').disabled=!records.length;
}
function setPlay(value){playing=value;text('play',value?'Ⅱ Pause':'▶ Play');$('play').setAttribute('aria-label',value?'Pause replay':'Play replay')}
function pause(){setPlay(false);if(video)video.pause()}
function seek(value){index=Math.max(0,Math.min(records.length,value));if(videoReady)video.currentTime=index/fps+0.00001;render()}
async function play(){if(!records.length)return;if(index===records.length)seek(0);lastClock=performance.now();accumulator=0;setPlay(true);if(videoReady){try{await video.play()}catch{pause();text('phase','Video playback unavailable. Use the timeline to inspect the recorded states.')}}}
function frame(now){if(playing){if(videoReady){const next=Math.min(records.length,Math.floor(video.currentTime*fps+0.0001));if(next!==index){index=next;render()}}else{accumulator+=(now-lastClock)*Number($('speed').value)*fps/1000;if(accumulator>=1){index=Math.min(records.length,index+Math.floor(accumulator));accumulator%=1;render();if(index===records.length)pause()}}}lastClock=now;requestAnimationFrame(frame)}
$('play').onclick=()=>playing?pause():play();$('back').onclick=()=>{pause();seek(index-1)};$('next').onclick=()=>{pause();seek(index+1)};$('timeline').oninput=event=>{pause();seek(Number(event.target.value))};$('speed').onchange=()=>{if(video)video.playbackRate=Number($('speed').value)};
if(video){$('viewport').classList.add('has-video');const ready=()=>{videoReady=true;video.playbackRate=Number($('speed').value);seek(index)};video.addEventListener('loadedmetadata',ready);if(video.readyState>=1)ready();video.addEventListener('ended',()=>{pause();index=records.length;render()});video.addEventListener('error',()=>{pause();videoReady=false;video.hidden=true;$('viewport').classList.remove('has-video');text('view-type','State replay · video unavailable');text('view-label','RECORDED STATE / SCHEMATIC');render()})}
else{text('view-type','Recorded state · no video');text('view-label','RECORDED STATE / SCHEMATIC')}
render();requestAnimationFrame(frame);
</script></body></html>
"""
