const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const modes = ['embeddings','text','none'];
const labels = {embeddings:'Learned embeddings',text:'Structured text',none:'No communication'};
const descriptions = {embeddings:'The receiver decodes the sender’s learned visual representation.',text:'The sender decodes the same visual representation into exact route text.',none:'The receiver uses a fixed A prior. Local perception still runs.'};
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not measured';
const number = value => Number.isFinite(value) ? value.toLocaleString() : '—';
const check = (passed,label,detail) => `<div class="pairing-row"><span class="check-icon ${passed?'pass':'fail'}">${passed?'✓':'!'}</span><div><strong>${esc(label)}</strong><small>${esc(detail)}</small></div></div>`;

function renderIntegration(integration){
  const section=$('comparison-integration');
  section.hidden=!integration||!Array.isArray(integration.runs)||!integration.aggregate;
  if(section.hidden)return;
  const planned=(integration.seeds?.length||3)*modes.length;
  const recorded=integration.runs.length, incomplete=recorded<planned;
  $('integration-status').textContent=`${incomplete?'In progress · ':''}${recorded} / ${planned} runs recorded`;
  $('integration-status').className='status-pill '+(incomplete?'planning':'succeeded');
  $('integration-modes').innerHTML=modes.map(mode=>{
    const a=integration.aggregate[mode]||{};
    return `<article class="mode-card ${mode}"><div class="mode-label"><i></i>${esc(labels[mode])}</div><div class="mode-score">${percent(a.task_success_rate)}<span>complete task success</span></div><dl><div><dt>Complete tasks</dt><dd>${number(a.successful_tasks)} / ${number(a.runs)}</dd></div><div><dt>Correct routes</dt><dd>${number(a.correct_routes)} / ${number(a.total_routes)}</dd></div><div><dt>Physically delivered</dt><dd>${number(a.physically_completed)} / ${number(a.total_routes)}</dd></div></dl></article>`;
  }).join('');
  $('integration-progress').textContent=incomplete
    ? 'Interim integration results. Channel counts may differ until all paired seeds finish. Refresh for the latest saved runs.'
    : `${integration.paired_checks?.length||0} complete paired seeds. This small suite checks the live planner and database path; it does not establish a channel latency advantage.`;
  if(integration.failure_note)$('integration-progress').textContent+=' '+integration.failure_note;
  if(integration.transport_recheck)$('integration-progress').textContent+=' Separate unchanged-condition recheck: '+integration.transport_recheck.status+' ('+integration.transport_recheck.correct_routes+'/3 routes). The original failure remains in the comparison.';
}

function render(data){
  if(data.schema!=='relay.factory.communication-benchmark.v1'||!Array.isArray(data.runs)||!data.aggregate)throw new Error('The attached result file is not a compatible communication comparison.');
  const expected=(data.seeds?.length||0)*3, runs=data.runs, pairs=data.paired_checks||[];
  const incomplete=expected&&runs.length<expected;
  $('comparison-status').textContent=incomplete?`In progress · ${runs.length} / ${expected}`:`${runs.length} / ${expected||runs.length} runs recorded`;
  $('comparison-status').className='status-pill '+(incomplete?'planning':'succeeded');
  $('comparison-progress-note').hidden=!incomplete;$('comparison-progress-note').textContent=`Interim results: ${runs.length} of ${expected} planned runs are recorded. Channel counts may differ until every paired seed is complete. Refresh to read the latest saved results.`;
  $('protocol-strip').innerHTML=[['Task','Private dispatch'],['Orders / run',data.orders],['Paired seeds',data.seeds?.length??'—'],['Tick budget',number(data.max_ticks)],['Planner',data.planner==='llm'?'Qwen mediator':'Rules reference'],['Coordination',data.coordination==='spacetime'?'SpacetimeDB':'Memory reference']].map(([key,value])=>`<div><span>${esc(key)}</span><strong>${esc(value)}</strong></div>`).join('');
  $('comparison-modes').innerHTML=modes.map(mode=>{
    const a=data.aggregate[mode]||{}, interval=a.task_success_wilson_95;
    return `<article class="mode-card ${mode}"><div class="mode-label"><i></i>${esc(labels[mode])}</div><p>${esc(descriptions[mode])}</p><div class="mode-score">${percent(a.task_success_rate)}<span>complete task success</span></div><div class="mode-interval">${a.runs&&Array.isArray(interval)?`95% Wilson interval ${percent(interval[0])}–${percent(interval[1])}`:'More runs needed for an interval'}</div><dl><div><dt>Complete tasks</dt><dd>${number(a.successful_tasks)} / ${number(a.runs)}</dd></div><div><dt>Correct routes</dt><dd>${number(a.correct_routes)} / ${number(a.total_routes)}</dd></div><div><dt>Physically delivered</dt><dd>${number(a.physically_completed)} / ${number(a.total_routes)}</dd></div><div><dt>Failed skills</dt><dd>${number(a.failed_skills)}</dd></div><div><dt>Mean simulation ticks</dt><dd>${Number.isFinite(a.mean_ticks)?number(Math.round(a.mean_ticks)):'—'}</dd></div><div><dt>Published packets</dt><dd>${number(a.packet_count)}</dd></div><div><dt>JSON content / cue</dt><dd>${Number.isFinite(a.source_payload_bytes)?number(a.source_payload_count?Math.round(a.source_payload_bytes/a.source_payload_count):0)+' B':'Not recorded'}</dd></div></dl></article>`;
  }).join('');
  $('outcome-bars').innerHTML=modes.map(mode=>{
    const a=data.aggregate[mode]||{};
    return `<div class="outcome-mode ${mode}"><strong>${esc(labels[mode])}</strong>${[['Task success',a.task_success_rate],['Route accuracy',a.route_accuracy]].map(([label,value])=>`<div class="outcome-row"><span>${label}</span><div class="bar-track" role="img" aria-label="${esc(labels[mode])} ${label}: ${percent(value)}"><i style="width:${Number.isFinite(value)?Math.min(100,Math.max(0,value*100)):0}%"></i></div><b>${percent(value)}</b></div>`).join('')}</div>`;
  }).join('');
  const checks=[['identical_initial_physics','Identical starting physics'],['identical_private_sensor_inputs','Identical private RGB observations'],['identical_actor_weights','Identical motor-policy weights'],['identical_codec_weights','Identical cue-codec weights'],['embedding_text_decisions_equal','Embedding/text decision parity']];
  $('pairing-checks').innerHTML=pairs.length?checks.map(([key,label])=>{const count=pairs.filter(p=>p[key]===true).length;return check(count===pairs.length,label,`${count} / ${pairs.length} complete seed groups match`);}).join('')+check(pairs.every(p=>p.none_packets===0),'No-communication isolation',`${pairs.filter(p=>p.none_packets===0).length} / ${pairs.length} groups publish zero packets`):'<p>No complete three-mode seed group has been recorded yet.</p>';
  $('protocol-parity').textContent=data.parity||'The same learned cue prediction is delivered through two representations.';
  $('protocol-partial').textContent=data.partial_information||'';
  $('run-count').textContent=`${pairs.length} complete paired seeds · ${runs.length} retained runs`;
  $('comparison-run-rows').innerHTML=[...runs].sort((a,b)=>a.seed-b.seed||modes.indexOf(a.mode)-modes.indexOf(b.mode)).map(run=>`<tr><td>${esc(run.seed)}</td><td><span class="table-channel ${esc(run.mode)}">${esc(labels[run.mode]||run.mode)}</span></td><td><span class="status-pill ${esc(run.status)}">${esc(run.status==='rejected'?'Routing mismatch':run.status)}</span>${run.error?`<small class="run-error">${esc(run.error)}</small>`:''}</td><td>${number(run.completed)} / ${number(run.orders)}</td><td>${number(run.correct_routes)} / ${number(run.orders)}</td><td>${number(run.ticks)}</td><td>${number(run.communication?.published)}</td><td>${number(run.failed_skills)}</td></tr>`).join('');
  renderIntegration(data.integration);
  $('comparison-limit-list').innerHTML=(data.limitations||[]).map(value=>`<li>${esc(value)}</li>`).join('');
  const actorHashes=new Set(runs.map(run=>run.actor_sha256));
  const models=document.createElement('li');models.textContent=`${actorHashes.size} motor checkpoint${actorHashes.size===1?'':'s'} represented. ${pairs.length} fully paired seed groups are available; incomplete groups are shown but excluded from pairing checks.`;$('comparison-limit-list').append(models);
  $('comparison-updated').textContent=data.created_at?`Protocol created ${new Date(data.created_at).toLocaleString()}`:'';
  $('comparison-content').hidden=false;$('comparison-empty').hidden=true;
}

async function load(){
  $('refresh-comparison').disabled=true;
  try{
    const response=await fetch('/api/factory/comparison',{cache:'no-store',signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error(response.status===404?'No recorded comparison is attached to this workspace yet. Return after a paired evaluation has finished.':`Results are unavailable (${response.status}).`);
    render(await response.json());
  }catch(error){$('comparison-empty').textContent=error.message;$('comparison-empty').hidden=false;$('comparison-content').hidden=true;$('comparison-status').textContent='Results unavailable';}
  finally{$('refresh-comparison').disabled=false;}
}
$('refresh-comparison').onclick=load;
load();
