const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const web = path.resolve(__dirname, '../../src/relay_lab/studio/web');
function context() {
  const ctx = vm.createContext({window:{}, URLSearchParams, structuredClone});
  vm.runInContext(fs.readFileSync(path.join(web, 'demo.js'), 'utf8'), ctx);
  return ctx;
}
test('hosted defaults to demo; explicit workspace preserves real login and local keeps profiles', () => {
  const demo=context().window.RelayDemo;
  assert.equal(demo.shouldEnter({hostname:'relay-studio.onrender.com', search:''}),true);
  assert.equal(demo.shouldEnter({hostname:'relay-studio.onrender.com', search:'?workspace=1'}),false);
  for(const hostname of ['localhost','127.0.0.1','[::1]']) {
    assert.equal(demo.shouldEnter({hostname,search:''}),false);
    assert.equal(demo.shouldEnter({hostname,search:'?demo=1'}),true);
    assert.equal(demo.shouldEnter({hostname,search:'?demo=1&workspace=1'}),false);
  }
});
test('all sample read paths work offline and return detached labeled records', async () => {
  const demo=context().window.RelayDemo;
  for(const p of ['/api/bootstrap','/api/workspace','/api/models','/api/tasks','/api/schedules','/api/workers','/api/team','/api/requests','/api/audit','/api/health','/api/coordination','/api/analytics']) {
    const data=await demo.request(p);
    assert.equal(data.demo,true); assert.match(data.provenance,/Illustrative/);
  }
  const first=await demo.request('/api/bootstrap');
  assert.equal(first.token,''); assert.equal(first.capabilities.experiments_enabled,false);
  assert.equal(first.permissions.includes('models.release'),false);
  first.workspace.agents[0].name='changed';
  assert.notEqual((await demo.request('/api/bootstrap')).workspace.agents[0].name,'changed');
});
test('sample task details agree with list and carry explicit nonexecution evidence', async () => {
  const demo=context().window.RelayDemo;
  for(const t of (await demo.request('/api/tasks')).tasks) {
    const detail=await demo.request(`/api/tasks/${t.id}`);
    assert.equal(detail.status,t.status); assert.equal(detail.readback_verified,false);
    assert.ok(detail.events.length); assert.equal(Object.keys(detail.assignments).length,2);
    assert.equal(detail.assignments.left.evidence.kind,'illustrative');
  }
});
test('demo denies every write and unknown/private record without fallback', async () => {
  const demo=context().window.RelayDemo;
  for(const p of ['/api/workspace','/api/tasks','/api/schedules','/api/workers','/api/team/invitations','/api/models/demo-model-left/release','/api/coordination/jobs','/api/auth/setup']) {
    await assert.rejects(demo.request(p,{}),/read-only/);
  }
  for(const p of ['/api/auth/status','/api/tasks/goal-deadbeefdeadbeef','/api/bootstrap?private=1','/__proto__']) await assert.rejects(demo.request(p),/not part/);
});
test('application API stays offline in demo and uses the unchanged real transport outside it', async () => {
  const ctx=context(); let calls=0;
  ctx.fetch=async()=>{calls++; return {ok:true,json:async()=>({real:true})};};
  const source=fs.readFileSync(path.join(web,'studio.js'),'utf8').replace(/\ninit\(\);\s*$/,'');
  vm.runInContext(source,ctx);
  vm.runInContext('state.demo = true',ctx);
  assert.equal((await vm.runInContext('api("/api/bootstrap")',ctx)).demo,true);
  await assert.rejects(vm.runInContext('api("/api/workers", {})',ctx),/read-only/);
  assert.equal(calls,0);
  vm.runInContext('state.demo = false',ctx);
  assert.equal((await vm.runInContext('api("/api/bootstrap")',ctx)).real,true);
  assert.equal(calls,1);
});
test('demo starts neither a private SSE stream nor a polling loop', () => {
  let calls=0, timers=0;
  const node={classList:{remove(){}},textContent:''};
  const ctx=vm.createContext({window:{},document:{hidden:false,getElementById:()=>node},location:{hostname:'example.com'},clearTimeout(){},setTimeout(){timers++;},EventSource:function(){calls++;}});
  const source=fs.readFileSync(path.join(web,'tasks.js'),'utf8').replace('window.RelayTasks =', 'window.demoProbe = () => {options = {isDemo:()=>true,authenticated:()=>true}; active = true; startEvents(); schedule();};\n  window.RelayTasks =');
  vm.runInContext(source,ctx); ctx.window.demoProbe();
  assert.equal(calls,0); assert.equal(timers,0); assert.match(node.textContent,/Illustrative/);
});

function timingContext() {
  const nodes = new Map();
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {value:'', disabled:false, hidden:false, textContent:'', listeners:{}, addEventListener(type,listener){this.listeners[type]=listener;}});
    return nodes.get(id);
  };
  const controls = ['tk-objective','tk-title','tk-criteria','tk-target','tk-deadline','tk-start','tk-start-at','tk-repeat','tk-runs'];
  node('tk-create-form').querySelectorAll = () => controls.map(node);
  for (const [id,value] of Object.entries({'tk-objective':'Fold the shirt','tk-criteria':'Shirt is folded','tk-target':'both','tk-deadline':'1800','tk-start':'now','tk-repeat':'0','tk-runs':'3'})) node(id).value=value;
  let now=Date.UTC(2026,9,25,12), sequence=0;
  class Clock extends Date {
    constructor(...args) { super(...(args.length ? args : [now])); }
    static now() { return now; }
  }
  const ctx=vm.createContext({window:{},document:{getElementById:node,addEventListener(){}},Date:Clock,crypto:{randomUUID:()=> `test-request-${++sequence}`}});
  const source=fs.readFileSync(path.join(web,'tasks.js'),'utf8').replace('window.RelayTasks =', `
    window.timingProbe = {
      render: renderTiming,
      submit: createTask,
      configure(api) {
        const models = ['left','right'].map(id=>({spec:{id},status:'released'}));
        mount({api, authenticated:()=>true, context:()=>({models,workspace:{agents:models.map(m=>m.spec)}})});
        loaded = true;
      }
    };
    window.RelayTasks =`);
  vm.runInContext(source,ctx);
  return {node,probe:ctx.window.timingProbe,advance:(ms)=>{now+=ms;}};
}

test('inactive scheduling controls are excluded from form validation', () => {
  const {node,probe}=timingContext();
  node('tk-start').value='later'; node('tk-repeat').value='60'; probe.render();
  assert.equal(node('tk-start-at').disabled,false);
  assert.equal(node('tk-runs').disabled,false);
  // A number below the native min must not block an unrelated immediate task.
  node('tk-runs').value='1';
  node('tk-start').value='now'; node('tk-repeat').value='0'; probe.render();
  assert.equal(node('tk-start-at-field').hidden,true);
  assert.equal(node('tk-start-at').disabled,true);
  assert.equal(node('tk-runs-field').hidden,true);
  assert.equal(node('tk-runs').disabled,true);
  node('tk-repeat').value='60'; probe.render();
  assert.equal(node('tk-runs').disabled,false);
  assert.equal(node('tk-start-at').disabled,true);
});

test('failed task submission restores timing-specific disabled states', async () => {
  const {node,probe}=timingContext();
  probe.configure(async () => { throw new Error('Connection interrupted'); });
  probe.render();
  await probe.submit({preventDefault(){}});
  assert.match(node('tk-create-error').textContent,/Connection interrupted/);
  assert.equal(node('tk-objective').disabled,false);
  assert.equal(node('tk-start').disabled,false);
  assert.equal(node('tk-repeat').disabled,false);
  assert.equal(node('tk-start-at').disabled,true);
  assert.equal(node('tk-runs').disabled,true);
});

test('scheduled retries preserve exact timing; editing starts a new request', async () => {
  const {node,probe,advance}=timingContext(), calls=[];
  probe.configure(async (url,payload) => {
    calls.push({url,payload:JSON.parse(JSON.stringify(payload))});
    throw new Error('Response lost after commit');
  });
  node('tk-repeat').value='60'; probe.render();
  await probe.submit({preventDefault(){}});
  advance(120000);
  await probe.submit({preventDefault(){}});
  assert.equal(calls.length,2);
  assert.equal(calls[0].url,'/api/schedules');
  assert.deepEqual(calls[1],calls[0]);
  node('tk-objective').value='Fold the next shirt';
  node('tk-create-form').listeners.input();
  await probe.submit({preventDefault(){}});
  assert.equal(calls.length,3);
  assert.notEqual(calls[2].payload.request_id,calls[0].payload.request_id);
  assert.equal(calls[2].payload.start_at,calls[0].payload.start_at+120);
  assert.equal(calls[2].payload.objective,'Fold the next shirt');
});

test('a scheduled start that passed can still retry its original request', async () => {
  const {node,probe,advance}=timingContext(), calls=[];
  probe.configure(async (_url,payload) => {
    calls.push(JSON.parse(JSON.stringify(payload)));
    throw new Error('Response lost after commit');
  });
  node('tk-start').value='later'; probe.render();
  await probe.submit({preventDefault(){}});
  advance(15*60000);
  await probe.submit({preventDefault(){}});
  assert.equal(calls.length,2);
  assert.deepEqual(calls[1],calls[0]);
  assert.match(node('tk-create-error').textContent,/Response lost/);
});
