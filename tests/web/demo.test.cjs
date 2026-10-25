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
  for(const p of ['/api/bootstrap','/api/workspace','/api/models','/api/tasks','/api/workers','/api/team','/api/requests','/api/audit','/api/health','/api/coordination','/api/analytics']) {
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
  for(const p of ['/api/workspace','/api/tasks','/api/workers','/api/team/invitations','/api/models/demo-model-left/release','/api/coordination/jobs','/api/auth/setup']) {
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
