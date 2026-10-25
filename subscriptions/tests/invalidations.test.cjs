const assert = require('node:assert/strict');
const {test} = require('node:test');
const {Invalidations} = require('../src/invalidations.cjs');
test('only approved topics and fixed statuses cross the bridge', () => {
  const lines = [];
  const feed = new Invalidations(line => lines.push(JSON.parse(line)));
  feed.status('ready');
  for (let i = 0; i < 1000; i++) feed.add('tasks');
  feed.add('workers'); feed.add('secret-token'); feed.flush();
  assert.deepEqual(lines, [{type:'status',status:'ready'}, {type:'update',topics:['tasks','workers']}]);
});
test('snapshot callbacks do not publish before subscription acknowledgement', () => {
  const lines=[]; const feed=new Invalidations(line=>lines.push(JSON.parse(line)));
  feed.add('tasks'); feed.flush();
  assert.equal(lines.length, 0);
  feed.status('ready');
  assert.equal(lines.length, 1);
});
