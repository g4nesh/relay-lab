'use strict';
const TOPICS = new Set(['tasks', 'workers', 'coordination']);
class Invalidations {
  constructor(write) { this.write = write; this.pending = new Set(); this.timer = null; this.ready = false; }
  emit(value) { this.write(JSON.stringify(value) + '\n'); }
  add(topic) {
    if (!TOPICS.has(topic)) return;
    this.pending.add(topic);
    if (!this.timer) this.timer = setTimeout(() => this.flush(), 100);
  }
  flush() {
    clearTimeout(this.timer); this.timer = null;
    if (this.ready && this.pending.size) this.emit({type: 'update', topics: [...this.pending].sort()});
    this.pending.clear();
  }
  status(status) { this.ready = status === 'ready'; this.pending.clear(); this.emit({type: 'status', status}); }
  heartbeat() { this.emit({type: 'heartbeat'}); }
}
module.exports = { Invalidations };
