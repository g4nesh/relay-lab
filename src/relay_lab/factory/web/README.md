# Factory viewer

The original Three.js factory meshes and interface are built from the shared factory scene contract. Robot poses, inverse-kinematic joint points, cargo custody and mission results come from the backend. Display interpolation smooths committed poses; it does not execute actions or invent outcomes.

The dashboard has no CDN or external media dependencies. `app.bundle.js` is checked in so the Python server works without Node. To rebuild after editing `app.js`:

```sh
npm ci --ignore-scripts
npm run build
```

Dependencies are pinned in `package-lock.json`. Three.js 0.180.0 is MIT licensed; its license is reproduced in `THIRD_PARTY_LICENSES.txt`. All factory objects are original procedural geometry.

The Python server serves only the four browser assets: `index.html`, `styles.css`, `app.bundle.js`, and an optional favicon. `node_modules`, package metadata and source files are not exposed by the server. Controls call the loopback-only `/api/factory/` endpoints; state is polled every 300 ms and errors remain visible. The 3D renderer requires WebGL 2.

Camera controls: drag to orbit, wheel to zoom, right-drag to pan. Select a robot in the fleet list, click a robot label, or click its body to inspect; use **Focus in environment** for a close-up. The three view presets restore overview, floor-level or top cameras. Click a communication row to inspect its causal envelope and payload.
