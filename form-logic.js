// IRONFIST — Form logic
// Pure maths for the camera form checker: no DOM, no camera, no network.
// Turns a stream of noisy pose keypoints (from MoveNet) into stable rep counts and form feedback.
// Loaded by form-check.html; also runs under Node for automated tests.
//
// What makes the counting reliable:
//  1. Keypoint confidence gating   - joints the model is unsure about are ignored, never guessed.
//  2. A sticky "working side"      - the limb nearest the camera is used consistently instead of flipping
//                                    between left and right whenever their confidence flickers.
//  3. Spike rejection + smoothing  - median-of-3 then a time-based moving average on every signal.
//  4. A progress signal (0..1)     - every exercise is reduced to "how far through the movement am I".
//  5. A strict rep state machine   - a rep counts only after: standing/top position held -> descends ->
//                                    reaches full depth -> returns to the top, within a sane time.
//                                    Shallow reps and fidgeting are NOT counted.
//  6. Tracking-loss handling       - if the body is lost, the rep in progress is discarded and the counter
//                                    must see the top position again before counting resumes.
//  7. Form faults need agreement   - a fault only flags when most frames over ~0.15 s agree, so one bad frame can't spoil a rep.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FormLogic = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var MIN_SCORE = 0.3;      // keypoints below this confidence are treated as "not seen"
  var LOST_MS = 600;        // body unseen this long => discard the rep in progress

  // ---------- small helpers ----------
  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function dist(a, b) { var dx = a.x - b.x, dy = a.y - b.y; return Math.sqrt(dx * dx + dy * dy); }

  function angleAt(a, b, c) { // angle at b, degrees 0..180
    var v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y;
    var m1 = Math.sqrt(v1x * v1x + v1y * v1y), m2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (!m1 || !m2) return null;
    return Math.acos(clamp((v1x * v2x + v1y * v2y) / (m1 * m2), -1, 1)) * 180 / Math.PI;
  }

  function indexKeypoints(keypoints) {
    var m = {};
    for (var i = 0; i < keypoints.length; i++) m[keypoints[i].name] = keypoints[i];
    return m;
  }

  function pt(m, name, min) { // a keypoint, only if the model is confident about it
    var p = m[name];
    return (p && typeof p.score === 'number' && p.score >= (min || MIN_SCORE)) ? p : null;
  }

  function median3(a, b, c) { return Math.max(Math.min(a, b), Math.min(Math.max(a, b), c)); }

  // Time-aware smoother: median-of-3 (kills single-frame spikes) then an exponential moving average.
  function Smoother(tauMs) { this.tau = tauMs; this.reset(); }
  Smoother.prototype.reset = function () { this.v = null; this.t = null; this.h = []; };
  Smoother.prototype.update = function (x, t) {
    this.h.push(x); if (this.h.length > 3) this.h.shift();
    var m = this.h.length === 3 ? median3(this.h[0], this.h[1], this.h[2]) : x;
    if (this.v === null) this.v = m;
    else this.v += (1 - Math.exp(-Math.max(1, t - this.t) / this.tau)) * (m - this.v);
    this.t = t;
    return this.v;
  };

  // Picks the limb side the camera sees best, with hysteresis so it doesn't flip-flop.
  function SidePicker() { this.l = null; this.r = null; this.side = 'right'; }
  SidePicker.prototype.update = function (m, joints) {
    var ls = 0, rs = 0, i;
    for (i = 0; i < joints.length; i++) {
      ls += (m['left_' + joints[i]] ? m['left_' + joints[i]].score : 0);
      rs += (m['right_' + joints[i]] ? m['right_' + joints[i]].score : 0);
    }
    ls /= joints.length; rs /= joints.length;
    if (this.l === null) { this.l = ls; this.r = rs; this.side = rs >= ls ? 'right' : 'left'; }
    else {
      this.l += 0.08 * (ls - this.l); this.r += 0.08 * (rs - this.r);
      if (this.side === 'right' && this.l > this.r + 0.12) this.side = 'left';
      else if (this.side === 'left' && this.r > this.l + 0.12) this.side = 'right';
    }
    return this.side;
  };

  function other(side) { return side === 'left' ? 'right' : 'left'; }

  // ---------- body measurements ----------
  function jointAngle(m, side, a, b, c, min) {
    var pa = pt(m, side + '_' + a, min), pb = pt(m, side + '_' + b, min), pc = pt(m, side + '_' + c, min);
    return (pa && pb && pc) ? angleAt(pa, pb, pc) : null;
  }

  // Torso lean away from vertical in degrees (0 = upright).
  function torsoLean(m, side) {
    var s = pt(m, side + '_shoulder'), h = pt(m, side + '_hip');
    if (!s || !h) { side = other(side); s = pt(m, side + '_shoulder'); h = pt(m, side + '_hip'); }
    if (!s || !h) return null;
    var len = dist(s, h);
    if (!len) return null;
    return Math.acos(clamp(-(s.y - h.y) / len, -1, 1)) * 180 / Math.PI;
  }

  // For floor exercises seen side-on. Returns null if the body isn't roughly horizontal in view,
  // else { sag } where sag>0 = hips below the shoulder-ankle line, sag<0 = hips above it (relative to its length).
  function bodyLine(m, side) {
    var s = pt(m, side + '_shoulder'), h = pt(m, side + '_hip'), a = pt(m, side + '_ankle');
    if (!s || !h) { side = other(side); s = pt(m, side + '_shoulder'); h = pt(m, side + '_hip'); a = pt(m, side + '_ankle'); }
    if (!s || !h) return null;
    var far = a || h;
    var len = dist(s, far);
    if (len < 1 || Math.abs(far.x - s.x) < 0.6 * len) return null; // not lying roughly horizontal
    if (!a || Math.abs(a.x - s.x) < 0.5 * dist(s, a)) return { sag: null };
    var lineY = s.y + (a.y - s.y) * (h.x - s.x) / (a.x - s.x);
    return { sag: (h.y - lineY) / dist(s, a) };
  }

  // ---------- rep counter ----------
  // p is progress 0 (top / start position) .. 1 (bottom / fully contracted).
  function RepCounter(cfg) { this.cfg = cfg; this.reps = 0; this.clean = 0; this.reset(); }
  RepCounter.prototype.reset = function () {
    this.state = 'waiting'; this.topSince = null; this.startT = 0; this.maxP = 0;
    this.reachedDepth = false; this.faults = {}; this.nF = 0; this.fC = {};
  };
  // returns null, or { type:'rep', clean, faults:[...] } / { type:'shallow' }
  RepCounter.prototype.update = function (p, t, faultsActive) {
    var c = this.cfg, i;
    if (this.state === 'waiting') {
      if (p <= c.arm) {
        if (this.topSince === null) this.topSince = t;
        if (t - this.topSince >= 250) this.state = 'ready';
      } else this.topSince = null;
      return null;
    }
    if (this.state === 'ready') {
      if (p >= c.start) {
        this.state = 'active'; this.startT = t; this.maxP = p; this.reachedDepth = false; this.faults = {}; this.nF = 0; this.fC = {};
      }
      return null;
    }
    // active
    if (p > this.maxP) this.maxP = p;
    if (this.maxP >= c.depth) this.reachedDepth = true;
    if (p >= 0.4) { // evidence: share of the working part of the rep in which each fault was seen
      this.nF++;
      for (i = 0; i < faultsActive.length; i++) this.fC[faultsActive[i]] = (this.fC[faultsActive[i]] || 0) + 1;
    }
    var dur = t - this.startT;
    if (dur > c.maxRepMs) { this.reset(); return null; } // stuck mid-rep: start over from the top
    if (p <= c.end) {
      this.state = 'ready';
      if (this.reachedDepth && dur >= c.minRepMs) {
        var f = [], k;
        for (k in this.fC) if (this.nF >= 3 && this.fC[k] / this.nF >= 0.3) f.push(k);
        this.reps++;
        if (!f.length) this.clean++;
        return { type: 'rep', clean: f.length === 0, faults: f };
      }
      if (this.maxP >= c.shallow) return { type: 'shallow' };
    }
    return null;
  };

  // ---------- exercise definitions ----------
  var BASE = { arm: 0.25, start: 0.45, end: 0.3, depth: 0.7, shallow: 0.5, minRepMs: 250, maxRepMs: 12000 };
  function cfg(over) { var o = {}, k; for (k in BASE) o[k] = BASE[k]; for (k in over) o[k] = over[k]; return o; }

  // Each measure() returns { valid, p, angle, faults:{name:true}, hint }
  function measureSquat(m, st) {
    var side = st.picker.update(m, ['hip', 'knee', 'ankle']);
    var ang = jointAngle(m, side, 'hip', 'knee', 'ankle');
    if (ang === null) ang = jointAngle(m, other(side), 'hip', 'knee', 'ankle');

    var ls = pt(m, 'left_shoulder'), rs = pt(m, 'right_shoulder'), lh = pt(m, 'left_hip'), rh = pt(m, 'right_hip');
    var lk = pt(m, 'left_knee'), rk = pt(m, 'right_knee'), la = pt(m, 'left_ankle'), ra = pt(m, 'right_ankle');

    // How "front-on" the camera view is (0 = side-on, 1 = facing the camera).
    var w = 0;
    if (ls && rs && lh && rh) {
      var torso = dist({ x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 }, { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 });
      if (torso > 1) w = clamp((Math.abs(ls.x - rs.x) / torso - 0.3) / 0.3, 0, 1);
    }
    // View-independent depth: how far the hips have dropped toward knee height.
    var pHip = null;
    if (lh && rh && lk && rk && la && ra) {
      var hy = (lh.y + rh.y) / 2, ky = (lk.y + rk.y) / 2, ay = (la.y + ra.y) / 2;
      if (ay - hy > 10) pHip = clamp((0.5 - (ky - hy) / (ay - hy)) / 0.42, 0, 1);
    }
    var pKnee = ang === null ? null : clamp((170 - ang) / 80, 0, 1);

    var p;
    if (pKnee !== null && pHip !== null) p = (1 - w) * pKnee + w * pHip;
    else if (pKnee !== null && w < 0.5) p = pKnee;
    else if (pHip !== null) p = pHip;
    else return { valid: false };

    var faults = {};
    var lean = torsoLean(m, side);
    if (w < 0.5 && lean !== null && lean > 70 && p > 0.3) faults.lean = true;
    if (w > 0.5 && lk && rk && la && ra && p > 0.4) {
      var aw = Math.abs(la.x - ra.x);
      if (aw > 10 && Math.abs(lk.x - rk.x) / aw < 0.85) faults.cave = true;
    }
    return { valid: true, p: p, angle: ang, faults: faults };
  }

  function measurePushup(m, st) {
    var side = st.picker.update(m, ['shoulder', 'elbow', 'wrist']);
    var ang = jointAngle(m, side, 'shoulder', 'elbow', 'wrist');
    if (ang === null) ang = jointAngle(m, other(side), 'shoulder', 'elbow', 'wrist');
    if (ang === null) return { valid: false };
    var body = bodyLine(m, side);
    if (!body) return { valid: false, hint: 'Get side-on to the camera, in push-up position, whole body in frame.' };
    var faults = {};
    if (body.sag !== null && body.sag > 0.10) faults.sag = true;
    if (body.sag !== null && body.sag < -0.10) faults.pike = true;
    return { valid: true, p: clamp((165 - ang) / 80, 0, 1), angle: ang, faults: faults };
  }

  function measureCurl(m, st) {
    var side = st.picker.update(m, ['shoulder', 'elbow', 'wrist']);
    var s = side;
    var ang = jointAngle(m, s, 'shoulder', 'elbow', 'wrist');
    if (ang === null) return { valid: false, hint: 'Keep your arm visible, side-on to the camera.' };
    var faults = {};
    var sh = pt(m, s + '_shoulder'), el = pt(m, s + '_elbow');
    var arm = dist(sh, el);
    if (arm > 1 && Math.abs(el.x - sh.x) / arm > 0.45) faults.elbow = true;
    var lean = torsoLean(m, s);
    if (lean !== null) {
      if (st.active && st.leanBase !== null && Math.abs(lean - st.leanBase) > 12) faults.swing = true;
      if (!st.active) st.leanBase = st.leanBase === null ? lean : st.leanBase + 0.05 * (lean - st.leanBase);
    }
    return { valid: true, p: clamp((158 - ang) / 110, 0, 1), angle: ang, faults: faults };
  }

  function legScore(m, side) {
    var h = m[side + '_hip'], k = m[side + '_knee'], a = m[side + '_ankle'];
    return (h && k && a) ? (h.score + k.score + a.score) / 3 : 0;
  }

  function measureLunge(m, st) {
    var side = st.picker.update(m, ['hip', 'knee', 'ankle']);
    var a1 = jointAngle(m, side, 'hip', 'knee', 'ankle', 0.4);
    var a2 = legScore(m, other(side)) >= 0.6 ? jointAngle(m, other(side), 'hip', 'knee', 'ankle', 0.5) : null;
    var ang = a1 === null ? null : (a2 === null ? a1 : Math.min(a1, a2)); // both legs only if both clearly seen
    if (ang === null) return { valid: false };
    var faults = {};
    var p = clamp((165 - ang) / 75, 0, 1);
    var lean = torsoLean(m, side);
    if (lean !== null && lean > 30 && p > 0.3) faults.lean = true;
    return { valid: true, p: p, angle: ang, faults: faults };
  }

  var SPECS = {
    squat: {
      cfg: cfg({}), measure: measureSquat,
      waiting: 'Stand tall, whole body in frame, to begin.', ready: 'Ready — start your squat.',
      lower: 'Go lower…', depth: 'Good depth — drive up!', up: 'Drive up through your feet.',
      shallowMsg: 'Too shallow — go lower. Rep not counted.',
      hint: 'Step back so your whole body is visible.',
      tips: { lean: 'Keep your chest up — don’t fold forward.', cave: 'Push your knees out — they’re caving in.' }
    },
    pushup: {
      cfg: cfg({}), measure: measurePushup,
      waiting: 'Get into push-up position and lock your arms out.', ready: 'Ready — lower with control.',
      lower: 'Lower further…', depth: 'Good depth — press up!', up: 'Press back up.',
      shallowMsg: 'Too shallow — chest lower. Rep not counted.',
      hint: 'Get side-on to the camera, in push-up position, whole body in frame.',
      tips: { sag: 'Hips are sagging — squeeze your core and glutes.', pike: 'Hips are piking up — lower them in line with your shoulders.' }
    },
    curl: {
      cfg: cfg({}), measure: measureCurl,
      waiting: 'Stand with your arm hanging straight to begin.', ready: 'Ready — curl up.',
      lower: 'Curl higher…', depth: 'Squeeze at the top — now lower slowly.', up: 'Lower with control.',
      shallowMsg: 'Curl higher — full range. Rep not counted.',
      hint: 'Keep your arm visible, side-on to the camera.',
      tips: { elbow: 'Keep your elbow pinned to your side.', swing: 'Stop swinging your body — let the arm do the work.' }
    },
    lunge: {
      cfg: cfg({ depth: 0.65, shallow: 0.45 }), measure: measureLunge,
      waiting: 'Stand tall, side-on to the camera, to begin.', ready: 'Ready — step into your lunge.',
      lower: 'Sink lower…', depth: 'Good depth — drive back up!', up: 'Drive back up.',
      shallowMsg: 'Too shallow — sink lower. Rep not counted.',
      hint: 'Step back so your legs are visible, side-on to the camera.',
      tips: { lean: 'Keep your torso upright — don’t lean over the front knee.' }
    }
  };

  // ---------- rep-based tracker ----------
  function createRepTracker(key) {
    var spec = SPECS[key];
    var st = {
      picker: new SidePicker(), pS: new Smoother(35), aS: new Smoother(90),
      counter: new RepCounter(spec.cfg), angle: null, p: 0, prevMax: 0,
      invalidSince: null, fE: {}, lastT: null, lastEvent: null, lastEventT: -1e9, active: false, leanBase: null
    };

    return {
      key: key, label: 'Reps',
      update: function (keypoints, t) {
        var m = keypoints ? indexKeypoints(keypoints) : null;
        st.active = st.counter.state === 'active';
        var meas = m ? spec.measure(m, st) : { valid: false };
        var active = [], name;

        if (meas.valid) {
          st.invalidSince = null;
          st.p = st.pS.update(meas.p, t);
          if (meas.angle !== null && meas.angle !== undefined) st.angle = st.aS.update(meas.angle, t);
          var dtf = st.lastT === null ? 33 : Math.max(1, t - st.lastT);
          var af = 1 - Math.exp(-dtf / 100);
          for (name in spec.tips) { // a fault is "on" once ~70% of the last ~0.15 s of frames agree
            var prev = st.fE[name] || 0;
            st.fE[name] = prev + af * ((meas.faults[name] ? 1 : 0) - prev);
            if (st.fE[name] >= 0.7) active.push(name);
          }
          var raw = []; for (name in meas.faults) raw.push(name);
          var ev = st.counter.update(st.p, t, raw);
          if (ev) {
            if (ev.type === 'rep') {
              st.lastEvent = ev.clean
                ? { cls: 'good', text: 'Rep ' + st.counter.reps + ' — clean ✓' }
                : { cls: 'warn', text: 'Rep ' + st.counter.reps + ' counted — ' + spec.tips[ev.faults[0]] };
            } else {
              st.lastEvent = { cls: 'warn', text: spec.shallowMsg };
            }
            st.lastEventT = t;
          }
        } else {
          st.fE = {};
          if (st.invalidSince === null) st.invalidSince = t;
          if (t - st.invalidSince > LOST_MS) { // body lost: forget the rep in progress, re-arm from the top
            st.counter.reset(); st.pS.reset(); st.aS.reset(); st.picker = new SidePicker();
            st.leanBase = null; st.lastEvent = null;
          }
        }

        st.lastT = t;
        var cls, text, c = st.counter;
        if (!meas.valid) { cls = 'warn'; text = meas.hint || spec.hint; }
        else if (st.lastEvent && t - st.lastEventT < 1300) { cls = st.lastEvent.cls; text = st.lastEvent.text; }
        else if (c.state === 'waiting') { cls = 'warn'; text = spec.waiting; }
        else if (c.state === 'ready') { cls = 'good'; text = spec.ready; }
        else if (active.length) { cls = 'bad'; text = spec.tips[active[0]]; }
        else if (c.reachedDepth) { cls = 'good'; text = st.p < c.maxP - 0.12 ? spec.up : spec.depth; }
        else { cls = 'warn'; text = spec.lower; }

        return {
          angle: st.angle, statusClass: cls, message: text,
          reps: c.reps, cleanReps: c.clean, state: c.state, progress: st.p, valid: meas.valid
        };
      }
    };
  }

  // ---------- plank tracker (counts seconds of good form) ----------
  function createPlankTracker() {
    var st = { picker: new SidePicker(), sS: new Smoother(120), holdMs: 0, badMs: 0, lastT: null, angle: null, aS: new Smoother(120), invalidSince: null };
    return {
      key: 'plank', label: 'Seconds held',
      update: function (keypoints, t) {
        var dt = st.lastT === null ? 0 : Math.min(200, Math.max(0, t - st.lastT));
        st.lastT = t;
        var m = keypoints ? indexKeypoints(keypoints) : null, body = null, side = 'right';
        if (m) { side = st.picker.update(m, ['shoulder', 'hip', 'ankle']); body = bodyLine(m, side); }
        var cls, text, sag = null;
        if (!body || body.sag === null) {
          if (st.invalidSince === null) st.invalidSince = t;
          cls = 'warn'; text = 'Get side-on to the camera with your whole body in frame.';
        } else {
          st.invalidSince = null;
          sag = st.sS.update(body.sag, t);
          var ang = m && jointAngle(m, side, 'shoulder', 'hip', 'ankle');
          if (ang !== null && ang !== false) st.angle = st.aS.update(ang, t);
          if (Math.abs(sag) <= 0.08) { st.badMs = 0; st.holdMs += dt; cls = 'good'; text = 'Solid line — hold steady and breathe.'; }
          else {
            st.badMs += dt; if (st.badMs <= 800) st.holdMs += dt; // brief wobble doesn't stop the clock
            cls = 'bad';
            text = sag > 0 ? 'Hips are sagging — engage your core and lift slightly.' : 'Hips are piking up — lower them in line with your shoulders.';
          }
        }
        return { angle: st.angle, statusClass: cls, message: text, reps: Math.floor(st.holdMs / 1000), cleanReps: null, state: 'hold', progress: 0, valid: !!body };
      }
    };
  }

  function createTracker(key) {
    return key === 'plank' ? createPlankTracker() : createRepTracker(key);
  }

  return {
    createTracker: createTracker,
    EXERCISE_KEYS: ['squat', 'pushup', 'plank', 'curl', 'lunge'],
    _internals: { angleAt: angleAt, Smoother: Smoother, RepCounter: RepCounter, SPECS: SPECS }
  };
});
