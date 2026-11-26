// IRONFIST — Form Check
// Live camera-based posture/form tracking using TensorFlow.js + MoveNet.
// Everything runs client-side in the browser: no video or pose data is ever
// sent to a server. Requires a real server (HTTPS or localhost), not a
// double-clicked file, and camera permission.

(function () {
  'use strict';

  // ---------- DOM refs ----------
  var video = document.getElementById('fc-video');
  var canvas = document.getElementById('fc-canvas');
  var placeholder = document.getElementById('fc-placeholder');
  var startBtn = document.getElementById('fc-start');
  var stopBtn = document.getElementById('fc-stop');
  var statusLine = document.getElementById('fc-status');
  var banner = document.getElementById('fc-banner');
  var exerciseButtons = document.querySelectorAll('.fc-ex-btn');
  var messageEl = document.getElementById('fc-message');
  var repsEl = document.getElementById('fc-reps');
  var repsLabelEl = document.getElementById('fc-reps-label');
  var angleEl = document.getElementById('fc-angle');
  var cuesEl = document.getElementById('fc-cues');

  if (!video || !canvas) return; // page not present, nothing to wire up

  var ctx = canvas.getContext('2d');

  function showBanner(html) {
    banner.innerHTML = html;
    banner.classList.add('show');
  }

  // ---------- upfront environment checks ----------
  // These run immediately so a broken setup is obvious instead of silently
  // doing nothing when "Start camera" is pressed.
  var blocked = false;

  if (location.protocol === 'file:') {
    blocked = true;
    showBanner(
      '<b>This page was opened directly from a file.</b> Camera access only works when a page is served from a real address — ' +
      'your live GitHub Pages link, or a local server on <code>localhost</code>. Double-clicking the HTML file will not work. ' +
      'Push this file to your GitHub repo and open it from your <code>https://&hellip;github.io/&hellip;</code> URL instead.'
    );
  } else if (!window.isSecureContext) {
    blocked = true;
    showBanner(
      '<b>This page isn’t running in a secure context.</b> Camera access requires HTTPS (GitHub Pages provides this automatically) ' +
      'or <code>localhost</code>. If you’re testing on a plain <code>http://</code> address other than localhost, switch to HTTPS.'
    );
  } else if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    blocked = true;
    showBanner('<b>This browser doesn’t support camera access.</b> Try a recent version of Chrome, Edge, Safari, or Firefox.');
  }

  if (blocked) {
    startBtn.disabled = true;
    statusLine.textContent = 'Camera unavailable — see the notice above.';
  } else {
    statusLine.textContent = 'Ready. Press "Start camera" to begin.';
  }

  // ---------- state ----------
  var detector = null;
  var modelReady = false;
  var running = false;
  var stream = null;
  var rafId = null;
  var currentExercise = 'squat';
  var repState = 'up'; // generic state machine: 'up' / 'down'
  var repCount = 0;
  var plankHoldStart = null;
  var plankHoldSeconds = 0;

  // ---------- exercise definitions ----------
  var EXERCISES = {
    squat: {
      label: 'Squat', repsLabel: 'Reps',
      cues: [
        'Feet roughly shoulder-width apart, toes slightly out.',
        'Send the hips back and down, chest up.',
        'Knees track over the toes — avoid them caving inward.',
        'Aim for hip crease at or below knee height.'
      ],
      downAngle: 100, upAngle: 160
    },
    pushup: {
      label: 'Push-Up', repsLabel: 'Reps',
      cues: [
        'Hands under or slightly wider than shoulders.',
        'Keep a straight line from shoulders to ankles.',
        'Lower until elbows reach about 90°.',
        'Avoid letting the hips sag or pike up.'
      ],
      downAngle: 95, upAngle: 155
    },
    plank: {
      label: 'Plank', repsLabel: 'Seconds held',
      cues: [
        'Elbows under shoulders, forearms flat on the floor.',
        'Keep a straight line from shoulders to ankles.',
        'Brace the core — don’t let the hips drop or pike.',
        'Breathe steadily through the hold.'
      ]
    },
    curl: {
      label: 'Bicep Curl', repsLabel: 'Reps',
      cues: [
        'Keep the elbow pinned close to your torso.',
        'Curl through the full range, then lower with control.',
        'Avoid swinging the shoulder or torso to move the weight.',
        'Squeeze at the top, don’t just drop back down.'
      ],
      downAngle: 150, upAngle: 55
    },
    lunge: {
      label: 'Lunge', repsLabel: 'Reps',
      cues: [
        'Step forward (or back) into a long, stable stance.',
        'Front knee bends to roughly 90°, tracking over the ankle.',
        'Keep the torso upright, not leaning forward.',
        'Back knee lowers toward the floor without slamming it.'
      ],
      downAngle: 100, upAngle: 160
    }
  };

  // ---------- geometry helpers ----------
  function angleBetween(a, b, c) {
    var v1x = a.x - b.x, v1y = a.y - b.y;
    var v2x = c.x - b.x, v2y = c.y - b.y;
    var dot = v1x * v2x + v1y * v2y;
    var mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    var mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (mag1 === 0 || mag2 === 0) return null;
    var cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  function kp(keypoints, name) {
    for (var i = 0; i < keypoints.length; i++) {
      if (keypoints[i].name === name) return keypoints[i];
    }
    return null;
  }

  function confident(point, minScore) {
    return point && typeof point.score === 'number' && point.score >= (minScore || 0.35);
  }

  function pickSide(keypoints, jointA, jointB, jointC) {
    var left = [kp(keypoints, 'left_' + jointA), kp(keypoints, 'left_' + jointB), kp(keypoints, 'left_' + jointC)];
    var right = [kp(keypoints, 'right_' + jointA), kp(keypoints, 'right_' + jointB), kp(keypoints, 'right_' + jointC)];
    var leftScore = left.every(Boolean) ? (left[0].score + left[1].score + left[2].score) : -1;
    var rightScore = right.every(Boolean) ? (right[0].score + right[1].score + right[2].score) : -1;
    if (leftScore < 0 && rightScore < 0) return null;
    return rightScore >= leftScore ? right : left;
  }

  // ---------- drawing ----------
  var SKELETON_PAIRS = [
    ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'],
    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
    ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
    ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
  ];

  function drawSkeleton(keypoints, statusClass) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var color = statusClass === 'good' ? '#8C9B7C' : statusClass === 'bad' ? '#B3311A' : '#D8A23B';

    ctx.lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';

    SKELETON_PAIRS.forEach(function (pair) {
      var p1 = kp(keypoints, pair[0]);
      var p2 = kp(keypoints, pair[1]);
      if (confident(p1) && confident(p2)) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    });

    ctx.fillStyle = color;
    keypoints.forEach(function (point) {
      if (confident(point)) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // ---------- posture measurements ----------
  // How far the torso leans away from vertical, in degrees (0 = perfectly upright).
  function torsoLean(keypoints) {
    var l = [kp(keypoints, 'left_shoulder'), kp(keypoints, 'left_hip')];
    var r = [kp(keypoints, 'right_shoulder'), kp(keypoints, 'right_hip')];
    var ls = (l[0] && l[1]) ? l[0].score + l[1].score : -1;
    var rs = (r[0] && r[1]) ? r[0].score + r[1].score : -1;
    if (ls < 0 && rs < 0) return null;
    var side = rs >= ls ? r : l;
    var dx = side[0].x - side[1].x;
    var dy = side[0].y - side[1].y; // negative when the shoulder is above the hip
    var len = Math.sqrt(dx * dx + dy * dy);
    if (!len) return null;
    var c = Math.max(-1, Math.min(1, -dy / len));
    return (Math.acos(c) * 180) / Math.PI;
  }

  // Where the hips sit relative to the straight line from shoulder to ankle, as a fraction of that
  // line's length. Positive = hips below the line (sagging), negative = above it (piking).
  // Returns null unless the body is roughly horizontal and seen side-on.
  function hipSag(keypoints) {
    var s = pickSide(keypoints, 'shoulder', 'hip', 'ankle');
    if (!s) return null;
    var dxl = s[2].x - s[0].x;
    var dyl = s[2].y - s[0].y;
    var len = Math.sqrt(dxl * dxl + dyl * dyl);
    if (len < 1 || Math.abs(dxl) < 0.5 * len) return null;
    var lineY = s[0].y + dyl * (s[1].x - s[0].x) / dxl;
    return (s[1].y - lineY) / len;
  }

  // ---------- per-exercise evaluation ----------
  function evaluateSquat(keypoints) {
    var side = pickSide(keypoints, 'hip', 'knee', 'ankle');
    if (!side) return noBodyMessage();

    var kneeAngle = angleBetween(side[0], side[1], side[2]);
    var lean = torsoLean(keypoints);

    var cfg = EXERCISES.squat;
    updateRepState(kneeAngle, cfg.downAngle, cfg.upAngle);

    var statusClass = 'warn';
    var message = 'Stand so your full body is visible.';

    if (kneeAngle !== null) {
      if (kneeAngle > cfg.upAngle) { statusClass = 'good'; message = 'Standing — ready to squat.'; }
      else if (kneeAngle <= cfg.downAngle) { statusClass = 'good'; message = 'Good depth — drive back up.'; }
      else { statusClass = 'warn'; message = 'Go a little lower for full depth.'; }
      // A forward lean past ~70° from vertical while squatting means the chest is folding over the thighs.
      if (lean !== null && lean > 70 && kneeAngle < 150) {
        statusClass = 'bad'; message = 'Keep your chest up — don’t fold forward.';
      }
    }
    return { angle: kneeAngle, statusClass: statusClass, message: message };
  }

  function evaluatePushup(keypoints) {
    var arm = pickSide(keypoints, 'shoulder', 'elbow', 'wrist');
    if (!arm) return noBodyMessage();

    var elbowAngle = angleBetween(arm[0], arm[1], arm[2]);
    var sag = hipSag(keypoints);
    var cfg = EXERCISES.pushup;
    updateRepState(elbowAngle, cfg.downAngle, cfg.upAngle);

    var statusClass = 'warn';
    var message = 'Get side-on to the camera with your whole body in frame.';

    if (elbowAngle !== null) {
      if (elbowAngle > cfg.upAngle) { statusClass = 'good'; message = 'Top of the rep — lower with control.'; }
      else if (elbowAngle <= cfg.downAngle) { statusClass = 'good'; message = 'Good depth — press back up.'; }
      else { statusClass = 'warn'; message = 'Lower a little further.'; }
    }
    if (sag !== null && sag > 0.10) {
      statusClass = 'bad'; message = 'Hips are sagging — squeeze your core and glutes.';
    } else if (sag !== null && sag < -0.10) {
      statusClass = 'bad'; message = 'Hips are piking up — lower them in line with your shoulders.';
    }
    return { angle: elbowAngle, statusClass: statusClass, message: message };
  }

  function evaluatePlank(keypoints) {
    var line = pickSide(keypoints, 'shoulder', 'hip', 'ankle');
    if (!line) return noBodyMessage();

    var bodyLine = angleBetween(line[0], line[1], line[2]);
    var sag = hipSag(keypoints);
    var statusClass, message;

    if (sag === null) { statusClass = 'warn'; message = 'Get side-on to the camera with your whole body in frame.'; }
    else if (sag > 0.08) { statusClass = 'bad'; message = 'Hips are sagging — engage your core and lift slightly.'; }
    else if (sag < -0.08) { statusClass = 'bad'; message = 'Hips are piking up — lower them in line with your shoulders.'; }
    else { statusClass = 'good'; message = 'Solid line — hold steady and breathe.'; }

    if (statusClass === 'good') {
      if (plankHoldStart === null) plankHoldStart = Date.now();
      plankHoldSeconds = Math.floor((Date.now() - plankHoldStart) / 1000);
    } else {
      plankHoldStart = null;
    }
    repCount = plankHoldSeconds;

    return { angle: bodyLine, statusClass: statusClass, message: message };
  }

  function evaluateCurl(keypoints) {
    var arm = pickSide(keypoints, 'shoulder', 'elbow', 'wrist');
    if (!arm) return noBodyMessage();

    var elbowAngle = angleBetween(arm[0], arm[1], arm[2]);
    var cfg = EXERCISES.curl;
    updateRepState(elbowAngle, cfg.upAngle, cfg.downAngle);

    // How far the elbow has swung forward/back of the shoulder, relative to upper-arm length.
    var upperArm = Math.sqrt(Math.pow(arm[1].x - arm[0].x, 2) + Math.pow(arm[1].y - arm[0].y, 2));
    var elbowDrift = upperArm > 0 ? Math.abs(arm[1].x - arm[0].x) / upperArm : 0;
    var statusClass = 'warn';
    var message = 'Keep your arm visible from the side.';

    if (elbowAngle !== null) {
      if (elbowAngle <= cfg.upAngle) { statusClass = 'good'; message = 'Fully curled — lower with control.'; }
      else if (elbowAngle >= cfg.downAngle) { statusClass = 'good'; message = 'Fully extended — curl back up.'; }
      else { statusClass = 'warn'; message = 'Keep going through the full range.'; }
    }
    if (elbowDrift > 0.5) {
      statusClass = 'bad'; message = 'Keep your elbow pinned to your side — it’s drifting forward.';
    }
    return { angle: elbowAngle, statusClass: statusClass, message: message };
  }

  function evaluateLunge(keypoints) {
    var leg = pickSide(keypoints, 'hip', 'knee', 'ankle');
    if (!leg) return noBodyMessage();

    var kneeAngle = angleBetween(leg[0], leg[1], leg[2]);
    var lean = torsoLean(keypoints);
    var cfg = EXERCISES.lunge;
    updateRepState(kneeAngle, cfg.downAngle, cfg.upAngle);

    var statusClass = 'warn';
    var message = 'Step into the lunge so your side is visible to the camera.';

    if (kneeAngle !== null) {
      if (kneeAngle > cfg.upAngle) { statusClass = 'good'; message = 'Standing tall — step into the next rep.'; }
      else if (kneeAngle <= cfg.downAngle + 10 && kneeAngle >= cfg.downAngle - 20) { statusClass = 'good'; message = 'Good depth on the front knee.'; }
      else if (kneeAngle < cfg.downAngle - 20) { statusClass = 'bad'; message = 'That’s very deep — ease up slightly to protect the knee.'; }
      else { statusClass = 'warn'; message = 'Bend the front knee a bit more.'; }
    }
    if (lean !== null && lean > 30) {
      statusClass = 'bad'; message = 'Keep your torso upright, not leaning over the front knee.';
    }
    return { angle: kneeAngle, statusClass: statusClass, message: message };
  }

  function noBodyMessage() {
    return { angle: null, statusClass: 'warn', message: 'Step back so your whole body is visible.' };
  }

  function updateRepState(angle, downThreshold, upThreshold) {
    if (angle === null) return;
    if (repState === 'up' && angle <= downThreshold) {
      repState = 'down';
    } else if (repState === 'down' && angle >= upThreshold) {
      repState = 'up';
      repCount += 1;
    }
  }

  var EVALUATORS = {
    squat: evaluateSquat, pushup: evaluatePushup, plank: evaluatePlank,
    curl: evaluateCurl, lunge: evaluateLunge
  };

  // ---------- UI wiring ----------
  function renderCues(exerciseKey) {
    var cfg = EXERCISES[exerciseKey];
    cuesEl.innerHTML = '';
    cfg.cues.forEach(function (cue) {
      var li = document.createElement('li');
      li.textContent = cue;
      cuesEl.appendChild(li);
    });
    repsLabelEl.textContent = cfg.repsLabel;
  }

  function setExercise(key) {
    currentExercise = key;
    repState = 'up';
    repCount = 0;
    plankHoldStart = null;
    plankHoldSeconds = 0;
    repsEl.textContent = '0';
    angleEl.textContent = '—';
    messageEl.className = 'fc-feedback-msg';
    messageEl.textContent = running ? 'Get in position…' : 'Waiting for camera…';
    renderCues(key);
    exerciseButtons.forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-exercise') === key);
    });
  }

  exerciseButtons.forEach(function (btn) {
    btn.addEventListener('click', function () { setExercise(btn.getAttribute('data-exercise')); });
  });

  // ---------- library loading ----------
  // The two tracking libraries live next to this page (tf.min.js, pose-detection.min.js).
  // If those files are missing from the site, fall back to public CDNs so the page can still work.
  var LIB_SOURCES = {
    tf: ['tf.min.js',
      'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
      'https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js'],
    pd: ['pose-detection.min.js',
      'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js',
      'https://unpkg.com/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js']
  };

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var el = document.createElement('script');
      el.src = src;
      el.onload = function () { resolve(); };
      el.onerror = function () { el.remove(); reject(new Error('failed: ' + src)); };
      document.head.appendChild(el);
    });
  }

  function loadFirstWorking(sources, isReady) {
    if (isReady()) return Promise.resolve();
    return sources.reduce(function (chain, src) {
      return chain.catch(function () {
        return loadScript(src).then(function () {
          if (!isReady()) throw new Error('loaded but not ready: ' + src);
        });
      });
    }, Promise.reject(new Error('start')));
  }

  function ensureLibs() {
    return loadFirstWorking(LIB_SOURCES.tf, function () { return !!window.tf; })
      .then(function () {
        return loadFirstWorking(LIB_SOURCES.pd, function () { return !!window.poseDetection; });
      })
      .catch(function () { throw new Error('library-missing'); });
  }

  // ---------- model loading ----------
  function loadModel() {
    if (modelReady) return Promise.resolve();
    if (!window.tf || !window.poseDetection) {
      return Promise.reject(new Error('library-missing'));
    }
    statusLine.textContent = 'Loading pose-detection model (first time can take a few seconds)\u2026';
    // Prefer the GPU (WebGL); fall back to the CPU if this device/browser can't do WebGL.
    return tf.setBackend('webgl').then(function (ok) {
      if (ok === false) return tf.setBackend('cpu');
    }).catch(function () {
      return tf.setBackend('cpu');
    }).then(function () {
      return tf.ready();
    }).then(function () {
      return poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
      });
    }).then(function (d) {
      detector = d;
      modelReady = true;
      statusLine.textContent = 'Model loaded. Camera ready.';
    }).catch(function (err) {
      var e = new Error('model-load');
      e.original = err;
      throw e;
    });
  }

  // ---------- camera control ----------
  function startCamera() {
    if (blocked) return;
    banner.classList.remove('show'); // clear any earlier error notice on retry
    startBtn.disabled = true;
    statusLine.textContent = 'Loading tracking library\u2026';

    ensureLibs()
      .then(function () {
        statusLine.textContent = 'Requesting camera access\u2026';
        return navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      })
      .then(function (s) {
        stream = s;
        video.srcObject = stream;
        return video.play();
      })
      .then(function () { return loadModel(); })
      .then(function () {
        placeholder.style.display = 'none';
        video.style.display = 'block';
        canvas.style.display = 'block';
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        stopBtn.disabled = false;
        running = true;
        messageEl.textContent = 'Get in position…';
        statusLine.textContent = 'Tracking live.';
        detectLoop();
      })
      .catch(function (err) {
        startBtn.disabled = false;
        // Always release the camera if startup failed part-way through.
        if (stream) {
          stream.getTracks().forEach(function (t) { t.stop(); });
          stream = null;
        }
        video.pause();
        if (err && err.name === 'NotAllowedError') {
          statusLine.textContent = 'Camera permission was denied. Allow camera access in your browser\u2019s address-bar icon and try again.';
        } else if (err && err.name === 'NotFoundError') {
          statusLine.textContent = 'No camera was found on this device.';
        } else if (err && err.name === 'NotReadableError') {
          statusLine.textContent = 'The camera is being used by another app or tab. Close it and try again.';
        } else if (err && err.message === 'library-missing') {
          showBanner(
            '<b>The pose-tracking library didn\u2019t load.</b> Upload <code>tf.min.js</code> and <code>pose-detection.min.js</code> to your ' +
            'GitHub repo, in the <b>same folder as form-check.html</b> (not inside another folder), then hard-refresh this page ' +
            '(Ctrl+Shift+R). The page also tries to fetch them from the internet as a backup, so if you see this on a school or work network, ' +
            'that network may be blocking them \u2014 try a different network.'
          );
          statusLine.textContent = 'Library failed to load \u2014 see the notice above.';
        } else if (err && err.message === 'model-load') {
          showBanner(
            '<b>The pose model couldn\u2019t be downloaded.</b> The tracker needs an internet connection the first time to fetch a small ' +
            'model from Google. Some school, work or VPN networks block it, and so do some privacy extensions. Try a different network ' +
            '(e.g. mobile hotspot) or disable blockers for this page, then press Start camera again.'
          );
          statusLine.textContent = 'Model download failed \u2014 see the notice above.';
        } else {
          statusLine.textContent = 'Could not start the camera: ' + (err && err.message ? err.message : err);
        }
      });
  }

  function stopCamera() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    video.pause();
    video.style.display = 'none';
    canvas.style.display = 'none';
    placeholder.style.display = 'block';
    startBtn.disabled = blocked;
    stopBtn.disabled = true;
    statusLine.textContent = 'Camera stopped.';
    messageEl.className = 'fc-feedback-msg';
    messageEl.textContent = 'Waiting for camera…';
  }

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  window.addEventListener('beforeunload', stopCamera);

  // ---------- main detection loop ----------
  function detectLoop() {
    if (!running || !detector) {
      rafId = requestAnimationFrame(detectLoop);
      return;
    }
    detector.estimatePoses(video).then(function (poses) {
      if (!running) return;
      if (poses && poses[0] && poses[0].keypoints) {
        var keypoints = poses[0].keypoints;
        var evaluator = EVALUATORS[currentExercise];
        var result = evaluator(keypoints);

        drawSkeleton(keypoints, result.statusClass);

        messageEl.className = 'fc-feedback-msg ' + result.statusClass;
        messageEl.textContent = result.message;
        angleEl.textContent = result.angle !== null ? Math.round(result.angle) + '°' : '—';
        repsEl.textContent = String(repCount);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        messageEl.className = 'fc-feedback-msg warn';
        messageEl.textContent = 'No person detected — step into frame.';
      }
      rafId = requestAnimationFrame(detectLoop);
    }).catch(function () {
      rafId = requestAnimationFrame(detectLoop);
    });
  }

  renderCues(currentExercise);
})();
