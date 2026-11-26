# IRONFIST

A static website for a gym + yoga studio. Plain HTML/CSS/JS — no build step,
no server, nothing to install. Works as-is on **GitHub Pages**.

## What's on the site

| Page | What it does |
|---|---|
| `index.html` | Home page, links to every feature |
| `gym.html` | Weekly split + 41-exercise library (push / pull / legs / core) with demo-video links |
| `yoga.html` | 4 class styles + sample flow + 34-pose library with icons and demo-video links |
| `calculator.html` | Calorie target (Mifflin–St Jeor) + a daily food log that is saved in the browser |
| `timer.html` | **Live interval workout timer** — warm-up, work/rest rounds, cooldown, beeps, presets (Tabata, HIIT, EMOM) |
| `form-check.html` | **Live camera form checker** — squat, push-up, plank, bicep curl, lunge (rep counting + right/wrong feedback) |

## Files (everything must sit together at the root of the repo)

```
index.html  gym.html  yoga.html  calculator.html  timer.html  form-check.html
style.css   script.js calculator.js timer.js form-check.js
vendor/tf.min.js   vendor/pose-detection.min.js     <- needed by form-check.html
README.md
```

> The `vendor` folder is required for the camera page. Upload it too.

## Deploy on GitHub Pages

1. Create a repository (e.g. `ironfist`) and upload **all** files and the `vendor` folder to the root
   (Add file → Upload files → drag everything in → Commit changes).
   If you already have an older version, upload over it — same file names replace the old ones.
2. Settings → Pages → Source: **Deploy from a branch** → Branch: `main` → folder `/ (root)` → Save.
3. Wait ~1 minute, then open `https://YOUR-USERNAME.github.io/ironfist/`.
4. Hard-refresh (Ctrl+Shift+R, or Cmd+Shift+R on Mac) so the browser doesn't show a cached old copy.

## The camera form checker — requirements

- Open it from the **https:// GitHub Pages address** (or `http://localhost`). Browsers block the camera on
  pages opened by double-clicking a file (`file://`) — the page will tell you if that's the case.
- Click **Allow** when the browser asks for camera permission.
- The first run downloads a small pose model (a few MB) from Google, so it needs internet. If your network
  blocks it, the page shows a clear message instead of failing silently.
- Stand back so your whole body is in frame. For squats, lunges, curls, push-ups and planks stand **side-on**
  to the camera.
- All processing happens in the browser. No video is uploaded or stored.

## Running locally (optional)

```
python3 -m http.server 8000
```
then visit `http://localhost:8000`.
