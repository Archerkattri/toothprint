/* ToothPrint frontend: the live tooth-print arch, the conformal demo, reveals. */
(() => {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------------- */
  /* Signature: a dental arch drawn as a luminous "tooth print".            */
  /* ---------------------------------------------------------------------- */
  const canvas = document.getElementById("archCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = canvas.width, H = canvas.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const cx = W / 2, cy = H * 0.46, N = 15;
    // Dental arch as a horseshoe (occlusal view): teeth on a U with the opening
    // downward. Each tooth sits on the curve, oriented along its outward normal.
    const teeth = [];
    const rx = W * 0.33, ry = H * 0.34;
    for (let i = 0; i < N; i++) {
      const f = i / (N - 1);                       // 0 .. 1 around the arch
      const ang = Math.PI * (0.07 + 0.86 * f);     // ~12deg .. ~168deg
      const x = cx - Math.cos(ang) * rx;
      const y = cy - Math.sin(ang) * ry;
      const size = 22 + Math.sin(ang) * 12;        // anterior (top) slimmer
      teeth.push({ x, y, a: ang - Math.PI / 2, size });
    }
    // Feature points (FPFH-like) sprinkled around the arch.
    const feats = [];
    for (let i = 0; i < 150; i++) {
      const t = teeth[Math.floor(Math.random() * N)];
      const r = Math.random() * t.size * 1.1;
      const th = Math.random() * Math.PI * 2;
      feats.push({ x: t.x + Math.cos(th) * r, y: t.y + Math.sin(th) * r, ph: Math.random() * Math.PI * 2 });
    }

    function tooth(t, glow) {
      ctx.save();
      ctx.translate(t.x, t.y); ctx.rotate(t.a);
      const w = t.size * 0.82, h = t.size * 1.05;
      ctx.beginPath();
      // a rounded crown silhouette
      ctx.moveTo(-w, -h * 0.2);
      ctx.bezierCurveTo(-w, -h, w, -h, w, -h * 0.2);
      ctx.bezierCurveTo(w * 1.05, h * 0.85, -w * 1.05, h * 0.85, -w, -h * 0.2);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, -h, 0, h);
      grad.addColorStop(0, `rgba(236,230,215,${0.10 * glow})`);
      grad.addColorStop(1, `rgba(191,224,236,${0.03 * glow})`);
      ctx.fillStyle = grad; ctx.fill();
      ctx.strokeStyle = `rgba(236,230,215,${0.55 * glow})`;
      ctx.lineWidth = 1.3;
      ctx.shadowColor = "rgba(191,224,236,0.7)"; ctx.shadowBlur = 14 * glow;
      ctx.stroke();
      ctx.restore();
    }

    let start = null;
    function frame(ts) {
      if (!start) start = ts;
      const elapsed = (ts - start) / 1000;
      ctx.clearRect(0, 0, W, H);

      // arch spine
      ctx.beginPath();
      teeth.forEach((t, i) => i ? ctx.lineTo(t.x, t.y) : ctx.moveTo(t.x, t.y));
      ctx.strokeStyle = "rgba(95,126,144,0.5)"; ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 4]); ctx.stroke(); ctx.setLineDash([]);

      // teeth draw-in (fast stagger; the full arch is present within ~0.6s)
      const grown = reduceMotion ? N : Math.min(N, elapsed / 0.035);
      for (let i = 0; i < N; i++) {
        const g = Math.max(0, Math.min(1, grown - i));
        if (g <= 0) continue;
        tooth(teeth[i], g);
      }

      // feature points twinkle once the arch is in
      const reveal = reduceMotion ? 1 : Math.max(0, Math.min(1, (elapsed - 0.6) / 0.9));
      ctx.shadowBlur = 0;
      feats.forEach((f) => {
        const tw = 0.45 + 0.55 * Math.abs(Math.sin(elapsed * 1.6 + f.ph));
        ctx.beginPath();
        ctx.arc(f.x, f.y, 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(230,169,63,${0.5 * reveal * tw})`;
        ctx.fill();
      });

      if (!reduceMotion) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    if (reduceMotion) requestAnimationFrame(frame);
  }

  /* ---------------------------------------------------------------------- */
  /* Conformal certainty demo.                                              */
  /* ---------------------------------------------------------------------- */
  const slider = document.getElementById("measured");
  if (slider) {
    const STABLE_T = 0.35, CHANGE_T = 0.75, RANGE = 2.0; // mm
    const band = document.getElementById("band");
    const pin = document.getElementById("pin");
    const verdict = document.getElementById("verdict");
    const word = document.getElementById("verdictWord");
    const measuredOut = document.getElementById("measuredOut");
    const intervalOut = document.getElementById("intervalOut");
    const sliderVal = document.getElementById("sliderVal");
    const noiseSel = document.getElementById("noise");

    const pct = (mm) => Math.max(0, Math.min(100, (mm / RANGE) * 100));

    function update() {
      const measured = slider.value / 100;          // 0..2 mm
      const radius = parseFloat(noiseSel.value);     // conformal radius (mm)
      const lo = Math.max(0, measured - radius);
      const hi = measured + radius;

      let state = "uncertain";
      if (lo >= CHANGE_T) state = "changed";
      else if (hi <= STABLE_T) state = "stable";

      band.style.left = pct(lo) + "%";
      band.style.width = (pct(hi) - pct(lo)) + "%";
      pin.style.left = pct(measured) + "%";

      verdict.dataset.state = state;
      word.textContent = state;
      measuredOut.textContent = measured.toFixed(2) + " mm";
      intervalOut.textContent = `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
      sliderVal.textContent = measured.toFixed(2);
    }
    slider.addEventListener("input", update);
    noiseSel.addEventListener("change", update);
    update();
  }

  /* ---------------------------------------------------------------------- */
  /* Scroll reveal.                                                         */
  /* ---------------------------------------------------------------------- */
  const reveals = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.16 });
    reveals.forEach((el) => io.observe(el));
  }
})();
