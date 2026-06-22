/* ToothPrint site: the interactive conformal-certificate demo + a light scroll reveal. */
(() => {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Conformal certainty: drag the measured change, watch the interval decide. */
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
      const lo = Math.max(0, measured - radius), hi = measured + radius;
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

  /* Light, single-pass reveal — communicates arrival, not decoration. */
  const reveals = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.16 });
    reveals.forEach((el) => io.observe(el));
  }
})();
