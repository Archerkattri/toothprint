(() => {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const slider = document.getElementById("measured");
  if (slider) {
    const stableThreshold = 0.35;
    const changeThreshold = 0.75;
    const range = 2.0;
    const band = document.getElementById("band");
    const pin = document.getElementById("pin");
    const verdict = document.getElementById("verdict");
    const word = document.getElementById("verdictWord");
    const measuredOut = document.getElementById("measuredOut");
    const intervalOut = document.getElementById("intervalOut");
    const sliderVal = document.getElementById("sliderVal");
    const noiseSel = document.getElementById("noise");
    const pct = (mm) => Math.max(0, Math.min(100, (mm / range) * 100));

    function update() {
      const measured = slider.value / 100;
      const radius = Number.parseFloat(noiseSel.value);
      const lo = Math.max(0, measured - radius);
      const hi = measured + radius;
      let state = "uncertain";
      if (lo >= changeThreshold) state = "changed";
      else if (hi <= stableThreshold) state = "stable";
      band.style.left = `${pct(lo)}%`;
      band.style.width = `${pct(hi) - pct(lo)}%`;
      pin.style.left = `${pct(measured)}%`;
      verdict.dataset.state = state;
      word.textContent = state;
      measuredOut.textContent = `${measured.toFixed(2)} mm`;
      intervalOut.textContent = `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
      sliderVal.textContent = measured.toFixed(2);
    }
    slider.addEventListener("input", update);
    noiseSel.addEventListener("change", update);
    update();
  }

  const reveals = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in"));
  } else {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.16 });
    reveals.forEach((el) => observer.observe(el));
  }
})();
