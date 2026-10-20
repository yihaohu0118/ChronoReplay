/* Respect motion preferences, visibility and the visitor's pause choice. */
(() => {
  const video = document.querySelector('.brand-player video');
  const button = document.querySelector('.cover-toggle');
  if (!video || !button) return;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  let wanted = !reduce.matches, visible = true, failed = false;
  button.classList.add('ready');
  function label() {
    button.textContent = video.paused ? '▶ Play cover' : 'Ⅱ Pause cover';
    button.setAttribute('aria-pressed', String(!video.paused));
  }
  function sync() {
    if (wanted && visible && !document.hidden && !failed) {
      video.play().then(label).catch(() => { wanted = false; label(); });
    } else { video.pause(); label(); }
  }
  button.addEventListener('click', () => {
    wanted = video.paused;
    sync();
  });
  video.addEventListener('play', label);
  video.addEventListener('pause', label);
  video.addEventListener('error', () => {
    failed = true; wanted = false; button.disabled = true;
    button.textContent = 'Static cover';
    // Keep the original poster visible if video decoding is unavailable.
    const fallback = document.createElement('img');
    fallback.src = video.poster;
    fallback.alt = video.getAttribute('aria-label');
    fallback.width = 1600; fallback.height = 820;
    video.replaceWith(fallback);
  });
  reduce.addEventListener('change', () => { wanted = !reduce.matches; sync(); });
  document.addEventListener('visibilitychange', sync);
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting; sync();
  }, {threshold:.05}).observe(video);
  label();
})();
