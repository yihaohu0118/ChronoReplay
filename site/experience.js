'use strict';

(() => {
  const board = document.querySelector('.demo-board');
  if (!board) return;
  const isChrono = document.body.dataset.project === 'chrono';
  const picture = board.querySelector('.demo-image');
  const strip = board.querySelector('.filmstrip');
  const label = board.querySelector('.frame-label');
  const corner = board.querySelector('.frame-corner');
  const question = board.querySelector('.question-text');
  const answer = board.querySelector('.answer-value');
  const caption = board.querySelector('.answer-caption');
  const play = board.querySelector('.play-trace');
  const modeButtons = [...board.querySelectorAll('[data-mode]')];
  const rounds = [1, 1, 2, 2, 3, 4, 5, 5];
  const chronoFrames = rounds.map((round, i) => ({
    src: `assets/replay/frame-${i + 1}.jpg`,
    label: `Round ${round}`,
    alt: `Observation ${i + 1} from round ${round} of the spatial episode illustrated in the ChronoReplay paper.`
  }));
  const vissFrames = times => times.map(time => ({
    src: `assets/replay/bean_${time.padStart(6, '0')}s.jpg`,
    label: `${time} s`,
    alt: `Video frame at ${time} seconds in the ViSS paper's causal-view case study.`
  }));
  const modes = isChrono ? [
    {
      name: 'temporal', frames: chronoFrames.slice(0, 6), selected: 2,
      question: 'When were both the bathtub and folded towel on the observer’s left?',
      answer: 'Round 2', caption: 'Temporal answer · asked in Round 4',
      corner: 'Shared observation history'
    },
    {
      name: 'reference', frames: chronoFrames, selected: 7,
      question: 'Using the Round-2 facing as forward, did the observer move left or right?',
      answer: 'Left', caption: 'Reference-frame answer · asked in Round 5',
      corner: 'Origin: Round 2'
    }
  ] : [
    {
      name: 'recent', frames: vissFrames(['141.00', '142.32', '143.68', '145.00']), selected: 3,
      question: 'Why is Mr. Bean under the bed now?',
      answer: 'D · Investigate something strange',
      caption: 'Recent proposal · margin 0.834',
      corner: '4 recent frames'
    },
    {
      name: 'event', frames: vissFrames(['117.40', '119.68', '122.00', '145.00']), selected: 1,
      question: 'Why is Mr. Bean under the bed now?',
      answer: 'A · Retrieve the teddy bear',
      caption: 'Event proposal · margin 0.991',
      corner: '4 shown / 20 input frames'
    },
    {
      name: 'history', frames: vissFrames(['23.00', '119.68', '122.00', '145.00']), selected: 1,
      question: 'Why is Mr. Bean under the bed now?',
      answer: 'A · Retrieve the teddy bear',
      caption: 'History agrees · margin 0.986',
      corner: '4 shown / 32 input frames'
    }
  ];
  let currentMode = 0;
  let currentFrame = modes[0].selected;
  let interval = null;

  function displayFrame(index) {
    currentFrame = index;
    const data = modes[currentMode].frames[index];
    picture.src = data.src;
    picture.alt = data.alt;
    label.textContent = data.label;
    picture.classList.remove('scene-refresh');
    requestAnimationFrame(() => picture.classList.add('scene-refresh'));
    strip.querySelectorAll('button').forEach((button, i) => {
      button.setAttribute('aria-pressed', String(index === i));
    });
  }
  function displayMode(index) {
    currentMode = index;
    const data = modes[index];
    modeButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(index === i)));
    question.textContent = data.question;
    answer.textContent = data.answer;
    caption.textContent = data.caption;
    corner.textContent = data.corner;
    strip.replaceChildren();
    data.frames.forEach((frame, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', `Show ${frame.label}, frame ${i + 1}`);
      const img = document.createElement('img');
      img.src = frame.src;
      img.alt = '';
      const text = document.createElement('span');
      text.textContent = frame.label;
      button.append(img, text);
      button.addEventListener('click', () => { stop(); displayFrame(i); });
      strip.append(button);
    });
    displayFrame(data.selected);
  }
  function stop() {
    if (interval !== null) clearInterval(interval);
    interval = null;
    play.textContent = '▶ Play trace';
    play.setAttribute('aria-pressed', 'false');
  }
  modeButtons.forEach((button, i) => button.addEventListener('click', () => { stop(); displayMode(i); }));
  play.addEventListener('click', () => {
    if (interval !== null) { stop(); return; }
    play.textContent = 'Ⅱ Pause';
    play.setAttribute('aria-pressed', 'true');
    if (isChrono) displayFrame(0); else displayMode(0);
    interval = setInterval(() => {
      if (isChrono) {
        if (currentFrame >= modes[currentMode].frames.length - 1) { stop(); return; }
        displayFrame(currentFrame + 1);
      } else {
        if (currentMode >= modes.length - 1) { stop(); return; }
        displayMode(currentMode + 1);
      }
    }, isChrono ? 1100 : 2400);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  displayMode(0);

  const progress = document.querySelector('.reading-progress');
  const nav = [...document.querySelectorAll('.nav-links a')];
  const inHero = () => scrollY < document.querySelector('#overview').offsetTop - innerHeight * .4;
  function updateProgress() {
    const range = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = (range > 0 ? scrollY / range * 100 : 0) + '%';
    if (inHero()) nav.forEach(link => {
      link.classList.remove('active');
      link.removeAttribute('aria-current');
    });
  }
  addEventListener('scroll', updateProgress, {passive: true});
  addEventListener('resize', updateProgress);
  updateProgress();
  const observer = new IntersectionObserver(entries => {
    if (inHero()) return;
    for (const entry of entries) if (entry.isIntersecting) {
      nav.forEach(link => {
        const active = link.getAttribute('href') === '#' + entry.target.id;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
  }, {rootMargin: '-15% 0px -65% 0px'});
  nav.forEach(link => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) observer.observe(target);
  });
})();
