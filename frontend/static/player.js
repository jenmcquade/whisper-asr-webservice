/* Whisper ASR — live job page wiring.
 *
 * Handles:
 *  - Building a sorted `words` array from the transcript once it's swapped in.
 *  - Highlighting the active word on <audio> timeupdate via binary search.
 *  - Clicking any segment (or individual word) seeks + plays from that point.
 *  - Auto-scrolling the active word into view when it leaves the viewport,
 *    but only when the user isn't actively scrolling.
 *  - Pausing playback when the user clicks anywhere outside the transcript.
 *  - Revealing the Download JSON button once the pipeline finishes.
 */

(function () {
  const audio = document.getElementById('audio');
  const resultEl = document.getElementById('result');
  const jsonEl = document.getElementById('result-json-store');

  if (!audio || !resultEl) return;

  /** @type {{start:number,end:number,el:HTMLElement}[]} */
  let words = [];
  let lastIdx = -1;

  // Timestamp of the most recent user-initiated scroll. While it's fresh we
  // skip auto-scrolling the active word so the user can freely browse.
  let lastUserScrollAt = 0;
  const USER_SCROLL_COOLDOWN_MS = 2500;

  function rebuildWordIndex() {
    const nodes = resultEl.querySelectorAll('.word[data-start][data-end]');
    const next = [];
    for (const node of nodes) {
      const start = parseFloat(node.dataset.start);
      const end = parseFloat(node.dataset.end);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      next.push({ start, end, el: node });
    }
    next.sort((a, b) => a.start - b.start);
    words = next;
    lastIdx = -1;
    if (words.length > 0) revealResultButtons();
  }

  const resultObserver = new MutationObserver(() => rebuildWordIndex());
  resultObserver.observe(resultEl, { childList: true, subtree: true });

  rebuildWordIndex();

  /** Binary search for the last word whose start <= t. */
  function findWordIdx(t) {
    let lo = 0, hi = words.length - 1, best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (words[mid].start <= t) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  }

  function setActiveWord(idx) {
    if (idx === lastIdx) return;
    if (lastIdx >= 0 && words[lastIdx]) {
      words[lastIdx].el.classList.remove('active');
    }
    if (idx >= 0 && words[idx]) {
      const el = words[idx].el;
      el.classList.add('active');
      const sinceScroll = Date.now() - lastUserScrollAt;
      if (sinceScroll > USER_SCROLL_COOLDOWN_MS) {
        const rect = el.getBoundingClientRect();
        const vh = window.innerHeight || document.documentElement.clientHeight;
        if (rect.top < 80 || rect.bottom > vh - 80) {
          el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    }
    lastIdx = idx;
  }

  // Mark scroll as user-initiated on direct input, not on programmatic
  // scrollIntoView (which also fires 'scroll' but not wheel/touch/keys).
  const markUserScroll = () => { lastUserScrollAt = Date.now(); };
  window.addEventListener('wheel', markUserScroll, { passive: true });
  window.addEventListener('touchmove', markUserScroll, { passive: true });
  window.addEventListener('keydown', (e) => {
    if (
      e.key === 'ArrowUp' || e.key === 'ArrowDown' ||
      e.key === 'PageUp' || e.key === 'PageDown' ||
      e.key === 'Home' || e.key === 'End' ||
      (e.key === ' ' && e.target === document.body)
    ) {
      markUserScroll();
    }
  });

  audio.addEventListener('timeupdate', () => {
    if (!words.length) {
      rebuildWordIndex();
      if (!words.length) return;
    }
    const idx = findWordIdx(audio.currentTime);
    if (idx >= 0 && words[idx].end < audio.currentTime && idx === words.length - 1) {
      setActiveWord(-1);
    } else {
      setActiveWord(idx);
    }
  });

  audio.addEventListener('seeked', () => {
    if (!words.length) rebuildWordIndex();
    if (!words.length) return;
    setActiveWord(findWordIdx(audio.currentTime));
  });

  // Single delegated handler for seek-on-click: a word wins over its parent
  // segment so clicking mid-sentence lands on the exact word.
  resultEl.addEventListener('click', (e) => {
    const w = e.target.closest('.word[data-start]');
    if (w) {
      const t = parseFloat(w.dataset.start);
      if (Number.isFinite(t)) {
        audio.currentTime = t;
        audio.play().catch(() => {});
      }
      return;
    }
    const seg = e.target.closest('.segment[data-seek]');
    if (seg) {
      const t = parseFloat(seg.dataset.seek);
      if (Number.isFinite(t)) {
        audio.currentTime = t;
        audio.play().catch(() => {});
      }
    }
  });

  resultEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const seg = e.target.closest && e.target.closest('.segment[data-seek]');
    if (!seg) return;
    e.preventDefault();
    const t = parseFloat(seg.dataset.seek);
    if (!Number.isFinite(t)) return;
    audio.currentTime = t;
    audio.play().catch(() => {});
  });

  // Pause playback when the user clicks anywhere outside the transcript or
  // audio controls — gives people an easy "stop" without hunting for the
  // play/pause button.
  document.addEventListener('click', (e) => {
    if (audio.paused) return;
    const t = e.target;
    if (!(t instanceof Element)) return;
    if (
      t.closest('#result') ||
      t.closest('#audio') ||
      t.closest('.player-card') ||
      t.closest('.result-buttons') ||
      t.closest('a') ||
      t.closest('button') ||
      t.closest('input') ||
      t.closest('select') ||
      t.closest('textarea')
    ) {
      return;
    }
    audio.pause();
  });

  // HTMX's afterSwap handles auxiliary signals (Download button reveal,
  // Stop button hide). The MutationObserver above already keeps the word
  // index in sync.
  document.body.addEventListener('htmx:afterSwap', (e) => {
    const t = e.target || (e.detail && e.detail.target);
    if (!t) return;

    if (t.id === 'result-json-store') {
      if (jsonEl && jsonEl.textContent.trim().length > 0) {
        revealResultButtons();
      }
    }

    if (t.classList && t.classList.contains('job-status')) {
      const badge = t.querySelector('.badge');
      const status = badge ? badge.textContent.trim() : '';
      if (status && status !== 'error') revealResultButtons();
      // Any terminal status should hide the Stop button — the pipeline
      // is no longer cancellable.
      if (status) hideStopButton();
    }

    // The `cancelled` SSE frame lands in #cancel-notice; treat it as an
    // immediate signal that the Stop button should disappear even before
    // the `done` event catches up.
    if (t.id === 'cancel-notice' && t.textContent.trim().length > 0) {
      hideStopButton();
    }
  });

  function revealResultButtons() {
    const buttons = document.getElementById('result-buttons');
    if (buttons) buttons.removeAttribute('hidden');
  }

  function hideStopButton() {
    const btn = document.getElementById('stop-btn');
    if (btn) btn.setAttribute('hidden', '');
  }
})();
