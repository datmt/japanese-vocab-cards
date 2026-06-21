const CHUNK_SIZE = 100;
const TOTAL_WORDS = 5000;
const QUIZ_QUESTIONS = 10;
const QUIZ_OPTIONS = 4;
const FEEDBACK_FORM = 'https://docs.google.com/forms/d/e/1FAIpQLSfq0cHmELEgXmvxHRJHlXV176v0CimWy58ghNB6_9AddrpUeA/viewform';
const FEEDBACK_ENTRY_TYPE = 'entry.1813869599';
const FEEDBACK_ENTRY_DETAILS = 'entry.999717781';

let db = null;
let activeChunk = null;

const sidebarList = document.getElementById('chunk-list');
const content = document.getElementById('content');
const loading = document.getElementById('loading');

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

async function initDb() {
  const SQL = await initSqlJs({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/sql.js@1.10.3/dist/${file}`,
  });
  const buf = await fetch('./vocab.db').then((r) => r.arrayBuffer());
  db = new SQL.Database(new Uint8Array(buf));
  loading.remove();
}

function queryAll(sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

function chunkIndexForRank(rank) {
  return Math.floor((rank - 1) / CHUNK_SIZE) + 1;
}

function chunkHash(chunkNum, rank) {
  return rank ? `#chunk=${chunkNum}&rank=${rank}` : `#chunk=${chunkNum}`;
}

function parseHash() {
  const m = /^#chunk=(\d+)(?:&rank=(\d+))?$/.exec(location.hash);
  if (!m) return null;
  let chunkNum = parseInt(m[1], 10);
  const rank = m[2] ? parseInt(m[2], 10) : null;
  const nChunks = Math.ceil(TOTAL_WORDS / CHUNK_SIZE);
  if (rank != null) {
    if (rank < 1 || rank > TOTAL_WORDS) return null;
    chunkNum = chunkIndexForRank(rank);
  }
  if (chunkNum < 1 || chunkNum > nChunks) return null;
  return { chunkNum, rank };
}

function renderSidebar() {
  const nChunks = Math.ceil(TOTAL_WORDS / CHUNK_SIZE);
  for (let i = 1; i <= nChunks; i++) {
    const start = (i - 1) * CHUNK_SIZE + 1;
    const end = Math.min(i * CHUNK_SIZE, TOTAL_WORDS);
    const li = document.createElement('li');
    li.dataset.chunk = i;
    const a = document.createElement('a');
    a.href = chunkHash(i);
    a.textContent = `${start}–${end}`;
    li.appendChild(a);
    sidebarList.appendChild(li);
  }
}

function handleHashChange() {
  const parsed = parseHash();
  document.querySelectorAll('#chunk-list li').forEach((el) => {
    el.classList.toggle('active', parsed != null && el.dataset.chunk === String(parsed.chunkNum));
  });
  if (parsed == null) {
    content.innerHTML = '<p class="hint">Pick a chunk from the left to start.</p>';
    return;
  }
  const { chunkNum, rank } = parsed;
  const start = (chunkNum - 1) * CHUNK_SIZE + 1;
  const end = Math.min(chunkNum * CHUNK_SIZE, TOTAL_WORDS);
  renderChunk(start, end, rank);
}

function renderChunk(start, end, focusRank) {
  activeChunk = { start, end };
  const chunkNum = chunkIndexForRank(start);
  const words = queryAll(
    'SELECT rank, headword, reading_llm, reading_dict, gloss, pos, mnemonic, has_kanji, image_path FROM words WHERE rank BETWEEN ? AND ? ORDER BY rank',
    [start, end]
  );

  content.innerHTML = '';

  const header = document.createElement('div');
  header.className = 'chunk-header';
  header.innerHTML = `<h2>Words ${start}–${end}</h2>`;

  const actions = document.createElement('div');
  actions.className = 'chunk-actions';

  const copyBtn = document.createElement('button');
  copyBtn.className = 'primary';
  copyBtn.textContent = 'Copy link to this batch';
  copyBtn.addEventListener('click', () => {
    const url = `${location.origin}${location.pathname}${chunkHash(chunkNum)}`;
    navigator.clipboard.writeText(url).then(() => {
      copyBtn.textContent = 'Link copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy link to this batch'; }, 1500);
    });
  });
  actions.appendChild(copyBtn);

  const quizBtn = document.createElement('button');
  quizBtn.className = 'quiz-btn';
  quizBtn.textContent = 'Quiz this chunk';
  quizBtn.addEventListener('click', () => startQuiz(start, end));
  actions.appendChild(quizBtn);

  header.appendChild(actions);
  content.appendChild(header);

  const list = document.createElement('div');
  words.forEach((w) => list.appendChild(renderWordRow(w, chunkNum, focusRank)));
  content.appendChild(list);

  if (focusRank) {
    const focusRow = list.querySelector(`[data-rank="${focusRank}"]`);
    if (focusRow) focusRow.scrollIntoView({ block: 'center' });
  }
}

function renderWordRow(w, chunkNum, focusRank) {
  const row = document.createElement('div');
  row.className = 'word-row';
  row.dataset.rank = w.rank;

  const summary = document.createElement('div');
  summary.className = 'word-summary';
  const reading = w.reading_llm || w.reading_dict || '';
  summary.innerHTML = `
    ${w.image_path ? `<img class="word-thumb" src="${escapeHtml(w.image_path)}" alt="" loading="lazy">` : ''}
    <span class="rank">#${w.rank}</span>
    <span class="headword">${escapeHtml(w.headword)}</span>
    <span class="reading">${escapeHtml(reading)}</span>
    <span class="gloss">${escapeHtml(w.gloss)}</span>
    ${w.pos ? `<span class="pos">${escapeHtml(w.pos)}</span>` : ''}
  `;

  let detailEl = null;
  function openDetail() {
    detailEl = renderWordDetail(w, chunkNum);
    row.appendChild(detailEl);
    history.replaceState(null, '', chunkHash(chunkNum, w.rank));
  }
  function closeDetail() {
    detailEl.remove();
    detailEl = null;
    history.replaceState(null, '', chunkHash(chunkNum));
  }
  summary.addEventListener('click', () => {
    if (detailEl) closeDetail();
    else openDetail();
  });

  row.appendChild(summary);
  if (focusRank === w.rank) openDetail();
  return row;
}

function feedbackFormUrl(type, details) {
  const params = new URLSearchParams({
    usp: 'pp_url',
    [FEEDBACK_ENTRY_TYPE]: type,
    [FEEDBACK_ENTRY_DETAILS]: details,
  });
  return `${FEEDBACK_FORM}?${params.toString()}`;
}

function wordFeedbackUrl(w, chunkNum) {
  const itemUrl = `${location.origin}${location.pathname}${chunkHash(chunkNum, w.rank)}`;
  const reading = w.reading_llm || w.reading_dict || '';
  const details = [
    `Word: ${w.headword} (#${w.rank})`,
    `Reading: ${reading}`,
    `Gloss: ${w.gloss || ''}`,
    `Link: ${itemUrl}`,
  ].join('\n');
  return feedbackFormUrl('Bug Report', details);
}

function renderWordDetail(w, chunkNum) {
  const detail = document.createElement('div');
  detail.className = 'word-detail';

  const body = document.createElement('div');
  body.className = 'detail-body';

  const detailActions = document.createElement('div');
  detailActions.className = 'detail-actions';

  const copyItemBtn = document.createElement('button');
  copyItemBtn.className = 'copy-item-link';
  copyItemBtn.textContent = 'Copy link to this word';
  copyItemBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const url = `${location.origin}${location.pathname}${chunkHash(chunkNum, w.rank)}`;
    navigator.clipboard.writeText(url).then(() => {
      copyItemBtn.textContent = 'Link copied!';
      setTimeout(() => { copyItemBtn.textContent = 'Copy link to this word'; }, 1500);
    });
  });
  detailActions.appendChild(copyItemBtn);

  const reportBtn = document.createElement('a');
  reportBtn.className = 'report-bug';
  reportBtn.href = wordFeedbackUrl(w, chunkNum);
  reportBtn.target = '_blank';
  reportBtn.rel = 'noopener';
  reportBtn.textContent = 'Report a bug / request a feature';
  reportBtn.addEventListener('click', (e) => e.stopPropagation());
  detailActions.appendChild(reportBtn);

  body.appendChild(detailActions);

  if (w.has_kanji && w.mnemonic) {
    const m = document.createElement('p');
    m.className = 'mnemonic';
    m.textContent = w.mnemonic;
    body.appendChild(m);
  }

  const examples = queryAll(
    'SELECT id, jp, jp_reading, en, audio_path, grammar_note FROM examples WHERE word_rank = ? ORDER BY id',
    [w.rank]
  );
  examples.forEach((ex) => {
    const exEl = document.createElement('div');
    exEl.className = 'example';
    exEl.innerHTML = `
      <div class="jp">${escapeHtml(ex.jp)}</div>
      <div class="reading">${escapeHtml(ex.jp_reading)}</div>
      <div class="en">${escapeHtml(ex.en)}</div>
      ${ex.audio_path ? `<audio controls src="${escapeHtml(ex.audio_path)}"></audio>` : ''}
      ${ex.grammar_note ? `<div class="grammar-note">${escapeHtml(ex.grammar_note)}</div>` : ''}
    `;

    const tokens = queryAll(
      'SELECT surface, reading, pos, meaning, note FROM example_breakdown WHERE example_id = ? ORDER BY seq',
      [ex.id]
    );
    if (tokens.length) {
      const breakdownEl = document.createElement('div');
      breakdownEl.className = 'breakdown';
      tokens.forEach((t) => {
        const chip = document.createElement('span');
        chip.className = 'breakdown-token';
        chip.innerHTML = `
          <span class="bt-surface">${escapeHtml(t.surface)}</span>
          <span class="bt-reading">${escapeHtml(t.reading)}</span>
          <span class="bt-meaning">${escapeHtml(t.meaning)}${t.pos ? ` <em>(${escapeHtml(t.pos)})</em>` : ''}${t.note ? `<br>${escapeHtml(t.note)}` : ''}</span>
        `;
        breakdownEl.appendChild(chip);
      });
      exEl.appendChild(breakdownEl);
    }

    body.appendChild(exEl);
  });

  detail.appendChild(body);

  if (w.image_path) {
    const img = document.createElement('img');
    img.className = 'detail-image';
    img.src = w.image_path;
    img.alt = '';
    img.loading = 'lazy';
    detail.appendChild(img);
  }

  return detail;
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function startQuiz(start, end) {
  const chunkWords = queryAll(
    'SELECT rank, headword, reading_llm, reading_dict, gloss, image_path FROM words WHERE rank BETWEEN ? AND ? ORDER BY rank',
    [start, end]
  );
  const allGlosses = queryAll('SELECT DISTINCT gloss FROM words WHERE gloss IS NOT NULL AND gloss != ""').map(
    (r) => r.gloss
  );

  const questions = shuffle(chunkWords).slice(0, Math.min(QUIZ_QUESTIONS, chunkWords.length)).map((w) => {
    const distractorPool = shuffle(allGlosses.filter((g) => g !== w.gloss));
    const distractors = distractorPool.slice(0, QUIZ_OPTIONS - 1);
    const options = shuffle([w.gloss, ...distractors]);
    return { word: w, options, correct: w.gloss };
  });

  let qIndex = 0;
  let score = 0;

  function renderQuestion() {
    content.innerHTML = '';
    const q = questions[qIndex];
    const reading = q.word.reading_llm || q.word.reading_dict || '';

    const progress = document.createElement('div');
    progress.className = 'quiz-progress';
    progress.textContent = `Question ${qIndex + 1} / ${questions.length} — score ${score}`;
    content.appendChild(progress);

    const qEl = document.createElement('div');
    qEl.className = 'quiz-question';
    qEl.textContent = q.word.headword;
    content.appendChild(qEl);

    if (q.word.image_path) {
      const img = document.createElement('img');
      img.className = 'quiz-image';
      img.src = q.word.image_path;
      img.alt = '';
      content.appendChild(img);
    }

    const readingEl = document.createElement('div');
    readingEl.className = 'quiz-reading';
    readingEl.textContent = reading;
    content.appendChild(readingEl);

    const optsEl = document.createElement('div');
    optsEl.className = 'quiz-options';
    let answered = false;
    q.options.forEach((opt) => {
      const btn = document.createElement('button');
      btn.textContent = opt;
      btn.addEventListener('click', () => {
        if (answered) return;
        answered = true;
        const isCorrect = opt === q.correct;
        if (isCorrect) {
          score++;
          btn.classList.add('correct');
        } else {
          btn.classList.add('incorrect');
          [...optsEl.children].find((b) => b.textContent === q.correct)?.classList.add('correct');
        }
        setTimeout(() => {
          qIndex++;
          if (qIndex < questions.length) {
            renderQuestion();
          } else {
            renderSummary();
          }
        }, 700);
      });
      optsEl.appendChild(btn);
    });
    content.appendChild(optsEl);
  }

  function renderSummary() {
    content.innerHTML = '';
    const summary = document.createElement('div');
    summary.className = 'quiz-summary';
    summary.textContent = `Score: ${score} / ${questions.length}`;
    content.appendChild(summary);

    const retryBtn = document.createElement('button');
    retryBtn.className = 'primary';
    retryBtn.textContent = 'Retry quiz';
    retryBtn.addEventListener('click', () => startQuiz(start, end));
    content.appendChild(retryBtn);

    const backBtn = document.createElement('button');
    backBtn.className = 'primary';
    backBtn.style.marginLeft = '0.5rem';
    backBtn.textContent = 'Back to word list';
    backBtn.addEventListener('click', () => renderChunk(start, end));
    content.appendChild(backBtn);
  }

  renderQuestion();
}

async function main() {
  renderSidebar();
  await initDb();
  window.addEventListener('hashchange', handleHashChange);
  handleHashChange();
}

main();
