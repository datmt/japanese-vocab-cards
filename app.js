const CHUNK_SIZE = 100;
const TOTAL_WORDS = 5000;
const QUIZ_QUESTIONS = 10;
const QUIZ_OPTIONS = 4;

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

function renderSidebar() {
  const nChunks = Math.ceil(TOTAL_WORDS / CHUNK_SIZE);
  for (let i = 0; i < nChunks; i++) {
    const start = i * CHUNK_SIZE + 1;
    const end = Math.min((i + 1) * CHUNK_SIZE, TOTAL_WORDS);
    const li = document.createElement('li');
    li.textContent = `${start}–${end}`;
    li.dataset.start = start;
    li.dataset.end = end;
    li.addEventListener('click', () => {
      document.querySelectorAll('#chunk-list li').forEach((el) => el.classList.remove('active'));
      li.classList.add('active');
      renderChunk(start, end);
    });
    sidebarList.appendChild(li);
  }
}

function renderChunk(start, end) {
  activeChunk = { start, end };
  const words = queryAll(
    'SELECT rank, headword, reading_llm, reading_dict, gloss, pos, mnemonic, has_kanji, image_path FROM words WHERE rank BETWEEN ? AND ? ORDER BY rank',
    [start, end]
  );

  content.innerHTML = '';

  const header = document.createElement('div');
  header.className = 'chunk-header';
  header.innerHTML = `<h2>Words ${start}–${end}</h2>`;
  const quizBtn = document.createElement('button');
  quizBtn.className = 'quiz-btn';
  quizBtn.textContent = 'Quiz this chunk';
  quizBtn.addEventListener('click', () => startQuiz(start, end));
  header.appendChild(quizBtn);
  content.appendChild(header);

  const list = document.createElement('div');
  words.forEach((w) => list.appendChild(renderWordRow(w)));
  content.appendChild(list);
}

function renderWordRow(w) {
  const row = document.createElement('div');
  row.className = 'word-row';

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
  summary.addEventListener('click', () => {
    if (detailEl) {
      detailEl.remove();
      detailEl = null;
      return;
    }
    detailEl = renderWordDetail(w);
    row.appendChild(detailEl);
  });

  row.appendChild(summary);
  return row;
}

function renderWordDetail(w) {
  const detail = document.createElement('div');
  detail.className = 'word-detail';

  const body = document.createElement('div');
  body.className = 'detail-body';

  if (w.has_kanji && w.mnemonic) {
    const m = document.createElement('p');
    m.className = 'mnemonic';
    m.textContent = w.mnemonic;
    body.appendChild(m);
  }

  const examples = queryAll(
    'SELECT id, jp, jp_reading, en, audio_path, grammar_note, reading_mismatch, reading_dict FROM examples WHERE word_rank = ? ORDER BY id',
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
      ${ex.reading_mismatch ? `<div class="reading-warning" title="Dictionary parser reads this as ${escapeHtml(ex.reading_dict)} — needs human check">⚠ reading needs check (dict: ${escapeHtml(ex.reading_dict)})</div>` : ''}
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
    'SELECT rank, headword, reading_llm, reading_dict, gloss FROM words WHERE rank BETWEEN ? AND ? ORDER BY rank',
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
}

main();
