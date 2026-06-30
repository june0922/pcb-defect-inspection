'use strict';

// ── 상수 ──────────────────────────────────────────────────────
const ANIM_DELAY_MS = 230;  // 셀당 애니메이션 속도 (ms)

const CELL_STYLES = {
  OK:     { fill: 'rgba(46, 160, 67, 0.22)',   stroke: '#2ea043', icon: '✓' },
  NG:     { fill: 'rgba(248, 81, 73, 0.28)',   stroke: '#f85149', icon: '✕' },
  REVIEW: { fill: 'rgba(210, 153, 34, 0.25)', stroke: '#d29922', icon: '?' },
  SCAN:   { fill: 'rgba(245, 200, 66, 0.18)', stroke: '#f5c842', icon: '' },
};

// ── 상태 ──────────────────────────────────────────────────────
const state = {
  selectedBoard: 'ok_board',
  inspectionData: null,   // POST /inspect_board 응답
  animIndex: 0,
  reviewDecisions: {},    // 'row_col' → 'OK' | 'NG'
  reviewing: false,
};

// ── 초기화 ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadThresholds();
  await checkBoards();

  document.querySelectorAll('.board-btn').forEach(btn => {
    btn.addEventListener('click', () => selectBoard(btn.dataset.board));
  });

  document.getElementById('start-btn').addEventListener('click', startInspection);
  document.getElementById('confirm-review-btn').addEventListener('click', confirmReview);
});

// ── 판정 기준 표시 ────────────────────────────────────────────
async function loadThresholds() {
  try {
    const data = await fetch('/judge-config').then(r => r.json());
    document.getElementById('thr-ng').textContent = data.conf_threshold.toFixed(2);
    document.getElementById('thr-lo').textContent = data.review_band[0].toFixed(2);
    document.getElementById('thr-hi').textContent = data.review_band[1].toFixed(2);
  } catch (_) { /* 실패해도 기본값 유지 */ }
}

// ── 보드 존재 여부 확인 ───────────────────────────────────────
async function checkBoards() {
  try {
    const { boards } = await fetch('/boards').then(r => r.json());
    if (boards.length === 0) {
      setPlaceholderHint('보드 파일 없음 — <code>python web/tools/build_demo_boards.py</code>');
      document.getElementById('start-btn').disabled = true;
    } else {
      setPlaceholderHint('가상 보드 생성: <code>python web/tools/build_demo_boards.py</code>');
    }
  } catch (_) { /* 네트워크 오류 무시 */ }
}

// ── 보드 선택 ────────────────────────────────────────────────
function selectBoard(boardId) {
  state.selectedBoard = boardId;
  document.querySelectorAll('.board-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.board === boardId);
  });
}

// ── 검사 시작 ────────────────────────────────────────────────
async function startInspection() {
  const boardId = state.selectedBoard;
  state.inspectionData = null;
  state.animIndex = 0;
  state.reviewDecisions = {};
  state.reviewing = false;

  const startBtn = document.getElementById('start-btn');
  startBtn.disabled = true;
  startBtn.textContent = '▶ 검사 중...';

  setStatus('서버 추론 중...', 'neutral');
  hideReviewPanel();
  resetResultPanel();
  clearCanvas();
  document.getElementById('progress-label').classList.remove('hidden');
  document.getElementById('scan-current').textContent = '0';

  let data;
  try {
    const form = new FormData();
    form.append('board_id', boardId);
    const resp = await fetch('/inspect_board', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    data = await resp.json();
  } catch (e) {
    setStatus('오류', 'error');
    setPlaceholderHint(e.message);
    startBtn.disabled = false;
    startBtn.textContent = '▶ 검사 시작';
    document.getElementById('progress-label').classList.add('hidden');
    return;
  }

  state.inspectionData = data;
  document.getElementById('scan-total').textContent = data.cells.length;

  await showBoardImage(boardId);
  animateInspection();
}

// ── 보드 이미지 표시 ────────────────────────────────────────
function showBoardImage(boardId) {
  return new Promise((resolve, reject) => {
    const placeholder = document.getElementById('board-placeholder');
    const wrap = document.getElementById('board-wrap');
    const img  = document.getElementById('board-img');
    const canvas = document.getElementById('board-canvas');

    placeholder.classList.add('hidden');
    wrap.classList.remove('hidden');

    img.onload = () => {
      // canvas 크기를 표시된 img 크기에 맞춤 (display px 기준)
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
      canvas.style.width  = img.offsetWidth  + 'px';
      canvas.style.height = img.offsetHeight + 'px';
      resolve();
    };
    img.onerror = () => reject(new Error('보드 이미지 로드 실패'));
    img.src = `/sample-files/boards/${boardId}.jpg?t=${Date.now()}`;
  });
}

// ── 애니메이션 루프 ──────────────────────────────────────────
function animateInspection() {
  const cells = state.inspectionData.cells;

  function step() {
    const idx = state.animIndex;

    // 이전 SCAN 셀 → 확정 색상으로 교체
    if (idx > 0) {
      drawCell(cells[idx - 1]);
    }

    // 모든 셀 완료
    if (idx >= cells.length) {
      finishInspection();
      return;
    }

    // 현재 셀 스캔 표시
    drawCellScan(cells[idx]);
    document.getElementById('scan-current').textContent = idx + 1;
    setStatus(`검사 중 ${idx + 1}/${cells.length}`, 'neutral');

    state.animIndex++;
    setTimeout(step, ANIM_DELAY_MS);
  }

  step();
}

// ── 캔버스 셀 그리기 ─────────────────────────────────────────
function getCanvas() {
  return document.getElementById('board-canvas');
}

function drawCellScan(cell) {
  const canvas = getCanvas();
  const ctx = canvas.getContext('2d');
  const { grid_cols } = state.inspectionData;
  const cellPx = canvas.width / grid_cols;
  const x = cell.col * cellPx;
  const y = cell.row * cellPx;
  const s = CELL_STYLES.SCAN;

  ctx.clearRect(x, y, cellPx, cellPx);
  ctx.fillStyle = s.fill;
  ctx.fillRect(x, y, cellPx, cellPx);
  ctx.strokeStyle = s.stroke;
  ctx.lineWidth = 3;
  ctx.strokeRect(x + 1.5, y + 1.5, cellPx - 3, cellPx - 3);

  // 스캔 아이콘 (회전하는 화살표 대신 간단한 텍스트)
  ctx.fillStyle = s.stroke;
  ctx.font = `bold ${Math.round(cellPx * 0.3)}px monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('⟳', x + cellPx / 2, y + cellPx / 2);
}

function drawCell(cell, verdictOverride) {
  const canvas = getCanvas();
  const ctx = canvas.getContext('2d');
  const { grid_cols } = state.inspectionData;
  const cellPx = canvas.width / grid_cols;
  const verdict = verdictOverride || cell.verdict;
  const x = cell.col * cellPx;
  const y = cell.row * cellPx;
  const s = CELL_STYLES[verdict] || CELL_STYLES.OK;

  ctx.clearRect(x, y, cellPx, cellPx);
  ctx.fillStyle = s.fill;
  ctx.fillRect(x, y, cellPx, cellPx);
  ctx.strokeStyle = s.stroke;
  ctx.lineWidth = 3;
  ctx.strokeRect(x + 1.5, y + 1.5, cellPx - 3, cellPx - 3);

  // 판정 아이콘
  ctx.fillStyle = s.stroke;
  ctx.font = `bold ${Math.round(cellPx * 0.3)}px monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(s.icon, x + cellPx / 2, y + cellPx / 2);
}

function clearCanvas() {
  const canvas = getCanvas();
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// ── 검사 완료 처리 ───────────────────────────────────────────
function finishInspection() {
  const data = state.inspectionData;

  document.getElementById('progress-label').classList.add('hidden');
  const startBtn = document.getElementById('start-btn');
  startBtn.disabled = false;
  startBtn.textContent = '▶ 검사 시작';

  showBoardVerdict(data.board_verdict);
  showSummary(data.summary);
  showCellDetails(data.cells);
  setStatus(data.board_verdict, data.board_verdict.toLowerCase());

  if (data.summary.review > 0) {
    showReviewPanel(data);
  }
}

// ── 우측 결과 패널 ───────────────────────────────────────────
function showBoardVerdict(verdict) {
  const badge   = document.getElementById('board-verdict-badge');
  const text    = document.getElementById('board-verdict-text');
  const sub     = document.getElementById('board-verdict-sub');
  const desc    = document.getElementById('board-verdict-desc');

  badge.className = `badge badge-${verdict.toLowerCase()}`;
  text.textContent = verdict;

  const info = {
    OK:     ['모든 칸 이상 없음',  '보드 전체 정상 판정. 결함이 감지되지 않았습니다.'],
    NG:     ['결함 감지됨',        '확실한 결함이 하나 이상 존재합니다. 보드를 교체하세요.'],
    REVIEW: ['수동 검토 필요',     '애매한 신뢰도의 결함이 있습니다. 하단에서 직접 확인하세요.'],
  };
  const [subText, descText] = info[verdict] || ['—', '—'];
  sub.textContent  = subText;
  desc.textContent = descText;
}

function showSummary(summary) {
  document.getElementById('board-summary').classList.remove('hidden');
  document.getElementById('sum-ok').textContent = summary.ok;
  document.getElementById('sum-ng').textContent = summary.ng;
  document.getElementById('sum-rv').textContent = summary.review;
}

function showCellDetails(cells) {
  const list = document.getElementById('cell-detail-list');
  const nonOk = cells.filter(c => c.verdict !== 'OK');
  if (!nonOk.length) {
    list.innerHTML = '<span class="text-secondary">이상 칸 없음</span>';
    return;
  }
  list.innerHTML = nonOk.map(c => `
    <div class="cell-detail-row">
      <span class="cell-pos">[${c.row + 1},${c.col + 1}]</span>
      <span class="cell-verdict-badge v-${c.verdict.toLowerCase()}">${c.verdict}</span>
      <span class="cell-count">${c.defect_count}개</span>
    </div>
  `).join('');
}

function resetResultPanel() {
  document.getElementById('board-verdict-badge').className = 'badge badge-idle';
  document.getElementById('board-verdict-text').textContent = '—';
  document.getElementById('board-verdict-sub').textContent  = '검사 대기';
  document.getElementById('board-verdict-desc').textContent = '—';
  document.getElementById('cell-detail-list').innerHTML = '<span class="text-secondary">—</span>';
  document.getElementById('board-summary').classList.add('hidden');
}

// ── REVIEW 패널 ──────────────────────────────────────────────
function showReviewPanel(data) {
  const panel      = document.getElementById('review-panel');
  const cards      = document.getElementById('review-cards');
  const confirmBtn = document.getElementById('confirm-review-btn');

  const reviewCells = data.cells.filter(c => c.verdict === 'REVIEW');

  cards.innerHTML = reviewCells.map(cell => {
    const defectTags = Object.entries(cell.by_class)
      .map(([cls, cnt]) => `<span class="rcard-cls">${cls}:${cnt}</span>`)
      .join('');
    return `
      <div class="review-card" id="rcard-${cell.row}-${cell.col}">
        <div class="rcard-header">
          <span class="rcard-pos">${cell.row + 1}행 ${cell.col + 1}열</span>
          <span class="rcard-count">${cell.defect_count}개 감지</span>
        </div>
        <img class="rcard-img"
             src="/board_cell/${data.board_id}/${cell.row}/${cell.col}"
             alt="${cell.row + 1}행 ${cell.col + 1}열 이미지"
             loading="lazy">
        <div class="rcard-defects">${defectTags || '<span class="rcard-cls">없음</span>'}</div>
        <div class="rcard-actions">
          <button class="btn-manual ok-btn"
                  onclick="manualDecide(${cell.row}, ${cell.col}, 'OK')">✓ 정상</button>
          <button class="btn-manual ng-btn"
                  onclick="manualDecide(${cell.row}, ${cell.col}, 'NG')">✕ 결함</button>
        </div>
      </div>
    `;
  }).join('');

  panel.classList.remove('hidden');
  confirmBtn.classList.add('hidden');
  state.reviewing = true;
}

function hideReviewPanel() {
  document.getElementById('review-panel').classList.add('hidden');
  state.reviewing = false;
}

// ── 수동 분류 ────────────────────────────────────────────────
function manualDecide(row, col, decision) {
  state.reviewDecisions[`${row}_${col}`] = decision;

  const card = document.getElementById(`rcard-${row}-${col}`);
  card.classList.remove('decided-ok', 'decided-ng');
  card.classList.add(`decided-${decision.toLowerCase()}`);

  // 캔버스에 수동 판정 결과 즉시 반영
  const cell = state.inspectionData.cells.find(c => c.row === row && c.col === col);
  if (cell) drawCell(cell, decision);

  // 모든 REVIEW 셀에 결정이 내려졌는지 확인
  const reviewCells = state.inspectionData.cells.filter(c => c.verdict === 'REVIEW');
  if (Object.keys(state.reviewDecisions).length >= reviewCells.length) {
    document.getElementById('confirm-review-btn').classList.remove('hidden');
  }
}

// ── 판정 확정 ────────────────────────────────────────────────
function confirmReview() {
  const cells = state.inspectionData.cells;

  // 수동 분류 결과를 셀 데이터에 반영
  const updated = cells.map(c => ({
    ...c,
    verdict: state.reviewDecisions[`${c.row}_${c.col}`] || c.verdict,
  }));

  const hasNG     = updated.some(c => c.verdict === 'NG');
  const hasReview = updated.some(c => c.verdict === 'REVIEW');
  const newVerdict = hasNG ? 'NG' : hasReview ? 'REVIEW' : 'OK';

  showBoardVerdict(newVerdict);
  setStatus(newVerdict, newVerdict.toLowerCase());

  const ok = updated.filter(c => c.verdict === 'OK').length;
  const ng = updated.filter(c => c.verdict === 'NG').length;
  const rv = updated.filter(c => c.verdict === 'REVIEW').length;
  showSummary({ ok, ng, review: rv });
  showCellDetails(updated);

  hideReviewPanel();
}

// ── 유틸 ─────────────────────────────────────────────────────
function setStatus(text, cls) {
  const el = document.getElementById('status-bar');
  el.textContent = text;
  el.className = `status-${cls}`;
}

function setPlaceholderHint(html) {
  const hint = document.getElementById('board-hint');
  if (hint) hint.innerHTML = html;
}
