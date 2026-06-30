// PCB 결함 검사 데모 — 프론트엔드 로직
// 순수 JS (프레임워크 없음), async/await 사용

"use strict";

// ── 초기화 ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  loadSamples();
  loadJudgeConfig();
  setupUpload();
});


// ── 샘플 목록 로드 ────────────────────────────────────────────────────────────

async function loadSamples() {
  try {
    const res = await fetch("/samples");
    const data = await res.json();
    renderSampleList(data.samples);
  } catch (e) {
    console.error("샘플 목록 로드 실패:", e);
  }
}

function renderSampleList(samples) {
  const list = document.getElementById("sample-list");
  const empty = document.getElementById("sample-empty");

  if (!samples || samples.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  samples.forEach((name) => {
    const item = document.createElement("div");
    item.className = "sample-item";
    item.dataset.filename = name;
    item.innerHTML = `
      <img class="sample-thumb" src="/sample-files/${encodeURIComponent(name)}" alt="${name}">
      <span class="sample-label">${name}</span>
    `;
    item.addEventListener("click", () => {
      setActiveSample(item);
      inspectSample(name);
    });
    list.appendChild(item);
  });
}

function setActiveSample(activeItem) {
  document.querySelectorAll(".sample-item").forEach((el) =>
    el.classList.remove("active")
  );
  activeItem.classList.add("active");
}


// ── 판정 기준 로드 (recall 우선 설계 가시화) ────────────────────────────────

async function loadJudgeConfig() {
  try {
    const res = await fetch("/judge-config");
    const data = await res.json();
    const confThr = data.conf_threshold ?? 0.5;
    const [lo, hi] = data.review_band ?? [0.3, 0.5];

    document.getElementById("thr-ng").textContent = confThr.toFixed(2);
    document.getElementById("thr-lo").textContent = lo.toFixed(2);
    document.getElementById("thr-hi").textContent = hi.toFixed(2);
  } catch (e) {
    console.error("판정 기준 로드 실패:", e);
  }
}


// ── 업로드 설정 ───────────────────────────────────────────────────────────────

function setupUpload() {
  const input = document.getElementById("upload-input");
  const btn = document.getElementById("upload-btn");

  btn.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      // 업로드 시 샘플 선택 해제
      document.querySelectorAll(".sample-item").forEach((el) =>
        el.classList.remove("active")
      );
      inspectUpload(file);
    }
    input.value = ""; // 같은 파일 재선택 허용
  });
}


// ── 검사 요청 ─────────────────────────────────────────────────────────────────

async function inspectSample(filename) {
  const fd = new FormData();
  fd.append("source", "sample");
  fd.append("filename", filename);
  await doInspect(fd);
}

async function inspectUpload(file) {
  const fd = new FormData();
  fd.append("source", "upload");
  fd.append("file", file);
  await doInspect(fd);
}

async function doInspect(formData) {
  showLoading();
  setStatus("추론 중...", "neutral");

  try {
    const res = await fetch("/inspect", { method: "POST", body: formData });

    if (!res.ok) {
      let msg = `서버 오류 (${res.status})`;
      try {
        const err = await res.json();
        msg = err.detail || msg;
      } catch (_) {}
      throw new Error(msg);
    }

    const data = await res.json();
    renderResult(data);

  } catch (e) {
    setStatus("오류", "error");
    showError(e.message);
  } finally {
    hideLoading();
  }
}


// ── 결과 렌더링 ──────────────────────────────────────────────────────────────

function renderResult(data) {
  const { verdict, defect_count, by_class, defects, review, image_b64 } = data;

  // 주석 이미지 표시
  const img = document.getElementById("result-image");
  const placeholder = document.getElementById("image-placeholder");
  img.src = image_b64;
  img.classList.remove("hidden");
  placeholder.style.display = "none";

  // 판정 배지
  renderVerdict(verdict);

  // 헤더 상태바
  setStatus(verdict, verdict.toLowerCase());

  // 결함 총계
  document.getElementById("defect-count").textContent = defect_count;

  // 클래스별 개수
  renderByClass(by_class);

  // 결함 목록
  renderDefectList(defects);

  // REVIEW 섹션
  renderReviewSection(verdict, review);
}

function renderVerdict(verdict) {
  const badge = document.getElementById("verdict-badge");
  const text = document.getElementById("verdict-text");
  const sub = document.getElementById("verdict-sub");

  badge.className = `badge badge-${verdict.toLowerCase()}`;
  text.textContent = verdict;

  const subText = {
    OK: "양품 — 이상 없음",
    NG: "불량 — 결함 확인됨",
    REVIEW: "수동 검토 필요",
  };
  sub.textContent = subText[verdict] ?? verdict;
}

function renderByClass(byClass) {
  const container = document.getElementById("by-class-list");
  container.innerHTML = "";

  const entries = Object.entries(byClass);
  if (entries.length === 0) {
    container.innerHTML = '<span class="text-secondary">결함 없음</span>';
    return;
  }

  entries.forEach(([cls, cnt]) => {
    const row = document.createElement("div");
    row.className = "by-class-row";
    row.innerHTML = `
      <span class="cls-name">${cls}</span>
      <span class="cls-count">${cnt}</span>
    `;
    container.appendChild(row);
  });
}

function renderDefectList(defects) {
  const container = document.getElementById("defect-list");
  container.innerHTML = "";

  if (defects.length === 0) {
    container.innerHTML = '<span class="text-secondary">결함 없음</span>';
    return;
  }

  defects.forEach((d, i) => {
    const cx = Math.round(d.center[0]);
    const cy = Math.round(d.center[1]);
    const conf = (d.conf * 100).toFixed(1) + "%";

    const row = document.createElement("div");
    row.className = "defect-row";
    row.innerHTML = `
      <span class="defect-idx">${i + 1}</span>
      <span class="defect-cls">${d.class_name}</span>
      <span class="defect-conf">${conf}</span>
      <span class="defect-pos">(${cx},${cy})</span>
    `;
    container.appendChild(row);
  });
}

function renderReviewSection(verdict, reviewItems) {
  const section = document.getElementById("review-section");
  const list = document.getElementById("review-list");

  if (verdict === "REVIEW" && reviewItems && reviewItems.length > 0) {
    section.classList.remove("hidden");
    list.innerHTML = reviewItems
      .map(
        (d) =>
          `<div class="review-item">${d.class_name} — ${(d.conf * 100).toFixed(1)}%</div>`
      )
      .join("");
  } else {
    section.classList.add("hidden");
  }
}


// ── UI 헬퍼 ──────────────────────────────────────────────────────────────────

function showLoading() {
  document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
  document.getElementById("loading-overlay").classList.add("hidden");
}

function setStatus(text, type = "neutral") {
  const bar = document.getElementById("status-bar");
  bar.textContent = text;
  bar.className = `status-${type}`;
}

function showError(message) {
  // TODO: 전용 에러 토스트로 교체 가능
  alert(`검사 오류: ${message}`);
}
