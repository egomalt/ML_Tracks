"use strict";

const API_URL = "/search";

// ── colour palette for avatars ────────────────────────────────────────────
const AVATAR_COLORS = [
  ["#7c3aed", "#5b21b6"],
  ["#0891b2", "#0e7490"],
  ["#be185d", "#9d174d"],
  ["#b45309", "#92400e"],
  ["#047857", "#065f46"],
  ["#6d28d9", "#4c1d95"],
];

function avatarStyle(text) {
  let h = 0;
  for (const c of text) h = (h * 31 + c.charCodeAt(0)) & 0xffffffff;
  const [a, b] = AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
  return `background: linear-gradient(135deg, ${a}, ${b})`;
}

// ── DOM refs ──────────────────────────────────────────────────────────────
const form            = document.getElementById("searchForm");
const input           = document.getElementById("queryInput");
const loader          = document.getElementById("loader");
const errorBox        = document.getElementById("errorBox");
const resultsGrid     = document.getElementById("resultsGrid");
const submitBtn       = form.querySelector(".search-btn");
const translateToggle = document.getElementById("translateToggle");
const searchIcon      = form.querySelector(".search-icon");

let activeGenre = "";

// ── genre pills ───────────────────────────────────────────────────────────
document.querySelectorAll(".genre-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".genre-pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    activeGenre = pill.dataset.genre;
  });
});

// ── chip clicks ───────────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    input.focus();
  });
});

// ── form submit ───────────────────────────────────────────────────────────
form.addEventListener("submit", async e => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  await doSearch(query);
});

async function doSearch(query) {
  setLoading(true);
  clearResults();
  hideError();
  searchIcon.hidden = false;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 3, translate: translateToggle.checked, genre: activeGenre || null }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    const { results } = await res.json();

    if (!results || results.length === 0) {
      showError("Ничего не найдено. Попробуйте другой запрос.");
      return;
    }

    renderResults(results);
    searchIcon.hidden = true;
  } catch (err) {
    showError(`Ошибка: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function renderResults(results) {
  results.forEach((song, i) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${i * 80}ms`;

    const initial = (song.artist || song.title || "?")[0].toUpperCase();
    const scorePercent = Math.round(song.score * 100);
    const hasUrl = song.url && song.url.startsWith("http");

    card.innerHTML = `
      <div class="card-avatar" style="${avatarStyle(song.artist + song.title)}">
        ${initial}
      </div>
      <div class="card-body">
        <div class="card-top">
          <div class="card-title">${esc(song.title)}</div>
          <span class="card-badge">${esc(song.genre)}</span>
        </div>
        <div class="card-artist">${esc(song.artist)}</div>
        <div class="card-snippet">${esc(song.matched_text)}</div>
        <div class="card-footer">
          <div class="card-score">
            Сходство: <span>${scorePercent}%</span>
          </div>
          ${hasUrl ? `
          <a class="card-link" href="${esc(song.url)}" target="_blank" rel="noopener">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Слушать
          </a>` : ""}
        </div>
      </div>
    `;
    resultsGrid.appendChild(card);
  });
}

// ── helpers ───────────────────────────────────────────────────────────────
function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setLoading(on) {
  loader.hidden = !on;
  submitBtn.disabled = on;
}

function clearResults() {
  resultsGrid.innerHTML = "";
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}
