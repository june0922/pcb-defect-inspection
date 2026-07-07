let images = [];
let modelsInfo = [];
let currentIndex = 0;
let isPlaying = false;
let playInterval = null;
let currentSpeed = 1; // 1x = 1000ms

const DOM = {
    filename: document.getElementById('filename-display'),
    progressInfo: document.getElementById('progress-info'),
    
    // Controls
    btnPrev: document.getElementById('btn-prev'),
    btnNext: document.getElementById('btn-next'),
    btnPlay: document.getElementById('btn-play'),
    iconPlay: document.getElementById('icon-play'),
    iconPause: document.getElementById('icon-pause'),
    speedBtns: document.querySelectorAll('.speed-btn')
};

// Data loading
function loadData() {
    try {
        if (typeof RESULTS_DATA === 'undefined') {
            throw new Error("RESULTS_DATA is not defined. Run generate_results.py first.");
        }
        
        images = RESULTS_DATA.images;
        modelsInfo = RESULTS_DATA.models;
        
        initModels(modelsInfo);
        
        if (images.length > 0) {
            updateView();
        }
    } catch (e) {
        console.error("Failed to load results:", e);
        DOM.filename.textContent = "Error loading data. Run generate_results.py first.";
    }
}

function initModels(models) {
    const format = val => (val * 100).toFixed(2) + '%';
    
    models.forEach((m, idx) => {
        // We assume index 0 -> model_a, index 1 -> model_b
        const sideId = idx === 0 ? 'model_a' : 'model_b';
        
        // Update Title & Tooltip
        const titleEl = document.getElementById(`${sideId}-title`);
        if (titleEl) {
            titleEl.textContent = m.title;
            titleEl.setAttribute('data-tooltip', m.tooltip);
        }
        
        // Update Metrics
        const recallEl = document.getElementById(`${sideId}-recall`);
        const map50El = document.getElementById(`${sideId}-map50`);
        const map5095El = document.getElementById(`${sideId}-map5095`);
        
        if (recallEl) recallEl.textContent = format(m.metrics.recall);
        if (map50El) map50El.textContent = format(m.metrics.map50);
        if (map5095El) map5095El.textContent = format(m.metrics.map50_95);
    });
}

function updateView() {
    if(images.length === 0) return;
    
    const imgName = images[currentIndex];
    DOM.filename.textContent = imgName;
    DOM.progressInfo.textContent = `Image: ${currentIndex + 1} / ${images.length}`;
    
    modelsInfo.forEach((m, idx) => {
        const sideId = idx === 0 ? 'model_a' : 'model_b';
        const imgEl = document.getElementById(`${sideId}-img`);
        if (imgEl) {
            imgEl.src = `results/${m.result_dir}/${imgName}`;
        }
    });
}

// Navigation
function goNext() {
    currentIndex = (currentIndex + 1) % images.length;
    updateView();
}

function goPrev() {
    currentIndex = (currentIndex - 1 + images.length) % images.length;
    updateView();
}

// Playback
function togglePlay() {
    isPlaying = !isPlaying;
    
    if (isPlaying) {
        DOM.iconPlay.style.display = 'none';
        DOM.iconPause.style.display = 'block';
        startInterval();
    } else {
        DOM.iconPlay.style.display = 'block';
        DOM.iconPause.style.display = 'none';
        stopInterval();
    }
}

function startInterval() {
    stopInterval();
    const delay = 1000 / currentSpeed;
    playInterval = setInterval(goNext, delay);
}

function stopInterval() {
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

function stopAutoPlayOnManualAction() {
    if (isPlaying) {
        togglePlay();
    }
}

// Event Listeners
DOM.btnPrev.addEventListener('click', () => {
    stopAutoPlayOnManualAction();
    goPrev();
});

DOM.btnNext.addEventListener('click', () => {
    stopAutoPlayOnManualAction();
    goNext();
});

DOM.btnPlay.addEventListener('click', togglePlay);

DOM.speedBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        // Update active state
        DOM.speedBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        // Update speed
        currentSpeed = parseInt(e.target.dataset.speed);
        
        // Restart interval if playing
        if (isPlaying) {
            startInterval();
        }
    });
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') {
        stopAutoPlayOnManualAction();
        goNext();
    } else if (e.key === 'ArrowLeft') {
        stopAutoPlayOnManualAction();
        goPrev();
    } else if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
    }
});

// Init
document.addEventListener('DOMContentLoaded', loadData);
