let images = [];
let currentIndex = 0;
let isPlaying = false;
let playInterval = null;
let currentSpeed = 1; // 1x = 1000ms

const DOM = {
    notuneImg: document.getElementById('notune-img'),
    yestuneImg: document.getElementById('yestune-img'),
    filename: document.getElementById('filename-display'),
    progressInfo: document.getElementById('progress-info'),
    
    // Metrics
    notuneRecall: document.getElementById('notune-recall'),
    notuneMap50: document.getElementById('notune-map50'),
    notuneMap5095: document.getElementById('notune-map5095'),
    yestuneRecall: document.getElementById('yestune-recall'),
    yestuneMap50: document.getElementById('yestune-map50'),
    yestuneMap5095: document.getElementById('yestune-map5095'),
    
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
        updateMetrics(RESULTS_DATA.metrics);
        
        if (images.length > 0) {
            updateView();
        }
    } catch (e) {
        console.error("Failed to load results:", e);
        DOM.filename.textContent = "Error loading data. Run generate_results.py first.";
    }
}

function updateMetrics(m) {
    const format = val => (val * 100).toFixed(2) + '%';
    
    DOM.notuneRecall.textContent = format(m.notune.recall);
    DOM.notuneMap50.textContent = format(m.notune.map50);
    DOM.notuneMap5095.textContent = format(m.notune.map50_95);
    
    DOM.yestuneRecall.textContent = format(m.yestune.recall);
    DOM.yestuneMap50.textContent = format(m.yestune.map50);
    DOM.yestuneMap5095.textContent = format(m.yestune.map50_95);
}

function updateView() {
    if(images.length === 0) return;
    
    const imgName = images[currentIndex];
    DOM.filename.textContent = imgName;
    DOM.progressInfo.textContent = `Image: ${currentIndex + 1} / ${images.length}`;
    
    // Force image reload cleanly to avoid flickering if possible
    DOM.notuneImg.src = `results/patience15_old/${imgName}`;
    DOM.yestuneImg.src = `results/patience15_new/${imgName}`;
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
