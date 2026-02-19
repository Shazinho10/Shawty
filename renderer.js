/**
 * Shawty - Renderer Process
 * Handles UI state management and user interactions
 */

// State
const state = {
    currentScreen: 'idle',
    videoPath: null,
    videoName: null,
    outputFolder: null,
    outputFileName: null,
    brandFile: null,
    logs: [],
    resultJson: null,
    clipsFolder: null,
    clips: []
};

// DOM Elements
const screens = {
    idle: document.getElementById('screen-idle'),
    selected: document.getElementById('screen-selected'),
    processing: document.getElementById('screen-processing'),
    completed: document.getElementById('screen-completed')
};

const elements = {
    dropZone: document.getElementById('dropZone'),
    fileName: document.getElementById('fileName'),
    filePath: document.getElementById('filePath'),
    selectedVideoThumbnail: document.getElementById('selectedVideoThumbnail'),
    selectedVideoPreview: document.getElementById('selectedVideoPreview'),
    selectedVideoDuration: document.getElementById('selectedVideoDuration'),
    selectedVideoResolution: document.getElementById('selectedVideoResolution'),
    selectedVideoSize: document.getElementById('selectedVideoSize'),
    selectedVideoAudio: document.getElementById('selectedVideoAudio'),
    outputFolderPreview: document.getElementById('outputFolderPreview'),
    outputFilePreview: document.getElementById('outputFilePreview'),
    modelSize: document.getElementById('modelSize'),
    llmProvider: document.getElementById('llmProvider'),
    apiKeyRow: document.getElementById('apiKeyRow'),
    apiKeyLabel: document.getElementById('apiKeyLabel'),
    apiKey: document.getElementById('apiKey'),
    diarizationToggle: document.getElementById('diarizationToggle'),
    hfTokenRow: document.getElementById('hfTokenRow'),
    hfToken: document.getElementById('hfToken'),
    advancedToggle: document.getElementById('advancedToggle'),
    advancedContent: document.getElementById('advancedContent'),
    brandFilePreview: document.getElementById('brandFilePreview'),
    autoOpenToggle: document.getElementById('autoOpenToggle'),
    portraitCropToggle: document.getElementById('portraitCropToggle'),
    processingStage: document.getElementById('processingStage'),
    processingPercent: document.getElementById('processingPercent'),
    processingCircle: document.getElementById('processingProgressCircle'),
    logsContent: document.getElementById('logsContent'),
    shortsList: document.getElementById('shortsList'),
    shortsCountBadge: document.getElementById('shortsCountBadge'),
    completedOutputPath: document.getElementById('completedOutputPath'),
    rawLogsToggle: document.getElementById('rawLogsToggle'),
    rawLogsPanel: document.getElementById('rawLogsPanel'),
    rawLogsContent: document.getElementById('rawLogsContent'),
    // Video modal elements
    videoModal: document.getElementById('videoModal'),
    videoModalOverlay: document.getElementById('videoModalOverlay'),
    videoModalClose: document.getElementById('videoModalClose'),
    videoModalTitle: document.getElementById('videoModalTitle'),
    clipThumbnail: document.getElementById('clipThumbnail')
};

const buttons = {
    changeFile: document.getElementById('changeFileBtn'),
    changeOutput: document.getElementById('changeOutputBtn'),
    selectBrand: document.getElementById('selectBrandBtn'),
    back: document.getElementById('backBtn'),
    convert: document.getElementById('convertBtn'),
    cancel: document.getElementById('cancelBtn'),
    openFolder: document.getElementById('openFolderBtn'),
    openJson: document.getElementById('openJsonBtn'),
    convertAnother: document.getElementById('convertAnotherBtn')
};

// Screen Management
function showScreen(screenName) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[screenName].classList.add('active');
    state.currentScreen = screenName;

    if (screenName === 'processing') {
        processingStageIndex = -1;
        setProcessingProgress(0);
    }
}

function setText(el, value) {
    if (!el) return;
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
        el.value = value ?? '';
    } else {
        el.textContent = value ?? '';
    }
}

// Generate timestamp for filename
function generateTimestamp() {
    const now = new Date();
    const pad = n => n.toString().padStart(2, '0');
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

// Sanitize filename for Windows
function sanitizeFilename(name) {
    return name.replace(/[<>:"/\\|?*]/g, '').replace(/\s+/g, '_').replace(/\.+$/, '');
}

function formatDuration(seconds) {
    const total = Math.max(0, Math.round(seconds || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatBytes(bytes) {
    const size = Number(bytes);
    if (!Number.isFinite(size) || size <= 0) return '?';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let idx = 0;
    let val = size;
    while (val >= 1024 && idx < units.length - 1) {
        val /= 1024;
        idx += 1;
    }
    const precision = val >= 100 ? 0 : val >= 10 ? 1 : 2;
    return `${val.toFixed(precision)} ${units[idx]}`;
}

function formatResolution(width, height) {
    if (!width || !height) return '?';
    const w = Number(width);
    const h = Number(height);
    if (!Number.isFinite(w) || !Number.isFinite(h)) return '?';
    let label = '';
    const maxDim = Math.max(w, h);
    if (maxDim >= 3840) label = '4K';
    else if (maxDim >= 2560) label = '1440p';
    else if (maxDim >= 1920) label = '1080p';
    else if (maxDim >= 1280) label = '720p';
    else label = `${maxDim}p`;
    return `${w} x ${h} (${label})`;
}

function formatAudio(channels, sampleRate) {
    const ch = Number(channels);
    const rate = Number(sampleRate);
    let channelLabel = '?';
    if (Number.isFinite(ch)) {
        if (ch === 1) channelLabel = 'Mono';
        else if (ch === 2) channelLabel = 'Stereo';
        else channelLabel = `${ch}ch`;
    }
    if (Number.isFinite(rate)) {
        return `${channelLabel} / ${(rate / 1000).toFixed(0)}kHz`;
    }
    return channelLabel;
}

let processingStageIndex = -1;
const processingStages = [
    /Validating video/i,
    /Video file validated/i,
    /Getting video duration/i,
    /Video duration/i,
    /Extracting audio/i,
    /Audio extracted/i,
    /Transcribing/i,
    /Transcription complete/i,
    /Detected language/i,
    /(Skipping speaker diarization|Speaker diarization)/i,
    /Initializing LLM/i,
    /Using LLM provider/i,
    /Analyzing transcript/i
];

function setProcessingProgress(percent) {
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    if (elements.processingPercent) {
        setText(elements.processingPercent, `${clamped}%`);
    }
    if (elements.processingCircle) {
        const circumference = Number(elements.processingCircle.dataset.circumference) || 282.7;
        const offset = circumference * (1 - clamped / 100);
        elements.processingCircle.style.strokeDashoffset = offset.toFixed(1);
    }
}

function updateProcessingProgressFromLine(line) {
    const steps = [
        { re: /Validating video/i, percent: 5 },
        { re: /Video file validated/i, percent: 10 },
        { re: /Getting video duration/i, percent: 15 },
        { re: /Video duration/i, percent: 20 },
        { re: /Extracting audio/i, percent: 30 },
        { re: /Audio extracted/i, percent: 40 },
        { re: /Transcribing/i, percent: 55 },
        { re: /Transcription complete/i, percent: 65 },
        { re: /Detected language/i, percent: 70 },
        { re: /(Skipping speaker diarization|Speaker diarization)/i, percent: 75 },
        { re: /Initializing LLM/i, percent: 80 },
        { re: /Using LLM provider/i, percent: 85 },
        { re: /Analyzing transcript/i, percent: 90 },
        { re: /Saving output/i, percent: 100 }
    ];

    for (const step of steps) {
        if (step.re.test(line)) {
            if (step.percent > (elements.processingPercent ? parseInt(elements.processingPercent.textContent, 10) || 0 : 0)) {
                setProcessingProgress(step.percent);
            }
            break;
        }
    }
}


async function updateSelectedVideoInfo(filePath) {
    if (!filePath) return;
    try {
        const info = await window.api.getVideoInfo(filePath);
        if (elements.selectedVideoThumbnail) {
            elements.selectedVideoThumbnail.src = info?.thumbnailDataUrl || 'file_upload.png';
        }
        if (elements.selectedVideoDuration) {
            setText(elements.selectedVideoDuration, formatDuration(info?.durationSec || 0));
        }
        if (elements.selectedVideoResolution) {
            setText(elements.selectedVideoResolution, formatResolution(info?.width, info?.height));
        }
        if (elements.selectedVideoSize) {
            setText(elements.selectedVideoSize, formatBytes(info?.sizeBytes));
        }
        if (elements.selectedVideoAudio) {
            setText(elements.selectedVideoAudio, formatAudio(info?.channels, info?.sampleRate));
        }
    } catch (err) {
        console.error('Failed to load video info:', err);
    }
}

// Update output preview
function updateOutputPreview() {
    if (!state.videoName) return;
    const baseName = sanitizeFilename(state.videoName.replace(/\.[^/.]+$/, ''));
    state.outputFileName = `${baseName}_${generateTimestamp()}.json`;
    setText(elements.outputFilePreview, state.outputFileName);
}

// File selection handler
async function handleFileSelect(filePath) {
    if (!filePath) return;

    state.videoPath = filePath;
    state.videoName = filePath.split(/[/\\]/).pop();

    setText(elements.fileName, state.videoName);
    setText(elements.filePath, filePath);

    if (!state.outputFolder) {
        state.outputFolder = await window.api.getDefaultOutputFolder();
    }
    setText(elements.outputFolderPreview, state.outputFolder);

    updateOutputPreview();
    await updateSelectedVideoInfo(filePath);
    showScreen('selected');
}

// Validate settings before convert
function validateSettings() {
    const provider = elements.llmProvider.value;

    // Check API key for cloud providers
    if (provider !== 'ollama' && !elements.apiKey.value.trim()) {
        alert(`Please enter your ${provider.charAt(0).toUpperCase() + provider.slice(1)} API key`);
        return false;
    }

    // Check HF token if diarization enabled
    if (elements.diarizationToggle.classList.contains('active') && !elements.hfToken.value.trim()) {
        alert('Please enter your HuggingFace token for speaker diarization');
        return false;
    }

    return true;
}

// Build options object for main process
function buildOptions() {
    const provider = elements.llmProvider.value;
    const diarizationEnabled = elements.diarizationToggle.classList.contains('active');

    const options = {
        videoPath: state.videoPath,
        outputPath: `${state.outputFolder}/${state.outputFileName}`,
        modelSize: elements.modelSize.value,
        skipDiarization: !diarizationEnabled
    };

    // Add provider-specific key (only if not Ollama)
    if (provider === 'openai') {
        options.openaiKey = elements.apiKey.value.trim();
    } else if (provider === 'anthropic') {
        options.anthropicKey = elements.apiKey.value.trim();
    } else if (provider === 'grok') {
        options.grokKey = elements.apiKey.value.trim();
    }

    // Add HF token if diarization enabled
    if (diarizationEnabled && elements.hfToken.value.trim()) {
        options.hfToken = elements.hfToken.value.trim();
    }

    // Add brand file if selected
    if (state.brandFile) {
        options.brandFile = state.brandFile;
    }

    return options;
}

// Parse stage from log line
function parseStage(line) {
    if (line.includes('Validating video')) return 'Validating video...';
    if (line.includes('Video file validated')) return 'Video file validated.';
    if (line.includes('Getting video duration')) return 'Getting video duration...';
    if (line.includes('Video duration')) return 'Video duration detected.';
    // Chunk progress messages
    const extractChunkMatch = line.match(/Extracting chunk (\d+)\/(\d+)/);
    if (extractChunkMatch) return `Extracting audio chunk ${extractChunkMatch[1]}/${extractChunkMatch[2]}...`;
    const transcribeChunkMatch = line.match(/Transcribing chunk (\d+)\/(\d+)/);
    if (transcribeChunkMatch) return `Transcribing chunk ${transcribeChunkMatch[1]}/${transcribeChunkMatch[2]}...`;
    if (line.includes('splitting into')) return 'Splitting video into chunks...';
    if (line.includes('Extracting audio')) return 'Extracting audio...';
    if (line.includes('Audio extracted')) return 'Audio extracted.';
    if (line.includes('Transcribing')) return 'Transcribing audio...';
    if (line.includes('Transcription complete')) return 'Transcription complete.';
    if (line.includes('Detected language')) return 'Language detected.';
    if (line.includes('Skipping speaker diarization')) return 'Skipping speaker diarization.';
    if (line.includes('diarization')) return 'Speaker diarization...';
    if (line.includes('Initializing LLM')) return 'Initializing LLM...';
    if (line.includes('Using LLM provider')) return 'LLM provider ready.';
    const llmChunkMatch = line.match(/Analyzing transcript chunk (\d+)\/(\d+)/);
    if (llmChunkMatch) return `Analyzing transcript chunk ${llmChunkMatch[1]}/${llmChunkMatch[2]}...`;
    if (line.includes('Analyzing transcript')) return 'Selecting shorts...';
    if (line.includes('Saving output')) return 'Saving output...';
    return null;
}

// Add log line with styling
function addLogLine(line) {
    state.logs.push(line);

    const stage = parseStage(line);
    if (stage) {
        setText(elements.processingStage, stage);
    }
    updateProcessingProgressFromLine(line);

    const logLine = document.createElement('div');
    logLine.textContent = line;

    if (/\b(success|completed|done)\b/i.test(line)) logLine.classList.add('log-success');
    else if (/\bwarning\b/i.test(line)) logLine.classList.add('log-warning');
    else if (/\b(error|failed)\b/i.test(line)) logLine.classList.add('log-error');
    else if (/\binfo\b/i.test(line)) logLine.classList.add('log-info');

    elements.logsContent.appendChild(logLine);
    elements.logsContent.parentElement.scrollTop = elements.logsContent.parentElement.scrollHeight;
}

// Render shorts list with clips
function renderShortsList(shorts, clips = []) {
    elements.shortsList.innerHTML = '';
    if (elements.shortsCountBadge) {
        setText(elements.shortsCountBadge, `${shorts.length} clips`);
    }

    const colors = ['#8B5CF6', '#F472B6', '#FBBF24', '#34D399'];

    shorts.forEach((short, index) => {
        const card = document.createElement('div');
        const accent = colors[index % colors.length];
        card.className = 'group bg-white rounded-2xl border-2 border-slate-200 overflow-hidden transition-all duration-200 flex flex-col h-full cursor-pointer';
        card.style.boxShadow = `4px 4px 0px 0px ${accent}`;
        card.onmouseenter = () => { card.style.transform = 'translate(-2px, -2px)'; card.style.boxShadow = `6px 6px 0px 0px ${accent}`; };
        card.onmouseleave = () => { card.style.transform = 'translate(0, 0)'; card.style.boxShadow = `4px 4px 0px 0px ${accent}`; };

        const startTime = Number(short.start_time || 0);
        const endTime = Number(short.end_time || 0);
        const duration = Math.max(0, endTime - startTime);

        const clip = clips.find(c => c.title === short.title);
        const hasClip = clip && clip.success && clip.clipPath;

        const formatTime = (secs) => {
            const total = Math.max(0, Math.floor(secs));
            const minutes = Math.floor(total / 60);
            const seconds = total % 60;
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
        };

        const score = short.score ?? short.viral_score ?? null;
        const scoreBadge = score ? `
            <div class="absolute top-3 left-3 px-2.5 py-1 rounded-full text-xs font-bold bg-[#F472B6] text-white border-2 border-slate-800 shadow-[2px_2px_0px_0px_#1E293B] flex items-center gap-1">
                <span class="material-icons-round text-sm">bolt</span>
                ${score}/100
            </div>
        ` : '';

        card.innerHTML = `
            <div class="relative aspect-video bg-slate-100 overflow-hidden border-b-2 border-slate-200">
                <img class="clip-thumb absolute inset-0 w-full h-full object-cover" src="file_upload.png" alt="Clip thumbnail" />
                <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
                <div class="absolute bottom-3 right-3 px-2.5 py-1 bg-white border-2 border-slate-800 rounded-full text-xs font-bold text-slate-800 shadow-[2px_2px_0px_0px_#1E293B]">
                    ${formatTime(duration)}
                </div>
                ${scoreBadge}
                <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-200">
                    ${hasClip ? `<button class="w-14 h-14 rounded-full bg-white border-2 border-slate-800 text-[#8B5CF6] flex items-center justify-center shadow-[3px_3px_0px_0px_#1E293B] transform scale-75 group-hover:scale-100 transition-transform duration-200 play-clip-btn" data-clip="${clip.clipPath}" data-title="${short.title}">
                        <span class="material-icons-round text-3xl ml-1">play_arrow</span>
                    </button>` : `<span class="text-xs text-white bg-slate-800 border-2 border-slate-600 px-3 py-1.5 rounded-full font-bold">Clipping...</span>`}
                </div>
            </div>
            <div class="p-5 flex flex-col flex-grow">
                <h3 class="font-extrabold text-base text-slate-800 leading-tight line-clamp-2 mb-1" contenteditable="true">${short.title}</h3>
                ${short.reason ? `<p class="text-xs text-slate-500 mb-4 leading-relaxed">${short.reason}</p>` : ''}
                <div class="mt-auto flex items-center gap-2 pt-4 border-t-2 border-slate-100">
                    ${hasClip ? `<button class="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-[#8B5CF6] text-white rounded-full text-sm font-bold border-2 border-slate-800 shadow-[2px_2px_0px_0px_#1E293B] hover:shadow-[3px_3px_0px_0px_#1E293B] hover:translate-x-[-1px] hover:translate-y-[-1px] transition-all duration-200 play-clip-btn" data-clip="${clip.clipPath}" data-title="${short.title}"><span class="material-icons-round text-sm">play_arrow</span>Preview</button>` : `<span class="text-xs text-slate-400 font-bold">Preparing preview...</span>`}
                </div>
            </div>
        `;

        elements.shortsList.appendChild(card);

        if (hasClip) {
            const thumb = card.querySelector('.clip-thumb');
            if (thumb) {
                window.api.getFileThumbnail(clip.clipPath).then(src => {
                    if (src) thumb.src = src;
                }).catch(err => {
                    console.error('Failed to load clip thumbnail:', err);
                });
            }
        }
    });
}

// Open clip preview modal and launch default OS player
async function openClipPreview(clipPath, title) {
    setText(elements.videoModalTitle, title);
    elements.videoModal.classList.add('open');

    if (elements.clipThumbnail) {
        elements.clipThumbnail.removeAttribute('src');
        const fallback = 'file_upload.png';
        try {
            const thumb = await window.api.getFileThumbnail(clipPath);
            if (thumb) {
                elements.clipThumbnail.src = thumb;
            } else {
                elements.clipThumbnail.src = fallback;
            }
        } catch (err) {
            console.error('Thumbnail load failed:', err);
            elements.clipThumbnail.src = fallback;
        }
    }

    try {
        await window.api.openFile(clipPath);
    } catch (err) {
        console.error('Failed to open OS player:', err);
    }
}

// Close video modal with proper cleanup
function closeVideoModal() {
    if (elements.clipThumbnail) {
        elements.clipThumbnail.removeAttribute('src');
    }

    elements.videoModal.classList.remove('open');
}

// Handle completion - clip videos and render
async function handleComplete(data) {
    state.resultJson = data.resultJson;

    // Update completed screen
    setText(elements.completedOutputPath, data.outputPath);

    // Store raw logs
    setText(elements.rawLogsContent, state.logs.join("\n"));

    // Render shorts initially (without clips)
    if (data.resultJson && data.resultJson.shorts) {
        renderShortsList(data.resultJson.shorts, []);
    }

    showScreen('completed');

    // Start clipping videos in background
    if (data.resultJson && data.resultJson.shorts && state.videoPath) {
        try {
            const clipResult = await window.api.clipVideos({
                videoPath: state.videoPath,
                outputDir: state.outputFolder,
                shorts: data.resultJson.shorts,
                portraitCrop: elements.portraitCropToggle?.classList.contains('active') || false
            });

            state.clipsFolder = clipResult.clipsFolder;
            state.clips = clipResult.clips;

            // Re-render with clips
            renderShortsList(data.resultJson.shorts, clipResult.clips);
        } catch (error) {
            console.error('Failed to clip videos:', error);
        }
    }
}

// Reset for new conversion
function resetState() {
    state.videoPath = null;
    state.videoName = null;
    state.brandFile = null;
    state.logs = [];
    state.resultJson = null;
    state.clipsFolder = null;
    state.clips = [];

    elements.logsContent.innerHTML = '';
    setText(elements.brandFilePreview, "None");
    setText(elements.processingStage, "Initializing...");
    elements.rawLogsPanel.classList.add('hidden');
    elements.rawLogsToggle.classList.remove('open');
    closeVideoModal();
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Drop zone - click
    elements.dropZone.addEventListener('click', async (e) => {
        // Don't trigger if clicking the button (it has its own handler)
        if (e.target.id === 'selectFileBtn') return;
        const filePath = await window.api.selectVideo();
        handleFileSelect(filePath);
    });

    // Select File button
    const selectFileBtn = document.getElementById('selectFileBtn');
    if (selectFileBtn) {
        selectFileBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const filePath = await window.api.selectVideo();
            handleFileSelect(filePath);
        });
    }

    // Drop zone - drag & drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('drag-over');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('drag-over');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file && /\.(mp4|mov|mkv|avi|webm)$/i.test(file.name)) {
            handleFileSelect(file.path);
        }
    });

    if (elements.selectedVideoPreview) {
        elements.selectedVideoPreview.addEventListener('click', () => {
            if (state.videoPath) window.api.openFile(state.videoPath);
        });
    }

    // Change file button
    buttons.changeFile.addEventListener('click', async () => {
        const filePath = await window.api.selectVideo();
        if (filePath) handleFileSelect(filePath);
    });

    // Change output folder button
    buttons.changeOutput.addEventListener('click', async () => {
        const folder = await window.api.selectOutputFolder();
        if (folder) {
            state.outputFolder = folder;
            setText(elements.outputFolderPreview, folder);
        }
    });

    // Select brand file button
    buttons.selectBrand.addEventListener('click', async () => {
        const filePath = await window.api.selectBrandFile();
        if (filePath) {
            state.brandFile = filePath;
            setText(elements.brandFilePreview, filePath.split(/[/\\]/).pop());
        }
    });

    // LLM Provider change
    elements.llmProvider.addEventListener('change', () => {
        const provider = elements.llmProvider.value;

        if (provider === 'ollama') {
            elements.apiKeyRow.classList.add('hidden');
        } else {
            elements.apiKeyRow.classList.remove('hidden');
            setText(elements.apiKeyLabel, `${provider.charAt(0).toUpperCase() + provider.slice(1)} API Key`);
            elements.apiKey.value = '';
        }
    });
    // LLM provider cards (UI buttons)
    const llmCards = Array.from(document.querySelectorAll('.llm-card'));
    const setProvider = (provider) => {
        if (!elements.llmProvider) return;
        elements.llmProvider.value = provider;
        elements.llmProvider.dispatchEvent(new Event('change'));
        llmCards.forEach(card => {
            card.dataset.active = card.dataset.provider === provider ? 'true' : 'false';
        });
    };

    if (llmCards.length) {
        llmCards.forEach(card => {
            card.addEventListener('click', () => {
                const provider = card.dataset.provider;
                if (provider) setProvider(provider);
            });
        });
        setProvider(elements.llmProvider.value);
    }

    // Diarization toggle
    elements.diarizationToggle.addEventListener('click', () => {
        elements.diarizationToggle.classList.toggle('active');

        if (elements.diarizationToggle.classList.contains('active')) {
            elements.hfTokenRow.classList.remove('hidden');
        } else {
            elements.hfTokenRow.classList.add('hidden');
        }
    });

    // Auto-open toggle
    elements.autoOpenToggle.addEventListener('click', () => {
        elements.autoOpenToggle.classList.toggle('active');
    });

    // Portrait crop toggle
    if (elements.portraitCropToggle) {
        elements.portraitCropToggle.addEventListener('click', () => {
            elements.portraitCropToggle.classList.toggle('active');
        });
    }

    // Advanced toggle
    elements.advancedToggle.addEventListener('click', () => {
        elements.advancedToggle.classList.toggle('open');
        elements.advancedContent.classList.toggle('open');
    });

    // Back button
    buttons.back.addEventListener('click', () => {
        resetState();
        showScreen('idle');
    });

    // Convert button
    buttons.convert.addEventListener('click', async () => {
        if (!validateSettings()) return;

        showScreen('processing');
        const options = buildOptions();
        await window.api.startProcessing(options);
    });

    // Cancel button
    buttons.cancel.addEventListener('click', async () => {
        await window.api.cancelProcessing();
        showScreen('selected');
    });

    // Open folder button
    buttons.openFolder.addEventListener('click', () => {
        window.api.openFolder(state.outputFolder);
    });

    // Open JSON button
    buttons.openJson.addEventListener('click', () => {
        window.api.openFile(`${state.outputFolder}/${state.outputFileName}`);
    });

    // Convert another button
    buttons.convertAnother.addEventListener('click', () => {
        resetState();
        showScreen('idle');
    });

    // Raw logs toggle
    elements.rawLogsToggle.addEventListener('click', () => {
        elements.rawLogsToggle.classList.toggle('open');
        elements.rawLogsPanel.classList.toggle('hidden');
    });

    // Video modal - close button
    elements.videoModalClose.addEventListener('click', closeVideoModal);

    // Video modal - click overlay to close
    elements.videoModalOverlay.addEventListener('click', closeVideoModal);

    // Video modal - play clip buttons (event delegation)
    elements.shortsList.addEventListener('click', (e) => {
        const playBtn = e.target.closest('.play-clip-btn');
        if (playBtn) {
            const clipPath = playBtn.dataset.clip;
            const title = playBtn.dataset.title;
            if (clipPath) {
                // Open in default OS player instead of embedded player
                openClipPreview(clipPath, title);
            }
        }
    });

    // Escape key to close video modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.videoModal.classList.contains('open')) {
            closeVideoModal();
        }
    });

    // Listen for process events
    window.api.onProcessOutput((line) => {
        addLogLine(line);
    });

    window.api.onProcessComplete((data) => {
        handleComplete(data);
    });

    window.api.onProcessError((error) => {
        addLogLine(`Error: ${error}`);
        alert(`Processing failed: ${error}`);
        showScreen('selected');
    });
});
