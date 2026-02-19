/**
 * Shawty - Electron Main Process
 * Handles window creation, IPC handlers, and Python process spawning
 */

const { app, BrowserWindow, ipcMain, dialog, shell, protocol, nativeImage } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess = null;

// ================================================
// CUSTOM PROTOCOL REGISTRATION - Must be before app.ready
// ================================================

protocol.registerSchemesAsPrivileged([{
    scheme: 'media',
    privileges: {
        stream: true,
        bypassCSP: true,
        supportFetchAPI: true,
        secure: true,
        corsEnabled: true
    }
}]);

// Helper: Convert Node stream to Web ReadableStream (robust version)
function nodeStreamToWeb(nodeStream) {
    let isStreamClosed = false;

    return new ReadableStream({
        start(controller) {
            nodeStream.on('data', (chunk) => {
                if (!isStreamClosed) {
                    try {
                        controller.enqueue(chunk);
                    } catch (e) {
                        // Controller might be closed, ignore
                    }
                }
            });

            nodeStream.on('end', () => {
                if (!isStreamClosed) {
                    isStreamClosed = true;
                    try {
                        controller.close();
                    } catch (e) {
                        // Already closed, ignore
                    }
                }
            });

            nodeStream.on('error', (err) => {
                if (!isStreamClosed) {
                    isStreamClosed = true;
                    try {
                        controller.error(err);
                    } catch (e) {
                        // Already errored/closed, ignore
                    }
                }
            });
        },
        cancel() {
            isStreamClosed = true;
            try {
                nodeStream.destroy();
            } catch (e) {
                // Already destroyed, ignore
            }
        }
    });
}

// Helper: Get MIME type from extension
function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo'
    };
    return mimeTypes[ext] || 'video/mp4';
}

function runFfprobe(args) {
    return new Promise((resolve, reject) => {
        const proc = spawn('ffprobe', args);
        let out = '';
        let err = '';

        proc.stdout.on('data', (data) => {
            out += data.toString();
        });
        proc.stderr.on('data', (data) => {
            err += data.toString();
        });
        proc.on('close', (code) => {
            if (code === 0) {
                resolve(out);
            } else {
                reject(new Error(err || `ffprobe exited with code ${code}`));
            }
        });
        proc.on('error', (error) => reject(error));
    });
}

function ensureEven(value) {
    const num = Math.floor(Number(value) || 0);
    return num % 2 === 0 ? num : num - 1;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

async function getVideoProbe(filePath) {
    const output = await runFfprobe([
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration:stream=width,height',
        '-of', 'json',
        filePath
    ]);
    const json = JSON.parse(output || '{}');
    const durationSec = json.format && json.format.duration ? Number(json.format.duration) : null;
    const stream = Array.isArray(json.streams) ? json.streams[0] : null;
    const width = stream && stream.width ? Number(stream.width) : null;
    const height = stream && stream.height ? Number(stream.height) : null;
    return { durationSec, width, height };
}

function runFfmpeg(args) {
    return new Promise((resolve, reject) => {
        const ffmpeg = spawn('ffmpeg', args);
        let stderrOutput = '';
        ffmpeg.stderr.on('data', (data) => {
            stderrOutput += data.toString();
        });
        ffmpeg.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(stderrOutput || `ffmpeg exited with code ${code}`));
            }
        });
        ffmpeg.on('error', (err) => {
            reject(err);
        });
    });
}

async function extractSampleFrames(videoPath, durationSec) {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'shawty-frames-'));
    const framePaths = [];
    const hasDuration = Number.isFinite(durationSec) && durationSec > 0;
    const samplePoints = hasDuration ? [0.25, 0.5, 0.75].map(p => durationSec * p) : [0];

    for (let i = 0; i < samplePoints.length; i++) {
        const timeSec = Math.max(0, samplePoints[i]);
        const framePath = path.join(tempDir, `frame_${i + 1}.jpg`);
        await runFfmpeg([
            '-y',
            '-ss', timeSec.toString(),
            '-i', videoPath,
            '-frames:v', '1',
            '-q:v', '2',
            framePath
        ]);
        framePaths.push(framePath);
    }

    return { tempDir, framePaths };
}

let faceModelPromise = null;
async function loadFaceModel() {
    if (faceModelPromise) return faceModelPromise;
    faceModelPromise = (async () => {
        const tf = require('@tensorflow/tfjs-node');
        const blazeface = require('@tensorflow-models/blazeface');
        const model = await blazeface.load();
        return { tf, model };
    })();
    return faceModelPromise;
}

async function detectAverageCenterX(framePaths) {
    const { tf, model } = await loadFaceModel();
    let totalCenterX = 0;
    let validDetections = 0;

    for (const framePath of framePaths) {
        const imageBuffer = fs.readFileSync(framePath);
        const tensor = tf.node.decodeImage(imageBuffer, 3);

        const predictions = await model.estimateFaces(tensor, false);
        if (predictions && predictions.length > 0) {
            let best = predictions[0];
            let bestArea = 0;
            for (const pred of predictions) {
                const tl = pred.topLeft;
                const br = pred.bottomRight;
                const w = br[0] - tl[0];
                const h = br[1] - tl[1];
                const area = w * h;
                if (area > bestArea) {
                    bestArea = area;
                    best = pred;
                }
            }
            const tl = best.topLeft;
            const br = best.bottomRight;
            const faceWidth = br[0] - tl[0];
            const centerX = tl[0] + (faceWidth / 2);
            if (Number.isFinite(centerX)) {
                totalCenterX += centerX;
                validDetections += 1;
            }
        }

        tf.dispose(tensor);
    }

    if (validDetections === 0) return null;
    return totalCenterX / validDetections;
}

async function buildPortraitCropPlan(videoPath) {
    const meta = await getVideoProbe(videoPath);
    if (!meta.width || !meta.height) {
        throw new Error('Unable to read video dimensions for portrait crop.');
    }

    const width = meta.width;
    const height = meta.height;
    const cropHeight = ensureEven(height);
    let cropWidth = ensureEven(Math.round(cropHeight * 9 / 16));
    if (cropWidth <= 0) cropWidth = ensureEven(width);
    if (cropWidth > width) cropWidth = ensureEven(width);

    let centerX = width / 2;
    let usedAi = false;
    let tempDir = null;

    try {
        const sample = await extractSampleFrames(videoPath, meta.durationSec);
        tempDir = sample.tempDir;
        const aiCenterX = await detectAverageCenterX(sample.framePaths);
        if (Number.isFinite(aiCenterX)) {
            centerX = aiCenterX;
            usedAi = true;
        }
    } catch (err) {
        console.warn('[portrait] Face detection failed, using center crop:', err.message || err);
    } finally {
        if (tempDir) {
            fs.rmSync(tempDir, { recursive: true, force: true });
        }
    }

    const cropX = clamp(Math.floor(centerX - (cropWidth / 2)), 0, Math.max(0, width - cropWidth));
    const needsCrop = cropWidth < width;

    return {
        cropWidth,
        cropHeight,
        cropX,
        needsCrop,
        usedAi
    };
}

async function getVideoInfo(filePath) {
    const info = {
        sizeBytes: null,
        durationSec: null,
        width: null,
        height: null,
        channels: null,
        sampleRate: null,
        thumbnailDataUrl: null
    };

    try {
        const stat = fs.statSync(filePath);
        info.sizeBytes = stat.size;
    } catch (err) {
        console.error('[videoInfo] Failed to read file size:', err);
    }

    try {
        const output = await runFfprobe([
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-show_streams',
            '-of', 'json',
            filePath
        ]);
        const json = JSON.parse(output || '{}');
        if (json.format && json.format.duration) {
            info.durationSec = Number(json.format.duration);
        }
        if (Array.isArray(json.streams)) {
            const videoStream = json.streams.find(s => s.codec_type === 'video');
            if (videoStream) {
                info.width = videoStream.width || null;
                info.height = videoStream.height || null;
            }
            const audioStream = json.streams.find(s => s.codec_type === 'audio');
            if (audioStream) {
                info.channels = audioStream.channels || null;
                info.sampleRate = audioStream.sample_rate ? Number(audioStream.sample_rate) : null;
            }
        }
    } catch (err) {
        console.error('[videoInfo] ffprobe failed:', err);
    }

    try {
        const image = await nativeImage.createThumbnailFromPath(filePath, { width: 1280, height: 720 });
        if (image && !image.isEmpty()) {
            info.thumbnailDataUrl = image.toDataURL();
        }
    } catch (err) {
        console.error('[videoInfo] Thumbnail creation failed:', err);
    }

    return info;
}

// Helper: Sanitize filename for Windows (Aggressive)
function sanitizeFilename(name) {
    // Replace ANY char that is not a letter, number, or space with nothing
    // Then replace spaces with underscores to be extra safe
    return name.replace(/[^a-zA-Z0-9 ]/g, "").replace(/\s+/g, "_");
}

// Create the main browser window
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 900,
        height: 750,
        minWidth: 700,
        minHeight: 600,
        autoHideMenuBar: true,
        backgroundColor: '#0a0a0f',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: true,
            preload: path.join(__dirname, 'preload.js')
        },
        titleBarStyle: 'default',
        show: false
    });

    mainWindow.loadFile('index.html');

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
        if (pythonProcess) {
            pythonProcess.kill();
            pythonProcess = null;
        }
    });
}

app.whenReady().then(() => {
    // Register media:// protocol handler
    protocol.handle('media', async (request) => {
        try {
            // Parse file path from URL
            let filePath = request.url.replace('media://', '');

            // Decode each path part (spaces, special chars) but keep structure
            filePath = filePath.split('/').map(part => decodeURIComponent(part)).join('/');

            // Normalize path separators for Windows
            filePath = filePath.replace(/\//g, path.sep);

            // Log for debugging
            console.log('[media://] Requested:', filePath);
            const exists = fs.existsSync(filePath);
            console.log('[media://] Exists?', exists);

            // Check if file exists
            if (!exists) {
                console.error('[media://] File not found:', filePath);
                return new Response('File not found', { status: 404 });
            }

            const stat = fs.statSync(filePath);
            const mimeType = getMimeType(filePath);
            const rangeHeader = request.headers.get('range');

            if (rangeHeader) {
                // Handle Range request for seeking (HTTP 206 Partial Content)
                const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
                if (match) {
                    const start = parseInt(match[1], 10);
                    const end = match[2] ? parseInt(match[2], 10) : stat.size - 1;
                    const chunkSize = end - start + 1;

                    const stream = fs.createReadStream(filePath, { start, end });

                    return new Response(nodeStreamToWeb(stream), {
                        status: 206,
                        headers: {
                            'Content-Type': mimeType,
                            'Content-Length': chunkSize.toString(),
                            'Content-Range': `bytes ${start}-${end}/${stat.size}`,
                            'Accept-Ranges': 'bytes'
                        }
                    });
                }
            }

            // Full file response (HTTP 200)
            const stream = fs.createReadStream(filePath);
            return new Response(nodeStreamToWeb(stream), {
                status: 200,
                headers: {
                    'Content-Type': mimeType,
                    'Content-Length': stat.size.toString(),
                    'Accept-Ranges': 'bytes'
                }
            });
        } catch (error) {
            console.error('Protocol handler error:', error);
            return new Response('Internal error', { status: 500 });
        }
    });

    createWindow();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ================================================
// IPC HANDLERS - Dialogs
// ================================================

ipcMain.handle('dialog:selectVideo', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Video File',
        filters: [
            { name: 'Video Files', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm'] }
        ],
        properties: ['openFile']
    });

    return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('dialog:selectOutputFolder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Output Folder',
        properties: ['openDirectory', 'createDirectory']
    });

    return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('dialog:selectBrandFile', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Brand JSON File',
        filters: [
            { name: 'JSON Files', extensions: ['json'] }
        ],
        properties: ['openFile']
    });

    return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('dialog:getDefaultOutputFolder', async () => {
    const downloadsPath = app.getPath('downloads');
    const outputFolder = path.join(downloadsPath, 'clips_json');

    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder, { recursive: true });
    }

    return outputFolder;
});

// ================================================
// IPC HANDLERS - Process Management
// ================================================

ipcMain.handle('process:start', async (_event, options) => {
    const {
        videoPath,
        outputPath,
        modelSize,
        skipDiarization,
        openaiKey,
        anthropicKey,
        grokKey,
        hfToken,
        brandFile
    } = options;

    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const args = ['-m', 'src.main', videoPath];

    args.push('--output', outputPath);
    args.push('--model-size', modelSize || 'base');

    if (skipDiarization) {
        args.push('--skip-diarization');
    }

    if (openaiKey) {
        args.push('--openai-key', openaiKey);
    } else if (anthropicKey) {
        args.push('--anthropic-key', anthropicKey);
    } else if (grokKey) {
        args.push('--grok-key', grokKey);
    }

    if (hfToken) {
        args.push('--hf-token', hfToken);
    }

    if (brandFile) {
        args.push('--brand-file', brandFile);
    }

    const venvPython = process.platform === 'win32'
        ? path.join(__dirname, 'venv', 'Scripts', 'python.exe')
        : path.join(__dirname, 'venv', 'bin', 'python');

    const pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python';

    // Spawn Python process
    pythonProcess = spawn(pythonExecutable, args, {
        cwd: __dirname,
        env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            ...((!openaiKey && !anthropicKey && !grokKey) && {
                OPENAI_API_KEY: '',
                ANTHROPIC_API_KEY: '',
                GROK_API_KEY: ''
            })
        }
    });

    mainWindow.setProgressBar(2);

    pythonProcess.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(line => line.trim());
        lines.forEach(line => {
            mainWindow.webContents.send('process:log', line);
        });
    });

    pythonProcess.stderr.on('data', (data) => {
        const lines = data.toString().split('\n').filter(line => line.trim());
        lines.forEach(line => {
            mainWindow.webContents.send('process:log', line);
        });
    });

    pythonProcess.on('close', (code) => {
        mainWindow.setProgressBar(-1);

        if (code === 0) {
            let resultJson = null;
            try {
                const jsonContent = fs.readFileSync(outputPath, 'utf8');
                resultJson = JSON.parse(jsonContent);
            } catch (err) {
                console.error('Failed to read output JSON:', err);
            }

            mainWindow.webContents.send('process:complete', {
                exitCode: code,
                outputPath: outputPath,
                resultJson: resultJson
            });
        } else if (code !== null) {
            mainWindow.webContents.send('process:error', `Process exited with code ${code}`);
        }

        pythonProcess = null;
    });

    pythonProcess.on('error', (err) => {
        mainWindow.setProgressBar(-1);
        mainWindow.webContents.send('process:error', err.message);
        pythonProcess = null;
    });

    return true;
});

ipcMain.handle('process:cancel', async () => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
        mainWindow.setProgressBar(-1);
        return true;
    }
    return false;
});

// ================================================
// IPC HANDLERS - File System
// ================================================

ipcMain.handle('fs:openFolder', async (_event, folderPath) => {
    shell.openPath(folderPath);
    return true;
});

ipcMain.handle('fs:openFile', async (_event, filePath) => {
    shell.openPath(filePath);
    return true;
});

ipcMain.handle('fs:getThumbnail', async (_event, filePath) => {
    try {
        const image = await nativeImage.createThumbnailFromPath(filePath, { width: 640, height: 360 });
        if (image && !image.isEmpty()) {
            return image.toDataURL();
        }
    } catch (err) {
        console.error('[thumbnail] Failed to create thumbnail:', err);
    }
    return null;
});

ipcMain.handle('fs:getVideoInfo', async (_event, filePath) => {
    return getVideoInfo(filePath);
});

// ================================================
// IPC HANDLERS - Video Clipping
// ================================================

ipcMain.handle('process:clipVideos', async (_event, { videoPath, outputDir, shorts, portraitCrop }) => {
    const videoBaseName = path.basename(videoPath, path.extname(videoPath));
    const clipsFolder = path.join(outputDir, `${sanitizeFilename(videoBaseName)}_clips`);

    // Create clips folder
    if (!fs.existsSync(clipsFolder)) {
        fs.mkdirSync(clipsFolder, { recursive: true });
    }

    const results = [];
    let cropPlan = null;

    if (portraitCrop) {
        try {
            cropPlan = await buildPortraitCropPlan(videoPath);
            console.log('[portrait] Crop plan:', cropPlan);
        } catch (err) {
            console.error('[portrait] Failed to build crop plan:', err);
            cropPlan = null;
        }
    }

    for (let i = 0; i < shorts.length; i++) {
        const short = shorts[i];
        const sanitizedTitle = sanitizeFilename(short.title);
        const clipPath = path.join(clipsFolder, `${i + 1}_${sanitizedTitle}.mp4`);
        const duration = short.end_time - short.start_time;

        // Send progress update
        mainWindow.webContents.send('clip:progress', {
            current: i + 1,
            total: shorts.length,
            title: short.title
        });

        try {
            await new Promise((resolve, reject) => {
                const args = [
                    '-y',
                    '-ss', short.start_time.toString(),
                    '-i', videoPath,
                    '-t', duration.toString()
                ];

                if (portraitCrop && cropPlan && cropPlan.needsCrop) {
                    args.push(
                        '-vf', `crop=${cropPlan.cropWidth}:${cropPlan.cropHeight}:${cropPlan.cropX}:0`,
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac'
                    );
                } else {
                    args.push('-c', 'copy');
                }

                args.push(
                    '-avoid_negative_ts', 'make_zero',
                    clipPath
                );

                const ffmpeg = spawn('ffmpeg', args);

                let stderrOutput = '';
                ffmpeg.stderr.on('data', (data) => {
                    stderrOutput += data.toString();
                });

                ffmpeg.on('close', (code) => {
                    if (code === 0) {
                        try {
                            const { size } = fs.statSync(clipPath);
                            if (size < 1000) {
                                console.error(`[FFmpeg] Clip failed (File too small): ${clipPath}`);
                                reject(new Error('Clip file too small (possible FFmpeg silent failure)'));
                                return;
                            }
                            console.log(`[FFmpeg] Clip created: ${clipPath} (${size} bytes)`);
                            resolve();
                        } catch (err) {
                            console.error(`[FFmpeg] Failed to stat output file:`, err);
                            reject(err);
                        }
                    } else {
                        console.error(`[FFmpeg] Error:`, stderrOutput);
                        reject(new Error(`FFmpeg exited with code ${code}`));
                    }
                });

                ffmpeg.on('error', (err) => {
                    reject(err);
                });
            });

            results.push({
                title: short.title,
                clipPath: clipPath,
                success: true
            });
        } catch (error) {
            console.error(`Failed to clip "${short.title}":`, error);
            results.push({
                title: short.title,
                clipPath: null,
                success: false,
                error: error.message
            });
        }
    }

    return {
        clipsFolder,
        clips: results
    };
});
