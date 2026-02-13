/**
 * Shawty - Electron Main Process
 * Handles window creation, IPC handlers, and Python process spawning
 */

const { app, BrowserWindow, ipcMain, dialog, shell, protocol, nativeImage } = require('electron');
const path = require('path');
const fs = require('fs');
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

ipcMain.handle('process:clipVideos', async (_event, { videoPath, outputDir, shorts }) => {
    const videoBaseName = path.basename(videoPath, path.extname(videoPath));
    const clipsFolder = path.join(outputDir, `${sanitizeFilename(videoBaseName)}_clips`);

    // Create clips folder
    if (!fs.existsSync(clipsFolder)) {
        fs.mkdirSync(clipsFolder, { recursive: true });
    }

    const results = [];

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
                const ffmpeg = spawn('ffmpeg', [
                    '-y',                              // Overwrite output
                    '-ss', short.start_time.toString(), // Start time (before -i for fast seek)
                    '-i', videoPath,                   // Input file
                    '-t', duration.toString(),         // Duration
                    '-c', 'copy',                      // Stream copy (fast, cuts at nearest keyframe)
                    '-avoid_negative_ts', 'make_zero', // Fix timestamp issues
                    clipPath                           // Output file
                ]);

                let stderrOutput = '';
                ffmpeg.stderr.on('data', (data) => {
                    stderrOutput += data.toString();
                });

                ffmpeg.on('close', (code) => {
                    if (code === 0) {
                        // Validate file size (Reason #2: 0-Byte File check)
                        try {
                            const { size } = fs.statSync(clipPath);
                            if (size < 1000) { // Less than 1KB
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
