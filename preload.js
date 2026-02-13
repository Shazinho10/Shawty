/**
 * Shawty - Preload Script
 * Exposes safe APIs to renderer via contextBridge
 */

const { contextBridge, ipcRenderer } = require('electron');

let outputCleanup = null;
let completeCleanup = null;
let errorCleanup = null;
let clipProgressCleanup = null;

contextBridge.exposeInMainWorld('api', {
    selectVideo: () => ipcRenderer.invoke('dialog:selectVideo'),
    selectOutputFolder: () => ipcRenderer.invoke('dialog:selectOutputFolder'),
    selectBrandFile: () => ipcRenderer.invoke('dialog:selectBrandFile'),
    getDefaultOutputFolder: () => ipcRenderer.invoke('dialog:getDefaultOutputFolder'),
    startProcessing: (options) => ipcRenderer.invoke('process:start', options),
    cancelProcessing: () => ipcRenderer.invoke('process:cancel'),
    openFolder: (path) => ipcRenderer.invoke('fs:openFolder', path),
    openFile: (path) => ipcRenderer.invoke('fs:openFile', path),
    getFileThumbnail: (path) => ipcRenderer.invoke('fs:getThumbnail', path),
    getVideoInfo: (path) => ipcRenderer.invoke('fs:getVideoInfo', path),

    // Video clipping
    clipVideos: (options) => ipcRenderer.invoke('process:clipVideos', options),

    // Convert file path to media:// URL for secure local video playback
    getMediaUrl: (filePath) => {
        // Use forward slashes and encode only special characters (not slashes or colons)
        const normalized = filePath.replace(/\\/g, '/');
        // Only encode spaces and other problematic chars, keep path structure intact
        const encoded = normalized.split('/').map(part => encodeURIComponent(part)).join('/');
        return `media://${encoded}`;
    },

    onProcessOutput: (callback) => {
        if (outputCleanup) outputCleanup();

        const handler = (_event, line) => callback(line);
        ipcRenderer.on('process:log', handler);

        outputCleanup = () => ipcRenderer.removeListener('process:log', handler);
        return outputCleanup;
    },

    onProcessComplete: (callback) => {
        if (completeCleanup) completeCleanup();

        const handler = (_event, data) => callback(data);
        ipcRenderer.on('process:complete', handler);

        completeCleanup = () => ipcRenderer.removeListener('process:complete', handler);
        return completeCleanup;
    },

    onProcessError: (callback) => {
        if (errorCleanup) errorCleanup();

        const handler = (_event, error) => callback(error);
        ipcRenderer.on('process:error', handler);

        errorCleanup = () => ipcRenderer.removeListener('process:error', handler);
        return errorCleanup;
    },

    onClipProgress: (callback) => {
        if (clipProgressCleanup) clipProgressCleanup();

        const handler = (_event, progress) => callback(progress);
        ipcRenderer.on('clip:progress', handler);

        clipProgressCleanup = () => ipcRenderer.removeListener('clip:progress', handler);
        return clipProgressCleanup;
    }
});
