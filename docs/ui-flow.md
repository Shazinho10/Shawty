# Shawty UI Flow (High Level)

This document describes the end-to-end UI flow at a high level, screen by screen.

## 1) Idle (Upload)

**Purpose**
- Entry point to start a new clip generation.

**User Actions**
- Drag-and-drop a video file into the upload zone.
- Click to browse and select a video file.

**System Response**
- A valid file selection advances the app to the **Selected** screen.

## 2) Selected (Setup)

**Purpose**
- Configure output location and processing settings before starting.

**User Actions**
- Change the selected video.
- Choose output folder.
- Select Whisper model size.
- Choose LLM provider and set API key if required.
- Enable speaker diarization (and add a token when needed).
- Open advanced options (brand file, auto-open behavior).
- Click **Convert** to start processing.
- Click **Back** to return to **Idle**.

**System Response**
- Input validation runs before conversion starts.
- If valid, transition to **Processing**.

## 3) Processing (In Progress)

**Purpose**
- Show progress and logs while the backend runs.

**User Actions**
- Monitor status and logs.
- Click **Cancel** to stop the process.

**System Response**
- On success, transition to **Completed**.
- On error or cancel, return to **Selected**.

## 4) Completed (Results)

**Purpose**
- Display output path and generated shorts list.

**User Actions**
- Open output folder.
- Open output JSON.
- Start a new conversion.

**System Response**
- Clicking **Convert Another** resets state and returns to **Idle**.

## 5) Clip Modal (Preview)

**Purpose**
- Present clip details and playback UI.

**User Actions**
- Open a clip from the shorts list.
- Close via close button, overlay click, or Escape key.

**System Response**
- Modal opens with the selected clip details.
- Closing returns to the **Completed** screen state.
