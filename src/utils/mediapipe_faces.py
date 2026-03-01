import argparse
import json
import os
import sys


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe face analysis for cropping.")
    parser.add_argument("--frames", nargs="+", required=True, help="Paths to image frames.")
    parser.add_argument(
        "--active-speaker",
        action="store_true",
        help="Enable active speaker heuristic (mouth motion across face tracks).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log(f"[mediapipe] python={sys.executable}")
    try:
        import mediapipe as mp
    except Exception as exc:
        log(f"[mediapipe] import failed: {exc}")
        print(json.dumps({"ok": False, "error": f"mediapipe import failed: {exc}"}))
        return 2

    log(f"[mediapipe] version={getattr(mp, '__version__', 'unknown')}")
    if not hasattr(mp, "solutions"):
        msg = (
            "mediapipe.solutions not available in this build. "
            "Install a solutions-enabled version (e.g., mediapipe==0.10.9)."
        )
        log(f"[mediapipe] {msg}")
        print(json.dumps({"ok": False, "error": msg}))
        return 2

    try:
        from PIL import Image
    except Exception as exc:
        log(f"[mediapipe] Pillow import failed: {exc}")
        print(json.dumps({"ok": False, "error": f"Pillow import failed: {exc}"}))
        return 2

    try:
        import numpy as np
    except Exception as exc:
        log(f"[mediapipe] numpy import failed: {exc}")
        print(json.dumps({"ok": False, "error": f"numpy import failed: {exc}"}))
        return 2

    per_frame = []
    per_frame_centers = []
    centers = []
    multi_face = False
    speaker_center_x = None
    speaker_motion = None
    speaker_frame_ratio = None
    speaker_motion_ratio = None

    log(f"[mediapipe] active_speaker={1 if args.active_speaker else 0}")

    if args.active_speaker:
        tracks = {}
        next_track_id = 1
        frames_with_faces = 0

        def _match_track(center, width):
            nonlocal tracks
            best_id = None
            best_dist = None
            for track_id, track in tracks.items():
                last = track["last_center"]
                dist = ((center[0] - last[0]) ** 2 + (center[1] - last[1]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = track_id
            if best_dist is None:
                return None
            if best_dist > (0.2 * width):
                return None
            return best_id

        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            for frame_path in args.frames:
                if not os.path.exists(frame_path):
                    log(f"[mediapipe] frame missing: {frame_path}")
                    per_frame.append({"path": frame_path, "faces": 0, "error": "missing"})
                    continue

                try:
                    image = Image.open(frame_path).convert("RGB")
                    image_np = np.array(image)
                except Exception as exc:
                    log(f"[mediapipe] failed to read frame {frame_path}: {exc}")
                    per_frame.append({"path": frame_path, "faces": 0, "error": str(exc)})
                    continue

                results = mesh.process(image_np)
                landmarks_list = results.multi_face_landmarks or []
                face_count = len(landmarks_list)
                per_frame.append({"path": frame_path, "faces": face_count})
                log(f"[mediapipe] {os.path.basename(frame_path)} faces={face_count}")

                if face_count >= 2:
                    multi_face = True

                detections = []
                for face_landmarks in landmarks_list:
                    xs = [lm.x for lm in face_landmarks.landmark]
                    ys = [lm.y for lm in face_landmarks.landmark]
                    if not xs or not ys:
                        continue
                    min_x = max(0.0, min(xs))
                    max_x = min(1.0, max(xs))
                    min_y = max(0.0, min(ys))
                    max_y = min(1.0, max(ys))
                    center_x = ((min_x + max_x) / 2.0) * float(image.width)
                    center_y = ((min_y + max_y) / 2.0) * float(image.height)

                    try:
                        upper = face_landmarks.landmark[13]
                        lower = face_landmarks.landmark[14]
                        mouth_open = abs((upper.y - lower.y)) * float(image.height)
                    except Exception:
                        mouth_open = 0.0

                    detections.append({
                        "center": (center_x, center_y),
                        "mouth_open": mouth_open,
                    })

                if detections:
                    frames_with_faces += 1
                    best_in_frame = max(detections, key=lambda d: d["mouth_open"])
                    per_frame_centers.append(float(best_in_frame["center"][0]))
                else:
                    per_frame_centers.append(None)

                for det in detections:
                    track_id = _match_track(det["center"], image.width)
                    if track_id is None:
                        track_id = next_track_id
                        next_track_id += 1
                        tracks[track_id] = {
                            "last_center": det["center"],
                            "last_mouth": det["mouth_open"],
                            "motion": 0.0,
                            "centers": [det["center"][0]],
                            "frames": 1,
                        }
                        continue

                    track = tracks[track_id]
                    if track["last_mouth"] is not None:
                        track["motion"] += abs(det["mouth_open"] - track["last_mouth"])
                    track["last_center"] = det["center"]
                    track["last_mouth"] = det["mouth_open"]
                    track["centers"].append(det["center"][0])
                    track["frames"] += 1

        best_track = None
        total_motion = 0.0
        for track in tracks.values():
            if track["frames"] < 2:
                continue
            total_motion += track["motion"]
            if best_track is None:
                best_track = track
                continue
            if track["motion"] > best_track["motion"]:
                best_track = track

        if best_track and best_track["centers"]:
            speaker_center_x = sum(best_track["centers"]) / len(best_track["centers"])
            speaker_motion = best_track["motion"]
            if frames_with_faces > 0:
                speaker_frame_ratio = best_track["frames"] / float(frames_with_faces)
            if total_motion > 0:
                speaker_motion_ratio = best_track["motion"] / total_motion
            centers.append(speaker_center_x)
            log(f"[mediapipe] speaker_motion={speaker_motion:.4f}")
            if speaker_frame_ratio is not None:
                log(f"[mediapipe] speaker_frame_ratio={speaker_frame_ratio:.2f}")
            if speaker_motion_ratio is not None:
                log(f"[mediapipe] speaker_motion_ratio={speaker_motion_ratio:.2f}")
        else:
            log("[mediapipe] speaker track not found")
    else:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as detector:
            for frame_path in args.frames:
                if not os.path.exists(frame_path):
                    log(f"[mediapipe] frame missing: {frame_path}")
                    per_frame.append({"path": frame_path, "faces": 0, "error": "missing"})
                    continue

                try:
                    image = Image.open(frame_path).convert("RGB")
                    image_np = np.array(image)
                except Exception as exc:
                    log(f"[mediapipe] failed to read frame {frame_path}: {exc}")
                    per_frame.append({"path": frame_path, "faces": 0, "error": str(exc)})
                    continue

                results = detector.process(image_np)
                detections = results.detections or []
                face_count = len(detections)
                per_frame.append({"path": frame_path, "faces": face_count})
                log(f"[mediapipe] {os.path.basename(frame_path)} faces={face_count}")

                if face_count >= 2:
                    multi_face = True

                if face_count >= 1:
                    best = None
                    best_area = 0.0
                    for det in detections:
                        bbox = det.location_data.relative_bounding_box
                        area = float(bbox.width * bbox.height)
                        if area > best_area:
                            best_area = area
                            best = bbox

                    if best is not None:
                        center_x = (best.xmin + (best.width / 2.0)) * float(image.width)
                        if np.isfinite(center_x):
                            centers.append(float(center_x))
                            per_frame_centers.append(float(center_x))
                        else:
                            per_frame_centers.append(None)
                    else:
                        per_frame_centers.append(None)
                else:
                    per_frame_centers.append(None)

    avg_center_x = (sum(centers) / len(centers)) if centers else None

    payload = {
        "ok": True,
        "multi_face": multi_face,
        "center_x": avg_center_x,
        "speaker_center_x": speaker_center_x,
        "speaker_motion": speaker_motion,
        "speaker_frame_ratio": speaker_frame_ratio,
        "speaker_motion_ratio": speaker_motion_ratio,
        "frame_centers": per_frame_centers,
        "frames": per_frame,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
