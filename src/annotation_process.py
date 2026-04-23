import sys
from pathlib import Path
import json
sys.path.append(str(Path("..").resolve()))

data_dir = Path("../data")

def process_bambi_annotations(
    ann_file: Path | str,
    output_dir: Path | str,
    flight_id: str,                     
    visibility_threshold: float = 0.3,
    skip_propagated: bool = True,
    img_width: int = 1024,
    img_height: int = 1024
) -> tuple[int, int, dict]:
    """
    Parses a BAMBI MOT annotation file, extracts tracking metadata, 
    and writes YOLO-formatted label files in a single pass.
    
    Returns:
        tuple: (written_frames, empty_frames, track_spans_dictionary)
    """
    ann_file = Path(ann_file)
    output_dir = Path(output_dir)

    SPECIES_MAP = {
        "Sus scrofa (Wild boar)": 0,
        "Cervus elaphus (Red deer)": 1,
        "Capreolus capreolus (Roe deer)": 2,
    }

    yolo_labels: dict[int, list[str]] = {}
    all_frame_ids: set[int] = set()
    track_spans: dict[int, dict] = {}

    with open(ann_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 13:
                continue

            frame_id      = int(parts[0])
            track_id      = int(parts[1])
            bb_left       = float(parts[2])
            bb_top        = float(parts[3])
            bb_width      = float(parts[4])
            bb_height     = float(parts[5])
            visibility    = float(parts[8])
            species       = parts[9].strip()
            is_propagated = int(parts[12])

            all_frame_ids.add(frame_id)

            if track_id not in track_spans:
                track_spans[track_id] = {
                    "first": frame_id,
                    "last": frame_id,
                    "species": species
                }
            else:
                span = track_spans[track_id]
                span["first"] = min(span["first"], frame_id)
                span["last"]  = max(span["last"], frame_id)

            if skip_propagated and is_propagated == 1:
                continue
            if visibility < visibility_threshold:
                continue
            if species not in SPECIES_MAP:
                continue

            center_x = (bb_left + bb_width  / 2) / img_width
            center_y = (bb_top  + bb_height / 2) / img_height
            norm_w   = bb_width  / img_width
            norm_h   = bb_height / img_height
            class_id = SPECIES_MAP[species]
            label = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
            yolo_labels.setdefault(frame_id, []).append(label)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_count = 0
    for frame_id, labels in yolo_labels.items():
        (output_dir / f"{flight_id}_{frame_id:08d}.txt").write_text("\n".join(labels))
        written_count += 1

    empty_count = len(all_frame_ids) - written_count
    return written_count, empty_count, track_spans