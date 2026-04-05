"""
Pre-computed audio analysis generator for ContentGuard.

Generates .analysis.json sidecar files for each audio clip in the manifest.
These files contain:
  1. Standard observation field values (aligned with archetype constraints)
  2. Audio-specific metadata (simulating librosa output: MFCC, chroma, tempo, etc.)

In a production system, this script would run librosa on real audio files.
For the hackathon, we generate realistic analysis data that matches the
archetype constraints while including plausible audio feature values.

Usage:
    python -m server.skills.precompute
"""

import json
import os
import random
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "audio"

# Maps archetype → observation field values and audio features.
# These are calibrated to match the archetype's ground_truth_range
# and correct_verdict from case_generator.py.
CLIP_ANALYSIS: dict[str, dict] = {
    "ai_synth_melody_01": {
        "assess_transformation": {
            "transformation_index": 0.22,
            "commentary_present": 0,
            "overlap_duration_pct": 0.78,
            "audio_metadata": {
                "tempo_bpm": 120.0,
                "duration_s": 22.0,
                "spectral_centroid_hz": 2840.5,
                "speech_ratio": 0.0,
                "description": "Instrumental — AI-reconstructed melody with no vocal content"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 0,
            "composition_similarity_score": 0.82,
            "audio_metadata": {
                "mfcc_signature": [-210.4, 85.2, -12.6, 35.8, -8.1],
                "chroma_profile": [0.45, 0.12, 0.68, 0.23, 0.81, 0.34, 0.56, 0.19, 0.72, 0.28, 0.63, 0.41],
                "fingerprint_method": "chromaprint",
                "note": "Fingerprint miss — AI-generated audio evades exact match but harmonic structure is nearly identical"
            }
        }
    },
    "ai_synth_melody_02": {
        "assess_transformation": {
            "transformation_index": 0.18,
            "commentary_present": 0,
            "overlap_duration_pct": 0.85,
            "audio_metadata": {
                "tempo_bpm": 96.0,
                "duration_s": 18.0,
                "spectral_centroid_hz": 3120.3,
                "speech_ratio": 0.0,
                "description": "AI chord progression — retimed but harmonically derivative"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 0,
            "composition_similarity_score": 0.79,
            "audio_metadata": {
                "mfcc_signature": [-195.8, 78.4, -18.2, 42.1, -5.9],
                "chroma_profile": [0.52, 0.18, 0.71, 0.29, 0.76, 0.31, 0.48, 0.22, 0.69, 0.35, 0.58, 0.44],
                "fingerprint_method": "chromaprint",
                "note": "Synthetic generation evades waveform matching; composition analysis reveals high similarity"
            }
        }
    },
    "bg_music_commercial_01": {
        "assess_transformation": {
            "transformation_index": 0.12,
            "commentary_present": 0,
            "overlap_duration_pct": 0.55,
            "audio_metadata": {
                "tempo_bpm": 128.0,
                "duration_s": 25.0,
                "spectral_centroid_hz": 3450.2,
                "speech_ratio": 0.15,
                "description": "Background music bed under commercial voiceover — minimal transformation"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.91,
            "audio_metadata": {
                "mfcc_signature": [-180.2, 92.1, -8.4, 28.7, -11.3],
                "chroma_profile": [0.38, 0.25, 0.82, 0.15, 0.73, 0.42, 0.61, 0.28, 0.55, 0.32, 0.71, 0.19],
                "fingerprint_method": "chromaprint",
                "note": "Direct fingerprint match — original track used as background music"
            }
        }
    },
    "bg_music_commercial_02": {
        "assess_transformation": {
            "transformation_index": 0.08,
            "commentary_present": 0,
            "overlap_duration_pct": 0.48,
            "audio_metadata": {
                "tempo_bpm": 110.0,
                "duration_s": 20.0,
                "spectral_centroid_hz": 2980.7,
                "speech_ratio": 0.22,
                "description": "Product advertisement — licensed track used beyond scope of license agreement"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.88,
            "audio_metadata": {
                "mfcc_signature": [-172.5, 88.6, -14.1, 31.2, -9.8],
                "chroma_profile": [0.41, 0.19, 0.75, 0.21, 0.78, 0.38, 0.55, 0.31, 0.62, 0.27, 0.68, 0.24],
                "fingerprint_method": "chromaprint",
                "note": "Exact match — track identified in commercial context exceeding license terms"
            }
        }
    },
    "commentary_speech_01": {
        "assess_transformation": {
            "transformation_index": 0.82,
            "commentary_present": 1,
            "overlap_duration_pct": 0.15,
            "audio_metadata": {
                "tempo_bpm": 0.0,
                "duration_s": 28.0,
                "spectral_centroid_hz": 1850.4,
                "speech_ratio": 0.85,
                "description": "Podcast commentary — 85% original speech, 15% copyrighted audio excerpt for critique"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.35,
            "audio_metadata": {
                "mfcc_signature": [-145.8, 62.3, -22.4, 18.9, -15.6],
                "chroma_profile": [0.28, 0.45, 0.31, 0.52, 0.22, 0.61, 0.35, 0.48, 0.29, 0.55, 0.33, 0.42],
                "fingerprint_method": "chromaprint",
                "note": "Partial match on brief excerpt — overwhelmingly original speech content"
            }
        }
    },
    "commentary_speech_02": {
        "assess_transformation": {
            "transformation_index": 0.78,
            "commentary_present": 1,
            "overlap_duration_pct": 0.20,
            "audio_metadata": {
                "tempo_bpm": 0.0,
                "duration_s": 24.0,
                "spectral_centroid_hz": 2010.8,
                "speech_ratio": 0.78,
                "description": "Music review — critical analysis with short excerpt playback for illustration"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.42,
            "audio_metadata": {
                "mfcc_signature": [-152.1, 58.7, -19.8, 22.5, -13.2],
                "chroma_profile": [0.32, 0.41, 0.28, 0.55, 0.25, 0.58, 0.38, 0.44, 0.31, 0.52, 0.36, 0.39],
                "fingerprint_method": "chromaprint",
                "note": "Brief excerpt match — context is critical commentary and review"
            }
        }
    },
    "edu_lecture_01": {
        "assess_transformation": {
            "transformation_index": 0.62,
            "commentary_present": 1,
            "overlap_duration_pct": 0.25,
            "audio_metadata": {
                "tempo_bpm": 0.0,
                "duration_s": 30.0,
                "spectral_centroid_hz": 1720.6,
                "speech_ratio": 0.75,
                "description": "University lecture — copyrighted audio used as teaching example with professor commentary"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.48,
            "audio_metadata": {
                "mfcc_signature": [-138.4, 55.1, -25.6, 15.3, -18.4],
                "chroma_profile": [0.25, 0.38, 0.42, 0.48, 0.31, 0.55, 0.29, 0.51, 0.35, 0.45, 0.38, 0.41],
                "fingerprint_method": "chromaprint",
                "note": "Educational context — excerpt used for illustrative purposes in lecture setting"
            }
        }
    },
    "edu_lecture_02": {
        "assess_transformation": {
            "transformation_index": 0.58,
            "commentary_present": 1,
            "overlap_duration_pct": 0.30,
            "audio_metadata": {
                "tempo_bpm": 0.0,
                "duration_s": 22.0,
                "spectral_centroid_hz": 1890.2,
                "speech_ratio": 0.70,
                "description": "Music theory lesson — brief copyrighted excerpt with detailed harmonic analysis overlay"
            }
        },
        "check_fingerprint": {
            "fingerprint_match": 1,
            "composition_similarity_score": 0.52,
            "audio_metadata": {
                "mfcc_signature": [-142.7, 51.8, -21.3, 19.1, -16.7],
                "chroma_profile": [0.29, 0.42, 0.38, 0.45, 0.28, 0.52, 0.33, 0.49, 0.32, 0.48, 0.35, 0.43],
                "fingerprint_method": "chromaprint",
                "note": "Educational — brief excerpt with analytical commentary throughout"
            }
        }
    }
}


def write_analysis_files() -> None:
    """Write .analysis.json files for each clip."""
    os.makedirs(AUDIO_DIR, exist_ok=True)

    for clip_id, analysis in CLIP_ANALYSIS.items():
        path = AUDIO_DIR / f"{clip_id}.analysis.json"
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"  wrote {path.name}")

    print(f"\n{len(CLIP_ANALYSIS)} analysis files written to {AUDIO_DIR}")


if __name__ == "__main__":
    write_analysis_files()
