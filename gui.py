from typing import Optional, Any
import os
import sys
import torch
import logging
import yt_dlp
from yt_dlp import YoutubeDL
import gradio as gr
import argparse
from audio_separator.separator import Separator
import numpy as np
import librosa
import soundfile as sf
from ensemble import ensemble_files
import shutil
import gradio_client.utils as client_utils
import matchering as mg
import gdown
from pydub import AudioSegment
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import scipy.io.wavfile
import subprocess
import torchaudio
from models_config import (
    EXTENDED_MODELS, get_all_models, get_categories, get_model_choices,
    find_model_filename, add_custom_model, delete_custom_model, load_custom_models,
    get_custom_models_list, ensure_model_files_downloaded,
    get_audio_duration, split_audio_segments, concatenate_segment_outputs,
    MAX_UNSPLIT_DURATION, SEGMENT_DURATION
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gradio JSON schema patch
original_json_schema_to_python_type = client_utils._json_schema_to_python_type

def patched_json_schema_to_python_type(schema: Any, defs: Optional[dict] = None) -> str:
    logger.debug(f"Parsing schema: {schema}")
    if isinstance(schema, bool):
        logger.info("Found boolean schema, returning 'boolean'")
        return "boolean"
    if not isinstance(schema, dict):
        logger.warning(f"Unexpected schema type: {type(schema)}, returning 'Any'")
        return "Any"
    if "enum" in schema and schema.get("type") == "string":
        logger.info(f"Handling enum schema: {schema['enum']}")
        return f"Literal[{', '.join(repr(e) for e in schema['enum'])}]"
    try:
        return original_json_schema_to_python_type(schema, defs)
    except client_utils.APIInfoParseError as e:
        logger.error(f"Failed to parse schema {schema}: {e}")
        return "str"

client_utils._json_schema_to_python_type = patched_json_schema_to_python_type

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"
logger.info(f"Using device: {device}")

# Constants
max_models = 6
max_retries = 2
time_budget = 300  # ZeroGPU için işlem sınırı
gpu_lock = Lock()

# ROFORMER_MODELS - now using EXTENDED_MODELS from models_config
ROFORMER_MODELS = get_all_models()

OUTPUT_FORMATS = ['wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'aiff', 'ac3']

# CSS (orijinal CSS korundu)
CSS = """
body {
    background: linear-gradient(to bottom, rgba(45, 11, 11, 0.9), rgba(0, 0, 0, 0.8)), url('/content/logo.jpg') no-repeat center center fixed;
    background-size: cover;
    min-height: 100vh;
    margin: 0;
    padding: 1rem;
    font-family: 'Poppins', sans-serif;
    color: #C0C0C0;
    overflow-x: hidden;
}
.header-text {
    text-align: center;
    padding: 100px 20px 20px;
    color: #ff4040;
    font-size: 3rem;
    font-weight: 900;
    text-shadow: 0 0 10px rgba(255, 64, 64, 0.5);
    z-index: 1500;
    animation: text-glow 2s infinite;
}
.header-subtitle {
    text-align: center;
    color: #C0C0C0;
    font-size: 1.2rem;
    font-weight: 300;
    margin-top: -10px;
    text-shadow: 0 0 5px rgba(255, 64, 64, 0.3);
}
.gr-tab {
    background: rgba(128, 0, 0, 0.5) !important;
    border-radius: 12px 12px 0 0 !important;
    margin: 0 5px !important;
    color: #C0C0C0 !important;
    border: 1px solid #ff4040 !important;
    z-index: 1500;
    transition: background 0.3s ease, color 0.3s ease;
    padding: 10px 20px !important;
    font-size: 1.1rem !important;
}
button {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: #800000 !important;
    border: 1px solid #ff4040 !important;
    color: #C0C0C0 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    box-shadow: 0 2px 10px rgba(255, 64, 64, 0.3);
}
button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 10px 40px rgba(255, 64, 64, 0.7) !important;
    background: #ff4040 !important;
}
.compact-upload.horizontal {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    max-width: 400px !important;
    height: 40px !important;
    padding: 0 12px !important;
    border: 1px solid #ff4040 !important;
    background: rgba(128, 0, 0, 0.5) !important;
    border-radius: 8px !important;
}
.compact-dropdown {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    border: 2px solid #ff6b6b !important;
    background: rgba(46, 26, 71, 0.7) !important;
    color: #e0e0e0 !important;
    width: 100%;
    font-size: 1rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    position: relative;
    z-index: 100;
}
.compact-dropdown:hover {
    border-color: #ff8787 !important;
    box-shadow: 0 2px 8px rgba(255, 107, 107, 0.4) !important;
}
.compact-dropdown select, .compact-dropdown .gr-dropdown {
    background: transparent !important;
    color: #e0e0e0 !important;
    border: none !important;
    width: 100% !important;
    padding: 8px !important;
    font-size: 1rem !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
}
.compact-dropdown .gr-dropdown-menu {
    background: rgba(46, 26, 71, 0.95) !important;
    border: 2px solid #ff6b6b !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    z-index: 300 !important;
    width: 100% !important;
    opacity: 1 !important;
    visibility: visible !important;
    position: absolute !important;
    top: 100% !important;
    left: 0 !important;
    pointer-events: auto !important;
}
.compact-dropdown:hover .gr-dropdown-menu {
    display: block !important;
}
.compact-dropdown .gr-dropdown-menu option {
    padding: 8px !important;
    color: #e0e0e0 !important;
    background: transparent !important;
}
.compact-dropdown .gr-dropdown-menu option:hover {
    background: rgba(255, 107, 107, 0.3) !important;
}
#custom-progress {
    margin-top: 10px;
    padding: 10px;
    background: rgba(128, 0, 0, 0.3);
    border-radius: 8px;
    border: 1px solid #ff4040;
}
#progress-bar {
    height: 20px;
    background: linear-gradient(to right, #6e8efb, #ff4040);
    border-radius: 5px;
    transition: width 0.5s ease-in-out;
    max-width: 100% !important;
}
.gr-accordion {
    background: rgba(128, 0, 0, 0.5) !important;
    border-radius: 10px !important;
    border: 1px solid #ff4040 !important;
}
.footer {
    text-align: center;
    padding: 20px;
    color: #ff4040;
    font-size: 14px;
    margin-top: 40px;
    background: rgba(128, 0, 0, 0.3);
    border-top: 1px solid #ff4040;
}
#log-accordion {
    max-height: 400px;
    overflow-y: auto;
    background: rgba(0, 0, 0, 0.7) !important;
    padding: 10px;
    border-radius: 8px;
}
@keyframes text-glow {
    0% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
    50% { text-shadow: 0 0 15px rgba(192, 192, 192, 1); }
    100% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
}
"""

def download_audio(url, cookie_file=None):
    """
    Downloads audio from YouTube or Google Drive and converts it to WAV format.
    
    Args:
        url (str): URL of the YouTube video or Google Drive file.
        cookie_file (file object): File object containing YouTube cookies in Netscape format.
    
    Returns:
        tuple: (file_path, message, (sample_rate, data)) or (None, error_message, None)
    """
    # Common output directory
    os.makedirs('ytdl', exist_ok=True)
    
    # Validate cookie file
    cookie_path = None
    if cookie_file:
        if not hasattr(cookie_file, 'name') or not os.path.exists(cookie_file.name):
            return None, "Invalid or missing cookie file. Ensure it's a valid Netscape format .txt file.", None
        cookie_path = cookie_file.name
        # Check if cookie file is in Netscape format
        with open(cookie_path, 'r') as f:
            content = f.read()
            if not content.startswith('# Netscape HTTP Cookie File'):
                return None, "Cookie file is not in Netscape format. See https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies", None
        logger.info(f"Using cookie file: {cookie_path}")
    
    if 'drive.google.com' in url:
        return download_from_google_drive(url)
    else:
        return download_from_youtube(url, cookie_path)

def download_from_youtube(url, cookie_path):
    # Common options
    base_opts = {
        'outtmpl': 'ytdl/%(title)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
        'geo_bypass': True,
        'force_ipv4': True,
        'referer': 'https://www.youtube.com/',
        'noplaylist': True,
        'cookiefile': cookie_path,
        'extractor_retries': 5,
        'ignoreerrors': False,
        'no_check_certificate': True,
        'verbose': True,
    }
    
    # Strategy 1: Video+audio (best quality)
    try:
        logger.info("Attempting video+audio download")
        ydl_opts = base_opts.copy()
        ydl_opts.update({
            'format': 'bestvideo+bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'merge_output_format': 'mp4',
        })
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
            
            if os.path.exists(file_path):
                sample_rate, data = scipy.io.wavfile.read(file_path)
                return file_path, "YouTube video+audio download successful", (sample_rate, data)
            else:
                logger.warning("Video+audio download succeeded but output file missing")
    except Exception as e:
        logger.warning(f"Video+audio download failed: {str(e)}")
    
    # Strategy 2: Audio-only (best quality)
    try:
        logger.info("Attempting audio-only download")
        ydl_opts = base_opts.copy()
        ydl_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
        })
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
            
            if os.path.exists(file_path):
                sample_rate, data = scipy.io.wavfile.read(file_path)
                return file_path, "YouTube audio-only download successful", (sample_rate, data)
            else:
                logger.warning("Audio-only download succeeded but output file missing")
    except Exception as e:
        logger.warning(f"Audio-only download failed: {str(e)}")
    
    # Strategy 3: Specific format IDs (common audio formats)
    format_ids = [
        '140',  # m4a 128k
        '139',  # m4a 48k
        '251',  # webm 160k (opus)
        '250',  # webm 70k (opus)
        '249',  # webm 50k (opus)
    ]
    
    for fid in format_ids:
        try:
            logger.info(f"Attempting download with format ID: {fid}")
            ydl_opts = base_opts.copy()
            ydl_opts.update({
                'format': fid,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
            })
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
                
                if os.path.exists(file_path):
                    sample_rate, data = scipy.io.wavfile.read(file_path)
                    return file_path, f"Download successful with format {fid}", (sample_rate, data)
        except Exception as e:
            logger.warning(f"Download with format {fid} failed: {str(e)}")
    
    # Strategy 4: Direct URL extraction
    try:
        logger.info("Attempting direct URL extraction")
        ydl_opts = base_opts.copy()
        ydl_opts.update({
            'format': 'best',
            'forceurl': True,
            'quiet': True,
        })
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            direct_url = info_dict.get('url')
            
            if direct_url:
                temp_path = 'ytdl/direct_audio.wav'
                ffmpeg_command = [
                    "ffmpeg", "-i", direct_url, "-c", "copy", temp_path
                ]
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                
                if os.path.exists(temp_path):
                    sample_rate, data = scipy.io.wavfile.read(temp_path)
                    return temp_path, "Direct URL download successful", (sample_rate, data)
    except Exception as e:
        logger.warning(f"Direct URL extraction failed: {str(e)}")
    
    return None, "All download strategies failed. This video may not be available in your region or requires authentication.", None
        
def download_from_google_drive(url):
    temp_output_path = 'ytdl/gdrive_temp_audio'
    output_path = 'ytdl/gdrive_audio.wav'
    
    try:
        # Extract file ID from URL
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download file
        gdown.download(download_url, temp_output_path, quiet=False)
        
        if not os.path.exists(temp_output_path):
            return None, "Google Drive downloaded file not found", None
        
        # Convert to WAV
        audio = AudioSegment.from_file(temp_output_path)
        audio.export(output_path, format="wav")
        
        sample_rate, data = scipy.io.wavfile.read(output_path)
        return output_path, "Google Drive audio download and conversion successful", (sample_rate, data)
    
    except Exception as e:
        return None, f"Failed to process Google Drive file: {str(e)}. Ensure the file contains audio (e.g., MP3, WAV, or video with audio track).", None
    
    finally:
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
                logger.info(f"Temporary file deleted: {temp_output_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_output_path}: {str(e)}")

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, output_dir, out_format, norm_thresh, amp_thresh, batch_size, exclude_stems="", progress=gr.Progress(track_tqdm=True)):
    if not audio:
        raise ValueError("No audio or video file provided.")
    temp_audio_path = None
    extracted_audio_path = None
    segment_temp_dir = None
    try:
        file_extension = os.path.splitext(audio)[1].lower().lstrip('.')
        supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'opus', 'm4a', 'aiff', 'ac3', 'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg', 'ts', 'vob']
        if file_extension not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(supported_formats)}")

        audio_to_process = audio
        if file_extension in ['mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg', 'ts', 'vob']:
            extracted_audio_path = os.path.join("/tmp", f"extracted_audio_{os.path.basename(audio)}.wav")
            logger.info(f"Extracting audio from video file: {audio}")
            ffmpeg_command = [
                "ffmpeg", "-i", audio, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                extracted_audio_path, "-y"
            ]
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                logger.info(f"Audio extracted to: {extracted_audio_path}")
                audio_to_process = extracted_audio_path
            except subprocess.CalledProcessError as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                if "No audio stream" in error_message:
                    raise RuntimeError("The provided video file does not contain an audio track.")
                elif "Invalid data" in error_message:
                    raise RuntimeError("The video file is corrupted or not supported.")
                else:
                    raise RuntimeError(f"Failed to extract audio from video: {error_message}")

        if isinstance(audio_to_process, tuple):
            sample_rate, data = audio_to_process
            temp_audio_path = os.path.join("/tmp", "temp_audio.wav")
            scipy.io.wavfile.write(temp_audio_path, sample_rate, data)
            audio_to_process = temp_audio_path

        if seg_size > 512:
            logger.warning(f"Segment size {seg_size} is large, this may cause issues.")
        override_seg_size = override_seg_size == "True"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio))[0].replace(' ', '_')

        # Find model from EXTENDED_MODELS + custom models
        model = find_model_filename(model_key)
        if not model:
            raise ValueError(f"Model '{model_key}' not found.")

        # Pre-download model files (checkpoint + config YAML) before loading
        # This is required for the separator.py bypass to work
        dl_success, dl_msg = ensure_model_files_downloaded(model, model_dir)
        if not dl_success:
            logger.warning(f"Pre-download warning for {model}: {dl_msg}")

        logger.info(f"Separating {base_name} with {model_key} on {device}")

        # ── Large file segmentation ──
        audio_duration = get_audio_duration(audio_to_process)
        was_segmented = False
        if audio_duration > MAX_UNSPLIT_DURATION:
            duration_min = audio_duration / 60
            logger.info(f"⚠️ Large audio detected: {duration_min:.0f} min. Splitting to prevent OOM...")
            progress(0.05, desc=f"Splitting {duration_min:.0f} min audio into segments...")
            segment_temp_dir = os.path.join("/tmp", f"sesa_segments_{base_name}")
            os.makedirs(segment_temp_dir, exist_ok=True)
            segments = split_audio_segments(audio_to_process, segment_temp_dir, SEGMENT_DURATION)
            if segments:
                was_segmented = True
                logger.info(f"Split into {len(segments)} segments")
                # Process each segment
                seg_output_dir = os.path.join("/tmp", f"sesa_seg_output_{base_name}")
                os.makedirs(seg_output_dir, exist_ok=True)
                for i, seg_path in enumerate(segments):
                    progress(0.1 + 0.7 * (i / len(segments)), desc=f"Processing segment {i+1}/{len(segments)}...")
                    separator = Separator(
                        log_level=logging.INFO,
                        model_file_dir=model_dir,
                        output_dir=seg_output_dir,
                        output_format=out_format,
                        normalization_threshold=norm_thresh,
                        amplification_threshold=amp_thresh,
                        use_autocast=use_autocast,
                        mdxc_params={"segment_size": seg_size, "override_model_segment_size": override_seg_size, "batch_size": batch_size, "overlap": overlap, "pitch_shift": pitch_shift}
                    )
                    separator.load_model(model_filename=model)
                    separator.separate(seg_path)
                    # Free GPU memory between segments
                    del separator
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                # Concatenate segment outputs
                progress(0.85, desc="Concatenating segments...")
                concatenate_segment_outputs(seg_output_dir, out_format)
                # Move final concatenated files to output_dir
                for f in os.listdir(seg_output_dir):
                    if '_seg' not in f.lower():  # Only move final merged files
                        shutil.move(os.path.join(seg_output_dir, f), os.path.join(output_dir, f))
                # Cleanup temp dirs
                shutil.rmtree(segment_temp_dir, ignore_errors=True)
                shutil.rmtree(seg_output_dir, ignore_errors=True)
                segment_temp_dir = None

        if not was_segmented:
            # Normal processing (no segmentation)
            separator = Separator(
                log_level=logging.INFO,
                model_file_dir=model_dir,
                output_dir=output_dir,
                output_format=out_format,
                normalization_threshold=norm_thresh,
                amplification_threshold=amp_thresh,
                use_autocast=use_autocast,
                mdxc_params={"segment_size": seg_size, "override_model_segment_size": override_seg_size, "batch_size": batch_size, "overlap": overlap, "pitch_shift": pitch_shift}
            )
            progress(0.2, desc="Loading model...")
            separator.load_model(model_filename=model)
            progress(0.7, desc="Separating audio...")
            separator.separate(audio_to_process)

        # Collect all output stems
        output_files = os.listdir(output_dir)
        stems = [os.path.join(output_dir, f) for f in output_files if os.path.isfile(os.path.join(output_dir, f))]
        file_list = []
        if exclude_stems.strip():
            excluded = [s.strip().lower() for s in exclude_stems.split(',')]
            filtered_stems = [stem for stem in stems if not any(ex in os.path.basename(stem).lower() for ex in excluded)]
            file_list = filtered_stems
            stem1 = filtered_stems[0] if filtered_stems else None
            stem2 = filtered_stems[1] if len(filtered_stems) > 1 else None
        else:
            file_list = stems
            stem1 = stems[0] if stems else None
            stem2 = stems[1] if len(stems) > 1 else None

        return stem1, stem2, file_list

    except Exception as e:
        logger.error(f"Separation error: {e}")
        raise RuntimeError(f"Separation error: {e}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Temporary file deleted: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            try:
                os.remove(extracted_audio_path)
                logger.info(f"Extracted audio file deleted: {extracted_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete extracted audio file {extracted_audio_path}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")

def auto_ensemble_process(audio, model_keys, state, seg_size=64, overlap=0.1, out_format="wav", use_tta="False", model_dir="/tmp/audio-separator-models/", output_dir="output", norm_thresh=0.9, amp_thresh=0.9, batch_size=1, ensemble_method="avg_wave", exclude_stems="", weights_str="", progress=gr.Progress(track_tqdm=True)):
    temp_audio_path = None
    extracted_audio_path = None
    resampled_audio_path = None
    start_time = time.time()
    try:
        if not audio:
            raise ValueError("No audio or video file provided.")
        if not model_keys:
            raise ValueError("No models selected.")
        if len(model_keys) > max_models:
            logger.warning(f"Selected {len(model_keys)} models, limiting to {max_models}.")
            model_keys = model_keys[:max_models]

        file_extension = os.path.splitext(audio)[1].lower().lstrip('.')
        supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'opus', 'm4a', 'aiff', 'ac3', 'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg', 'ts', 'vob']
        if file_extension not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(supported_formats)}")

        audio_to_process = audio
        if file_extension in ['mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg', 'ts', 'vob']:
            extracted_audio_path = os.path.join("/tmp", f"extracted_audio_{os.path.basename(audio)}.wav")
            logger.info(f"Extracting audio from video file: {audio}")
            ffmpeg_command = [
                "ffmpeg", "-i", audio, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                extracted_audio_path, "-y"
            ]
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                logger.info(f"Audio extracted to: {extracted_audio_path}")
                audio_to_process = extracted_audio_path
            except subprocess.CalledProcessError as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                if "No audio stream" in error_message:
                    raise RuntimeError("The provided video file does not contain an audio track.")
                elif "Invalid data" in error_message:
                    raise RuntimeError("The video file is corrupted or not supported.")
                else:
                    raise RuntimeError(f"Failed to extract audio from video: {error_message}")

        # Load audio and resample to 48 kHz
        audio_data, sr = librosa.load(audio_to_process, sr=None, mono=False)
        logger.info(f"Original sample rate: {sr} Hz, Audio duration: {librosa.get_duration(y=audio_data, sr=sr):.2f} seconds")
        if sr != 48000:
            logger.info(f"Resampling audio from {sr} Hz to 48000 Hz")
            resampled_audio_path = os.path.join("/tmp", f"resampled_audio_{os.path.basename(audio)}.wav")
            waveform, _ = torchaudio.load(audio_to_process)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
            resampled_waveform = resampler(waveform)
            torchaudio.save(resampled_audio_path, resampled_waveform, 48000)
            audio_to_process = resampled_audio_path
            audio_data, sr = librosa.load(audio_to_process, sr=None, mono=False)
            logger.info(f"Resampled audio saved to: {resampled_audio_path}, new sample rate: {sr} Hz")

        duration = librosa.get_duration(y=audio_data, sr=sr)
        dynamic_batch_size = max(1, min(4, 1 + int(900 / (duration + 1)) - len(model_keys) // 2))
        logger.info(f"Using batch size: {dynamic_batch_size} for {len(model_keys)} models, duration {duration:.2f}s")

        if isinstance(audio_to_process, tuple):
            sample_rate, data = audio_to_process
            temp_audio_path = os.path.join("/tmp", "temp_audio.wav")
            scipy.io.wavfile.write(temp_audio_path, sample_rate, data)
            audio_to_process = temp_audio_path

        if not state:
            state = {
                "current_audio": None,
                "current_model_idx": 0,
                "processed_stems": [],
                "model_outputs": {}
            }

        if state["current_audio"] != audio:
            state["current_audio"] = audio
            state["current_model_idx"] = 0
            state["processed_stems"] = []
            state["model_outputs"] = {model_key: {"vocals": [], "other": []} for model_key in model_keys}
            logger.info("New audio detected, resetting ensemble state.")

        use_tta = use_tta == "True"
        base_name = os.path.splitext(os.path.basename(audio))[0].replace(' ', '_')  # Boşlukları alt çizgi ile değiştir
        logger.info(f"Ensemble for {base_name} with {model_keys} on {device}")

        permanent_output_dir = os.path.join(output_dir, "permanent_stems")
        os.makedirs(permanent_output_dir, exist_ok=True)

        model_cache = {}
        all_stems = []
        total_tasks = len(model_keys)
        current_idx = state["current_model_idx"]
        logger.info(f"Current model index: {current_idx}, total models: {len(model_keys)}")

        if current_idx >= len(model_keys):
            logger.info("All models processed, running ensemble...")
            progress(0.9, desc="Running ensemble...")

            excluded_stems_list = [s.strip().lower() for s in exclude_stems.split(',')] if exclude_stems.strip() else []
            for model_key, stems_dict in state["model_outputs"].items():
                for stem_type in ["vocals", "other"]:
                    if stems_dict[stem_type]:
                        if stem_type.lower() in excluded_stems_list:
                            logger.info(f"Excluding {stem_type} for {model_key} from ensemble")
                            continue
                        all_stems.extend(stems_dict[stem_type])

            # Dosyaların gerçekten var olduğundan emin ol
            valid_stems = []
            for stem in all_stems:
                if os.path.exists(stem):
                    valid_stems.append(stem)
                else:
                    logger.warning(f"Stem file not found: {stem}")
            
            if not valid_stems:
                raise ValueError("No valid stems found for ensemble after excluding specified stems.")

            weights = [float(w.strip()) for w in weights_str.split(',')] if weights_str.strip() else [1.0] * len(valid_stems)
            if len(weights) != len(valid_stems):
                weights = [1.0] * len(valid_stems)
                logger.info("Weights mismatched, defaulting to 1.0")
            
            # Mutlak yol kullanarak çıktı dosyasını belirle
            output_file = os.path.abspath(os.path.join(output_dir, f"{base_name}_ensemble_{ensemble_method}.{out_format}"))
            # Çıktı dizinini oluştur
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            ensemble_args = [
                "--files", *valid_stems,
                "--type", ensemble_method,
                "--weights", *[str(w) for w in weights],
                "--output", output_file
            ]
            logger.info(f"Running ensemble with args: {ensemble_args}")
            try:
                # Ensemble işlemini denetimli çalıştır
                result = ensemble_files(ensemble_args)
            except Exception as e:
                logger.error(f"Ensemble processing failed: {str(e)}")
                raise RuntimeError(f"Ensemble processing failed: {str(e)}")
            
            # Çıktı dosyasının oluştuğundan emin ol
            if not os.path.exists(output_file):
                # Alternatif yol deneyelim
                alt_path = os.path.join(output_dir, f"{base_name}_ensemble_{ensemble_method}.{out_format}")
                if os.path.exists(alt_path):
                    logger.info(f"Found ensemble output at alternative path: {alt_path}")
                    output_file = alt_path
                else:
                    raise RuntimeError(f"Ensemble output file not created: {output_file}")

            state["current_model_idx"] = 0
            state["current_audio"] = None
            state["processed_stems"] = []
            state["model_outputs"] = {}

            elapsed = time.time() - start_time
            logger.info(f"Ensemble completed, output: {output_file}, took {elapsed:.2f}s")
            progress(1.0, desc="Ensemble completed")
            status = f"Ensemble completed with {ensemble_method}, excluded: {exclude_stems if exclude_stems else 'None'}, {len(model_keys)} models in {elapsed:.2f}s<br>Download files:<ul>"
            file_list = [output_file] + valid_stems
            for file in file_list:
                file_name = os.path.basename(file)
                status += f"<li><a href='file={file}' download>{file_name}</a></li>"
            status += "</ul>"
            return output_file, status, file_list, state

        model_key = model_keys[current_idx]
        logger.info(f"Processing model {current_idx + 1}/{len(model_keys)}: {model_key}")
        progress(0.1, desc=f"Processing model {model_key}...")

        with torch.no_grad():
            for attempt in range(max_retries + 1):
                try:
                    for category, models in ROFORMER_MODELS.items():
                        if model_key in models:
                            model = models[model_key]
                            break
                    else:
                        logger.warning(f"Model {model_key} not found, skipping")
                        state["current_model_idx"] += 1
                        return None, f"Model {model_key} not found, proceeding to next model.", [], state

                    elapsed = time.time() - start_time
                    if elapsed > time_budget:
                        logger.error(f"Time budget ({time_budget}s) exceeded")
                        raise TimeoutError("Processing took too long")

                    if model_key not in model_cache:
                        logger.info(f"Loading {model_key} into cache")
                        # Pre-download model files for bypass
                        dl_ok, dl_msg = ensure_model_files_downloaded(model, model_dir)
                        if not dl_ok:
                            logger.warning(f"Pre-download warning: {dl_msg}")
                        separator = Separator(
                            log_level=logging.INFO,
                            model_file_dir=model_dir,
                            output_dir=output_dir,
                            output_format=out_format,
                            normalization_threshold=norm_thresh,
                            amplification_threshold=amp_thresh,
                            use_autocast=use_autocast,
                            mdxc_params={
                                "segment_size": seg_size,
                                "overlap": overlap,
                                "use_tta": use_tta,
                                "batch_size": dynamic_batch_size
                            }
                        )
                        separator.load_model(model_filename=model)
                        model_cache[model_key] = separator
                    else:
                        separator = model_cache[model_key]

                    with gpu_lock:
                        progress(0.3, desc=f"Separating with {model_key}")
                        logger.info(f"Separating with {model_key}")
                        separation = separator.separate(audio_to_process)
                        stems = [os.path.join(output_dir, file_name) for file_name in separation]
                        result = []
                        for stem in stems:
                            stem_type = "vocals" if "vocals" in os.path.basename(stem).lower() else "other"
                            permanent_stem_path = os.path.join(permanent_output_dir, f"{base_name}_{stem_type}_{model_key.replace(' | ', '_').replace(' ', '_')}.{out_format}")
                            shutil.copy(stem, permanent_stem_path)
                            state["model_outputs"][model_key][stem_type].append(permanent_stem_path)
                            if stem_type not in exclude_stems.lower():
                                result.append(permanent_stem_path)
                        state["processed_stems"].extend(result)
                        break

                except Exception as e:
                    logger.error(f"Error processing {model_key}, attempt {attempt + 1}/{max_retries + 1}: {e}")
                    if attempt == max_retries:
                        logger.error(f"Max retries reached for {model_key}, skipping")
                        state["current_model_idx"] += 1
                        return None, f"Failed to process {model_key} after {max_retries} attempts.", [], state
                    time.sleep(1)

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info(f"Cleared CUDA cache after {model_key}")

        model_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared model cache and GPU memory")

        state["current_model_idx"] += 1
        elapsed = time.time() - start_time
        logger.info(f"Model {model_key} completed in {elapsed:.2f}s")

        if state["current_model_idx"] >= len(model_keys):
            logger.info("Last model processed, running ensemble immediately...")
            return auto_ensemble_process(audio, model_keys, state, seg_size, overlap, out_format, use_tta, model_dir, output_dir, norm_thresh, amp_thresh, batch_size, ensemble_method, exclude_stems, weights_str, progress)

        file_list = state["processed_stems"]
        status = f"Model {model_key} (Model {current_idx + 1}/{len(model_keys)}) completed in {elapsed:.2f}s<br>Click 'Run Ensemble!' to process the next model.<br>Processed stems:<ul>"
        for file in file_list:
            file_name = os.path.basename(file)
            status += f"<li><a href='file={file}' download>{file_name}</a></li>"
        status += "</ul>"
        return file_list[0] if file_list else None, status, file_list, state

    except Exception as e:
        logger.error(f"Ensemble error: {e}")
        # Daha açıklayıcı hata mesajı
        error_msg = f"Processing failed: {e}\n\nPossible solutions:\n"
        error_msg += "1. Try fewer models (max 6)\n"
        error_msg += "2. Upload a local WAV/MP4 file instead of YouTube URL\n"
        error_msg += "3. Reduce segment size or overlap\n"
        error_msg += "4. Check if output directory has write permissions"
        raise RuntimeError(error_msg)

    finally:
        for temp_file in [temp_audio_path, extracted_audio_path, resampled_audio_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Temporary file deleted: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
            
def update_roformer_models(category):
    all_models = get_all_models()
    choices = list(all_models.get(category, {}).keys()) or []
    logger.debug(f"Updating roformer models for category {category}: {choices}")
    return gr.update(choices=choices, value=choices[0] if choices else None)

def update_ensemble_models(category):
    all_models = get_all_models()
    choices = list(all_models.get(category, {}).keys()) or []
    logger.debug(f"Updating ensemble models for category {category}: {choices}")
    return gr.update(choices=choices, value=[])

def download_audio_wrapper(url, cookie_file):
    file_path, status, audio_data = download_audio(url, cookie_file)
    return file_path, status  # Return file_path instead of audio_data

# ─── Batch Processing ────────────────────────────────────────────────────────
def batch_separator(audio_files, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, output_dir, out_format, norm_thresh, amp_thresh, batch_size, exclude_stems="", progress=gr.Progress(track_tqdm=True)):
    """Process up to 10 audio files sequentially."""
    if not audio_files:
        raise ValueError("No audio files provided.")
    if len(audio_files) > 10:
        raise ValueError("Maximum 10 files per batch.")

    all_output_files = []
    status_lines = []
    for i, audio in enumerate(audio_files):
        # Handle gr.File objects
        audio_path = audio.name if hasattr(audio, 'name') else audio
        base = os.path.splitext(os.path.basename(audio_path))[0]
        progress((i) / len(audio_files), desc=f"Processing file {i+1}/{len(audio_files)}: {base}")
        try:
            stem1, stem2, files = roformer_separator(
                audio_path, model_key, seg_size, override_seg_size, overlap, pitch_shift,
                model_dir, output_dir, out_format, norm_thresh, amp_thresh, batch_size,
                exclude_stems, progress
            )
            all_output_files.extend(files)
            status_lines.append(f"✅ {base}: {len(files)} stems")
        except Exception as e:
            status_lines.append(f"❌ {base}: {str(e)[:100]}")
            logger.error(f"Batch processing error for {base}: {e}")

    status_text = "\n".join(status_lines)
    return status_text, all_output_files

# ─── Custom Model Management UI handlers ─────────────────────────────────────
def add_custom_model_handler(name, checkpoint_url, config_url, custom_py_url):
    success, msg = add_custom_model(name, checkpoint_url, config_url, custom_py_url)
    # Refresh ROFORMER_MODELS
    global ROFORMER_MODELS
    ROFORMER_MODELS = get_all_models()
    # Get updated custom model list
    custom_list_data = get_custom_models_list()
    custom_list = "\n".join([f"• {n}: {u}" for n, u in custom_list_data]) if custom_list_data else "No custom models"
    # Return updated categories
    cats = get_categories()
    return msg, custom_list, gr.update(choices=cats), gr.update(choices=cats)

def delete_custom_model_handler(name):
    success, msg = delete_custom_model(name)
    global ROFORMER_MODELS
    ROFORMER_MODELS = get_all_models()
    custom_list_data = get_custom_models_list()
    custom_list = "\n".join([f"• {n}: {u}" for n, u in custom_list_data]) if custom_list_data else "No custom models"
    cats = get_categories()
    return msg, custom_list, gr.update(choices=cats), gr.update(choices=cats)

def create_interface():
    with gr.Blocks(title="🎵 SESA Fast Separation 🎵", css=CSS, elem_id="app-container") as app:
        gr.Markdown("<h1 class='header-text'>🎵 SESA Fast Separation 🎵</h1>")
        gr.Markdown("**Note**: If YouTube downloads fail, upload a valid cookies file or a local WAV/MP4/MOV file. [Cookie Instructions](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)")
        gr.Markdown("**Tip**: For best results, use audio/video shorter than 15 minutes or fewer models (up to 6) to ensure smooth processing.")
        ensemble_state = gr.State(value={
            "current_audio": None,
            "current_model_idx": 0,
            "processed_stems": [],
            "model_outputs": {}
        })
        with gr.Tabs():
            with gr.Tab("⚙️ Settings"):
                with gr.Group(elem_classes="dubbing-theme"):
                    gr.Markdown("### General Settings")
                    model_file_dir = gr.Textbox(value="/tmp/audio-separator-models/", label="📂 Model Cache", placeholder="Path to model directory", interactive=True)
                    output_dir = gr.Textbox(value="output", label="📤 Output Directory", placeholder="Where to save results", interactive=True)
                    output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMATS, label="🎶 Output Format", interactive=True)
                    norm_threshold = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="🔊 Normalization Threshold", interactive=True)
                    amp_threshold = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="📈 Amplification Threshold", interactive=True)
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="⚡ Batch Size", interactive=True)
            with gr.Tab("🎤 Roformer"):
                with gr.Group(elem_classes="dubbing-theme"):
                    gr.Markdown("### Audio Separation")
                    with gr.Row():
                        roformer_audio = gr.File(label="🎧 Upload Audio or Video (WAV, MP3, MP4, MOV, etc.)", file_types=['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aiff', '.ac3', '.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.ts', '.vob'], interactive=True)
                        url_ro = gr.Textbox(label="🔗 Or Paste URL", placeholder="YouTube or audio/video URL", interactive=True)
                        cookies_ro = gr.File(label="🍪 Cookies File", file_types=[".txt"], interactive=True)
                        download_roformer = gr.Button("⬇️ Download", variant="secondary")
                    roformer_download_status = gr.Textbox(label="📢 Download Status", interactive=False)
                    roformer_exclude_stems = gr.Textbox(label="🚫 Exclude Stems", placeholder="e.g., vocals, drums (comma-separated)", interactive=True)
                    with gr.Row():
                        roformer_category = gr.Dropdown(label="📚 Category", choices=get_categories(), value="Vocals", interactive=True)
                        roformer_model = gr.Dropdown(label="🛠️ Model", choices=get_model_choices("Vocals"), interactive=True, allow_custom_value=True)
                    with gr.Row():
                        roformer_seg_size = gr.Slider(32, 512, value=64, step=32, label="📏 Segment Size", interactive=True)
                        roformer_overlap = gr.Slider(2, 10, value=8, step=1, label="🔄 Overlap", interactive=True)
                    with gr.Row():
                        roformer_pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="🎵 Pitch Shift", interactive=True)
                        roformer_override_seg_size = gr.Dropdown(choices=["True", "False"], value="False", label="🔧 Override Segment Size", interactive=True)
                    roformer_button = gr.Button("✂️ Separate Now!", variant="primary")
                    with gr.Row():
                        roformer_stem1 = gr.Audio(label="🎸 Stem 1", type="filepath", interactive=False)
                        roformer_stem2 = gr.Audio(label="🥁 Stem 2", type="filepath", interactive=False)
                    roformer_files = gr.File(label="📥 Download Stems", interactive=False)
            with gr.Tab("🎚️ Auto Ensemble"):
                with gr.Group(elem_classes="dubbing-theme"):
                    gr.Markdown("### Ensemble Processing")
                    gr.Markdown("Note: If weights are not specified, equal weights (1.0) are applied. Use up to 6 models for best results.")
                    with gr.Row():
                        ensemble_audio = gr.File(label="🎧 Upload Audio or Video (WAV, MP3, MP4, MOV, etc.)", file_types=['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aiff', '.ac3', '.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.ts', '.vob'], interactive=True)
                        url_ensemble = gr.Textbox(label="🔗 Or Paste URL", placeholder="YouTube or audio/video URL", interactive=True)
                        cookies_ensemble = gr.File(label="🍪 Cookies File", file_types=[".txt"], interactive=True)
                        download_ensemble = gr.Button("⬇️ Download", variant="secondary")
                    ensemble_download_status = gr.Textbox(label="📢 Download Status", interactive=False)
                    ensemble_exclude_stems = gr.Textbox(label="🚫 Exclude Stems", placeholder="e.g., vocals, drums (comma-separated)", interactive=True)
                    with gr.Row():
                        ensemble_category = gr.Dropdown(label="📚 Category", choices=get_categories(), value="Instrumentals", interactive=True)
                        ensemble_models = gr.Dropdown(label="🛠️ Models (Max 6)", choices=get_model_choices("Instrumentals"), multiselect=True, interactive=True, allow_custom_value=True)
                    with gr.Row():
                        ensemble_seg_size = gr.Slider(32, 512, value=64, step=32, label="📏 Segment Size", interactive=True)
                        ensemble_overlap = gr.Slider(2, 10, value=8, step=1, label="🔄 Overlap", interactive=True)
                        ensemble_use_tta = gr.Dropdown(choices=["True", "False"], value="False", label="🔍 Use TTA", interactive=True)
                    ensemble_method = gr.Dropdown(label="⚙️ Ensemble Method", choices=['avg_wave', 'median_wave', 'max_wave', 'min_wave', 'avg_fft', 'median_fft', 'max_fft', 'min_fft'], value='avg_wave', interactive=True)
                    ensemble_weights = gr.Textbox(label="⚖️ Weights", placeholder="e.g., 1.0, 1.0, 1.0 (comma-separated)", interactive=True)
                    ensemble_button = gr.Button("🎛️ Run Ensemble!", variant="primary")
                    ensemble_output = gr.Audio(label="🎶 Ensemble Result", type="filepath", interactive=False)
                    ensemble_status = gr.HTML(label="📢 Status")
                    ensemble_files = gr.File(label="📥 Download Ensemble and Stems", interactive=False)
            with gr.Tab("📦 Batch Processing"):
                with gr.Group(elem_classes="dubbing-theme"):
                    gr.Markdown("### Batch Processing (Max 10 Files)")
                    gr.Markdown("Upload multiple audio files and process them all with the same model.")
                    batch_audio = gr.File(label="🎧 Upload Audio Files", file_count="multiple", file_types=['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aiff', '.ac3', '.mp4', '.mov', '.avi', '.mkv'], interactive=True)
                    with gr.Row():
                        batch_category = gr.Dropdown(label="📚 Category", choices=get_categories(), value="Vocals", interactive=True)
                        batch_model = gr.Dropdown(label="🛠️ Model", choices=get_model_choices("Vocals"), interactive=True, allow_custom_value=True)
                    with gr.Row():
                        batch_seg_size = gr.Slider(32, 512, value=64, step=32, label="📏 Segment Size", interactive=True)
                        batch_overlap = gr.Slider(2, 10, value=8, step=1, label="🔄 Overlap", interactive=True)
                    batch_pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="🎵 Pitch Shift", interactive=True)
                    batch_override_seg = gr.Dropdown(choices=["True", "False"], value="False", label="🔧 Override Segment Size", interactive=True)
                    batch_exclude = gr.Textbox(label="🚫 Exclude Stems", placeholder="e.g., vocals, drums (comma-separated)", interactive=True)
                    batch_button = gr.Button("🚀 Process Batch!", variant="primary")
                    batch_status = gr.Textbox(label="📢 Batch Status", interactive=False, lines=5)
                    batch_files = gr.File(label="📥 Download All Stems", interactive=False)
            with gr.Tab("🔧 Custom Models"):
                with gr.Group(elem_classes="dubbing-theme"):
                    gr.Markdown("### Custom Model Management")
                    gr.Markdown("Add custom models from HuggingFace or other sources by providing download URLs. The model will be automatically downloaded when used.")
                    with gr.Row():
                        custom_model_name = gr.Textbox(label="📝 Model Display Name", placeholder="e.g., My Custom Vocal Model", interactive=True)
                    with gr.Row():
                        custom_checkpoint_url = gr.Textbox(label="📦 Checkpoint URL (required)", placeholder="https://huggingface.co/.../resolve/main/model.ckpt", interactive=True)
                    with gr.Row():
                        custom_config_url = gr.Textbox(label="📄 Config URL (optional)", placeholder="https://huggingface.co/.../resolve/main/config.yaml", interactive=True)
                    with gr.Row():
                        custom_py_url = gr.Textbox(label="🐍 Custom .py URL (optional)", placeholder="https://huggingface.co/.../resolve/main/bs_roformer.py", interactive=True)
                    with gr.Row():
                        add_model_btn = gr.Button("➕ Add Model", variant="primary")
                        del_model_name = gr.Textbox(label="🗑️ Model Name to Delete", placeholder="Exact model name", interactive=True)
                        del_model_btn = gr.Button("🗑️ Delete Model", variant="stop")
                    custom_model_status = gr.Textbox(label="📢 Status", interactive=False)
                    custom_model_list = gr.Textbox(label="📋 Custom Models", interactive=False, lines=8,
                        value="\n".join([f"• {n}: {u}" for n, u in get_custom_models_list()]) or "No custom models")
        gr.HTML("<div class='footer'>Powered by Audio-Separator 🌟🎶 | Made with ❤️</div>")
        roformer_category.change(update_roformer_models, inputs=[roformer_category], outputs=[roformer_model])
        download_roformer.click(
            fn=download_audio_wrapper,
            inputs=[url_ro, cookies_ro],
            outputs=[roformer_audio, roformer_download_status]
        )
        roformer_button.click(
            fn=roformer_separator,
            inputs=[
                roformer_audio, roformer_model, roformer_seg_size, roformer_override_seg_size,
                roformer_overlap, roformer_pitch_shift, model_file_dir, output_dir,
                output_format, norm_threshold, amp_threshold, batch_size, roformer_exclude_stems
            ],
            outputs=[roformer_stem1, roformer_stem2, roformer_files]
        )
        ensemble_category.change(update_ensemble_models, inputs=[ensemble_category], outputs=[ensemble_models])
        download_ensemble.click(
            fn=download_audio_wrapper,
            inputs=[url_ensemble, cookies_ensemble],
            outputs=[ensemble_audio, ensemble_download_status]
        )
        ensemble_button.click(
            fn=auto_ensemble_process,
            inputs=[
                ensemble_audio, ensemble_models, ensemble_state, ensemble_seg_size, ensemble_overlap,
                output_format, ensemble_use_tta, model_file_dir, output_dir,
                norm_threshold, amp_threshold, batch_size, ensemble_method,
                ensemble_exclude_stems, ensemble_weights
            ],
            outputs=[ensemble_output, ensemble_status, ensemble_files, ensemble_state]
        )
        # Batch processing events
        batch_category.change(update_roformer_models, inputs=[batch_category], outputs=[batch_model])
        batch_button.click(
            fn=batch_separator,
            inputs=[
                batch_audio, batch_model, batch_seg_size, batch_override_seg, batch_overlap,
                batch_pitch_shift, model_file_dir, output_dir, output_format,
                norm_threshold, amp_threshold, batch_size, batch_exclude
            ],
            outputs=[batch_status, batch_files]
        )
        # Custom model events
        add_model_btn.click(
            fn=add_custom_model_handler,
            inputs=[custom_model_name, custom_checkpoint_url, custom_config_url, custom_py_url],
            outputs=[custom_model_status, custom_model_list, roformer_category, ensemble_category]
        )
        del_model_btn.click(
            fn=delete_custom_model_handler,
            inputs=[del_model_name],
            outputs=[custom_model_status, custom_model_list, roformer_category, ensemble_category]
        )
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Source Separation Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on")
    args = parser.parse_args()
    app = create_interface()
    try:
        app.launch(server_name="0.0.0.0", server_port=args.port, share=True)
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        raise
    finally:
        app.close()
