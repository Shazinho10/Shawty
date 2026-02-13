"""Main CLI entry point for Shortie"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
import click
from dotenv import load_dotenv

from .asr import Transcriber, Diarizer
from .llm import get_llm_provider, ShortsAgent
from .models.brand import BrandInfo
from .models.output import ShortsOutput
from .utils.video import validate_video_file, extract_audio, get_video_duration, extract_audio_chunks


# Load environment variables
load_dotenv()

print("DEBUG: STARTING WITH PATCHES v2", flush=True)

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--anthropic-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
@click.option('--grok-key', help='Grok API key (or set GROK_API_KEY env var)')
@click.option('--brand-file', type=click.Path(exists=True), help='Path to JSON file with brand information')
@click.option('--output', default='shorts_output.json', help='Output JSON file path')
@click.option('--skip-diarization', is_flag=True, help='Skip speaker diarization')
@click.option('--model-size', default='tiny', help='Whisper model size (tiny, base, small, medium, large-v2, large-v3). Default is "tiny" for speed.')
@click.option('--hf-token', help='HuggingFace token for diarization (or set HF_TOKEN env var)')
@click.option('--cpu-threads', default=4, help='Number of threads for CPU inference')
@click.option('--chunk-duration', default=600, help='Audio chunk duration in seconds (default 600 = 10 min). Set lower to test chunking on short videos.')
@click.option('--target-shorts', default=None, type=int, help='Number of shorts to select (default: 5, or 15 for 60+ min videos)')
@click.option('--min-gap-seconds', default=90, type=int, help='Minimum spacing between clips by midpoint (seconds)')
def main(
    video_path: str,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    grok_key: Optional[str],
    brand_file: Optional[str],
    output: str,
    skip_diarization: bool,
    model_size: str,
    hf_token: Optional[str],
    cpu_threads: int,
    chunk_duration: int,
    target_shorts: Optional[int],
    min_gap_seconds: int
):

    """
    Process a video file to generate YouTube shorts.
    
    VIDEO_PATH: Path to the video file to process
    """
    try:
        # Validate video file
        click.echo("Validating video file...")
        validate_video_file(video_path)
        click.echo("✓ Video file validated")
        
        # Load brand information if provided
        brand_info = None
        if brand_file:
            click.echo(f"Loading brand information from {brand_file}...")
            with open(brand_file, 'r') as f:
                brand_data = json.load(f)
                brand_info = BrandInfo(**brand_data).model_dump()
            click.echo("✓ Brand information loaded")
        
        # Get video duration to decide chunking strategy
        click.echo("Getting video duration...")
        duration = get_video_duration(video_path)
        if target_shorts is None:
            target_shorts = 15 if duration >= 3600 else 5
        click.echo(f"✓ Video duration: {duration:.1f}s ({duration/60:.1f} min)")
        
        use_chunking = duration > chunk_duration
        audio_paths_to_cleanup = []
        
        if use_chunking:
            # ── CHUNKED PATH (long videos) ──
            num_chunks = -(-int(duration) // chunk_duration)  # ceiling division
            click.echo(f"Video is long — splitting into {num_chunks} chunks of {chunk_duration}s each")
            
            # Extract audio chunks
            click.echo("Extracting audio chunks...")
            def on_extract_progress(current, total):
                click.echo(f"  Extracting chunk {current}/{total}...")
            
            chunks = extract_audio_chunks(
                video_path,
                chunk_duration=float(chunk_duration),
                on_progress=on_extract_progress
            )
            click.echo(f"✓ Extracted {len(chunks)} audio chunks")
            
            # Track chunk files for cleanup
            audio_paths_to_cleanup = [path for path, _ in chunks]
            
            # Transcribe all chunks
            click.echo(f"Transcribing audio chunks with Faster Whisper (Model: {model_size}, Threads: {cpu_threads})...")
            with Transcriber(
                model_size=model_size,
                compute_type="int8",
                cpu_threads=cpu_threads
            ) as transcriber:
                def on_transcribe_progress(current, total):
                    click.echo(f"  Transcribing chunk {current}/{total}...")
                
                transcription_result = transcriber.transcribe_chunks(
                    chunks,
                    word_timestamps=True,
                    vad_filter=True,
                    on_progress=on_transcribe_progress
                )
            click.echo(f"✓ Transcription complete ({len(transcription_result['segments'])} segments from {len(chunks)} chunks)")
            click.echo(f"  Detected language: {transcription_result['language']}")
            
            # For diarization, we need a full audio file
            if not skip_diarization:
                click.echo("ℹ Diarization with chunked audio: extracting full audio for speaker detection...")
                full_audio_path = extract_audio(video_path)
                audio_paths_to_cleanup.append(full_audio_path)
                audio_path_for_diarization = full_audio_path
            else:
                audio_path_for_diarization = None
        else:
            # ── SINGLE-FILE PATH (short videos, original behavior) ──
            click.echo("Extracting audio from video...")
            audio_path = extract_audio(video_path)
            audio_paths_to_cleanup.append(audio_path)
            click.echo(f"✓ Audio extracted to {audio_path}")
            
            click.echo(f"Transcribing audio with Faster Whisper (Model: {model_size}, Threads: {cpu_threads})...")
            with Transcriber(
                model_size=model_size,
                compute_type="int8",
                cpu_threads=cpu_threads
            ) as transcriber:
                transcription_result = transcriber.transcribe(
                    audio_path,
                    word_timestamps=True,
                    vad_filter=True
                )
            click.echo(f"✓ Transcription complete ({len(transcription_result['segments'])} segments)")
            click.echo(f"  Detected language: {transcription_result['language']}")
            
            audio_path_for_diarization = audio_path
        
        # Perform speaker diarization (optional)
        if not skip_diarization and audio_path_for_diarization:
            hf_token = hf_token or os.getenv("HF_TOKEN")
            if hf_token:
                click.echo("Performing speaker diarization...")
                try:
                    diarizer = Diarizer(hf_token=hf_token)
                    diarization_result = diarizer.diarize(audio_path_for_diarization, transcription_result)
                    if diarization_result:
                        transcription_result = diarizer.merge_with_transcription(
                            transcription_result,
                            diarization_result
                        )
                        click.echo(f"✓ Diarization complete ({transcription_result.get('num_speakers', 0)} speakers)")
                    else:
                        click.echo("⚠ Diarization unavailable (no HF token)")
                except Exception as e:
                    click.echo(f"⚠ Diarization failed: {e}. Continuing without speaker labels...")
            else:
                click.echo("ℹ Skipping diarization (HF_TOKEN not provided - optional feature)")
        else:
            if skip_diarization:
                click.echo("Skipping speaker diarization (--skip-diarization flag)")
        
        # Initialize LLM provider
        click.echo("Initializing LLM provider...")
        llm, provider = get_llm_provider(
            openai_key=openai_key,
            anthropic_key=anthropic_key,
            grok_key=grok_key
        )
        click.echo(f"✓ Using LLM provider: {provider.value}")
        
        # Create shorts agent
        if use_chunking:
            click.echo("Analyzing transcript in chunks and selecting shorts...")
            def on_llm_progress(current, total):
                click.echo(f"  Analyzing transcript chunk {current}/{total}...")
        else:
            click.echo("Analyzing transcript and selecting shorts...")
            on_llm_progress = None
        
        agent = ShortsAgent(llm)
        shorts_output = agent.select_shorts_with_retry(
            transcription_result,
            brand_info=brand_info,
            use_chunking=use_chunking,
            chunk_minutes=8.0,
            on_progress=on_llm_progress,
            target_shorts=target_shorts,
            min_gap_seconds=float(min_gap_seconds)
        )
        click.echo(f"✓ Selected {shorts_output.total_shorts} shorts")
        
        # Save output
        click.echo(f"Saving output to {output}...")
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(shorts_output.model_dump(), f, indent=2)
        click.echo(f"✓ Output saved to {output}")
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("SHORTS SUMMARY")
        click.echo("="*50)
        for i, short in enumerate(shorts_output.shorts, 1):
            click.echo(f"\n{i}. {short.title}")
            click.echo(f"   Time: {short.start_time:.2f}s - {short.end_time:.2f}s")
            click.echo(f"   Reason: {short.reason}")
        click.echo("\n" + "="*50)
        
        # Cleanup temporary audio files
        for audio_file in audio_paths_to_cleanup:
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except Exception:
                    pass
        
        click.echo("\n✓ Processing complete!")
        
    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
