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
from .utils.video import validate_video_file, extract_audio


# Load environment variables
load_dotenv()


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--anthropic-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
@click.option('--grok-key', help='Grok API key (or set GROK_API_KEY env var)')
@click.option('--brand-file', type=click.Path(exists=True), help='Path to JSON file with brand information')
@click.option('--output', default='shorts_output.json', help='Output JSON file path')
@click.option('--skip-diarization', is_flag=True, help='Skip speaker diarization')
@click.option('--model-size', default='base', help='Whisper model size (tiny, base, small, medium, large-v2, large-v3)')
@click.option('--hf-token', help='HuggingFace token for diarization (or set HF_TOKEN env var)')
def main(
    video_path: str,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    grok_key: Optional[str],
    brand_file: Optional[str],
    output: str,
    skip_diarization: bool,
    model_size: str,
    hf_token: Optional[str]
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
        
        # Extract audio from video
        click.echo("Extracting audio from video...")
        audio_path = extract_audio(video_path)
        click.echo(f"✓ Audio extracted to {audio_path}")
        
        # Transcribe audio
        click.echo("Transcribing audio with Faster Whisper...")
        with Transcriber(model_size=model_size) as transcriber:
            transcription_result = transcriber.transcribe(
                audio_path,
                word_timestamps=True,
                vad_filter=True
            )
        click.echo(f"✓ Transcription complete ({len(transcription_result['segments'])} segments)")
        click.echo(f"  Detected language: {transcription_result['language']}")
        
        # Perform speaker diarization (optional)
        if not skip_diarization:
            hf_token = hf_token or os.getenv("HF_TOKEN")
            if hf_token:
                click.echo("Performing speaker diarization...")
                try:
                    diarizer = Diarizer(hf_token=hf_token)
                    diarization_result = diarizer.diarize(audio_path, transcription_result)
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
        click.echo("Analyzing transcript and selecting shorts...")
        agent = ShortsAgent(llm)
        shorts_output = agent.select_shorts_with_retry(
            transcription_result,
            brand_info=brand_info
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
        
        # Cleanup temporary audio file
        if os.path.exists(audio_path) and audio_path.startswith('/tmp'):
            try:
                os.remove(audio_path)
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
