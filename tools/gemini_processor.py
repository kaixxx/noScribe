#!/usr/bin/env python3
"""
Gemini Audio Processor - Transcription and analysis using Gemini 2.5 Flash.

Processes audio files through Gemini API to get:
1. Full verbatim transcription
2. Meeting metadata (title, summary, key points, etc.)
3. Lailix-specific business coaching feedback
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# Lailix Analysis Prompt - requests full transcription + analysis
LAILIX_ANALYSIS_PROMPT = """You are transcribing and analyzing a meeting recording for Lailix, a Swiss AI consultancy.

## CRITICAL: FULL TRANSCRIPTION REQUIRED

You MUST transcribe the ENTIRE audio recording word-for-word. Do not summarize or skip any parts.
This is a COMPLETE transcription task - every word spoken must be captured.

Include speaker labels where identifiable:
- Use "Matthias:" for the Lailix consultant (usually the main speaker/host)
- Use "Client:", "Speaker 1:", "Speaker 2:", etc. for other participants
- If you can identify speakers by name from context, use their names

Include timestamps approximately every 2-3 minutes in the format [MM:SS] or [HH:MM:SS] for longer recordings.

## OUTPUT FORMAT

Return a single JSON object (no markdown code blocks, just raw JSON) with these exact fields:

{
  "transcript": "The COMPLETE verbatim transcript of the entire meeting with speaker labels and timestamps",
  "language": "Primary language detected (e.g., 'German', 'English', 'Swiss German', 'Mixed German/English')",
  "title": "A concise, descriptive title for the meeting",
  "summary": "2-3 sentence executive summary of the meeting (max 300 chars)",
  "key_points": ["Key point 1", "Key point 2", "Key point 3", ...],
  "tags": ["relevant", "tags", "for", "categorization"],
  "participants": [
    {"name": "Matthias", "role": "host", "speaking_pct": 60},
    {"name": "Client Name", "role": "participant", "speaking_pct": 40}
  ],
  "sentiment": "positive|neutral|negative|mixed",
  "meeting_type": "client_call|internal|sales|support|interview|workshop|presentation|homeschooling|other",
  "action_items": ["Action item 1", "Action item 2", ...],
  "decisions_made": ["Decision 1", "Decision 2", ...],
  "lailix_feedback": {
    "communication_score": 7,
    "communication_feedback": "Detailed feedback on clarity, articulation, pace, and effectiveness of communication...",
    "sales_score": 8,
    "sales_feedback": "Detailed feedback on discovery questions, value articulation, objection handling...",
    "strategic_alignment": "Analysis of how well this conversation aligns with Lailix positioning as an AI consultancy helping teams become AI Agent Ready...",
    "improvement_areas": ["Specific area 1 to improve", "Specific area 2 to improve"],
    "strengths": ["Strength 1 demonstrated", "Strength 2 demonstrated"],
    "overall_assessment": "Brief overall assessment of Matthias's performance in this meeting"
  }
}

## LAILIX CONTEXT (for feedback evaluation)

Lailix is a Swiss AI consultancy with these characteristics:
- Mission: Helping product teams become "AI Agent Ready"
- Approach: Connecting the "Four Voices" - Customer, Prospect, Product, Team
- Values: No slides, no hype, just real outcomes
- Target clients: Product-led tech companies (50-300 employees)
- Key offering: AI Agent Readiness Diagnostic

## LAILIX FEEDBACK CRITERIA

When evaluating Matthias's performance, consider:

1. **Discovery Quality** (communication_score component)
   - Are we asking probing questions to uncover real pain points?
   - Are we listening actively and building on responses?
   - Are we avoiding assumptions and truly understanding the client's situation?

2. **Value Articulation** (sales_score component)
   - Are we clearly explaining how Lailix helps?
   - Are we connecting our capabilities to their specific needs?
   - Are we demonstrating expertise without being arrogant?

3. **Consultative Approach** (both scores)
   - Are we asking more than telling?
   - Are we positioning ourselves as a trusted advisor?
   - Are we providing genuine value in the conversation itself?

4. **Strategic Fit** (strategic_alignment)
   - Is this the right type of client for Lailix?
   - Are they in our target segment (product-led, 50-300 employees)?
   - Do they have problems we can genuinely solve?

5. **Communication Clarity** (communication_score)
   - Is the communication clear and well-structured?
   - Are complex concepts explained simply?
   - Is the pace appropriate?

Score both communication and sales on a 1-10 scale:
- 1-3: Significant improvement needed
- 4-5: Below expectations
- 6-7: Meets expectations
- 8-9: Exceeds expectations
- 10: Exceptional performance

## IMPORTANT NOTES

- The transcript field must contain the COMPLETE meeting, not a summary
- If the audio is very long (>1 hour), still transcribe everything
- Detect and preserve the original language(s) used
- For mixed-language meetings, transcribe each segment in its original language
- If audio quality is poor in sections, indicate [inaudible] or [unclear]
"""


@dataclass
class LailixFeedback:
    """Lailix-specific coaching feedback."""
    communication_score: int
    communication_feedback: str
    sales_score: int
    sales_feedback: str
    strategic_alignment: str
    improvement_areas: list[str]
    strengths: list[str]
    overall_assessment: str = ""


@dataclass
class GeminiResult:
    """Result from Gemini audio processing."""
    # Transcript
    transcript: str
    language: str

    # Metadata
    title: str
    summary: str
    key_points: list[str]
    tags: list[str]
    participants: list[dict]
    sentiment: str
    meeting_type: str
    action_items: list[str] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)

    # Lailix feedback
    lailix_feedback: Optional[LailixFeedback] = None

    # Processing metadata
    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    model: str = "gemini-2.5-flash"

    # Raw response for debugging
    raw_response: Optional[dict] = None

    # Error (if processing failed)
    error: Optional[str] = None

    @property
    def parsed_response(self) -> dict:
        """Reconstruct the dict form that legacy consumers expect."""
        d = {
            "transcript": self.transcript,
            "language": self.language,
            "title": self.title,
            "summary": self.summary,
            "key_points": self.key_points,
            "tags": self.tags,
            "participants": self.participants,
            "sentiment": self.sentiment,
            "meeting_type": self.meeting_type,
            "action_items": self.action_items,
            "decisions_made": self.decisions_made,
        }
        if self.lailix_feedback:
            d["lailix_feedback"] = {
                "communication_score": self.lailix_feedback.communication_score,
                "communication_feedback": self.lailix_feedback.communication_feedback,
                "sales_score": self.lailix_feedback.sales_score,
                "sales_feedback": self.lailix_feedback.sales_feedback,
                "strategic_alignment": self.lailix_feedback.strategic_alignment,
                "improvement_areas": self.lailix_feedback.improvement_areas,
                "strengths": self.lailix_feedback.strengths,
                "overall_assessment": self.lailix_feedback.overall_assessment,
            }
        return d


class GeminiAudioProcessor:
    """Processor for audio files using Gemini 2.5 Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_output_tokens: int = 65536,
        temperature: float = 0.1,
        timeout_seconds: int = 600
    ):
        """Initialize the Gemini processor.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            model: Gemini model to use (default: gemini-2.5-flash)
            max_output_tokens: Maximum output tokens (default: 65536 for full transcripts)
            temperature: Generation temperature (default: 0.1 for accuracy)
            timeout_seconds: Request timeout in seconds (default: 600 for long audio)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as an environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        # Import here to allow the module to load without the dependency
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            # Use standard client - custom httpx clients cause issues
            # The Files API handles large uploads reliably
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

    # Audio duration (seconds) above which chunked processing is used.
    # Flash models hallucinate on long single-shot audio; Pro is slow+flaky.
    # 15 min is a sweet spot — stays well under model thinking-loop thresholds.
    CHUNK_THRESHOLD_SEC = 15 * 60
    CHUNK_DURATION_SEC = 15 * 60
    CHUNK_OVERLAP_SEC = 30

    def _get_duration(self, audio_path: Path) -> float:
        import subprocess
        r = subprocess.run(
            ['/opt/homebrew/bin/ffprobe', '-v', 'error',
             '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', str(audio_path)],
            capture_output=True, text=True)
        return float(r.stdout.strip() or 0)

    def _chunk_audio(self, audio_path: Path) -> list:
        """Split audio into overlapping chunks for reliable long-audio transcription.

        Returns list of (chunk_path, offset_seconds) tuples.
        Single-element list if audio doesn't need chunking.
        """
        import subprocess, tempfile
        duration = self._get_duration(audio_path)
        if duration <= self.CHUNK_THRESHOLD_SEC:
            return [(audio_path, 0.0)]

        temp_dir = Path(tempfile.gettempdir()) / f"gemini_chunks_{audio_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        chunks = []
        start = 0.0
        idx = 0
        while start < duration:
            end = min(start + self.CHUNK_DURATION_SEC, duration)
            chunk_path = temp_dir / f"chunk_{idx:02d}.mp3"
            r = subprocess.run(
                ['/opt/homebrew/bin/ffmpeg', '-y', '-i', str(audio_path),
                 '-ss', f'{start:.2f}', '-t', f'{end-start:.2f}',
                 '-c', 'copy', str(chunk_path)],
                capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg chunk failed: {r.stderr.decode()[:200]}")
            chunks.append((chunk_path, start))
            if end >= duration:
                break
            start = end - self.CHUNK_OVERLAP_SEC
            idx += 1
        logger.info(f"Split {duration/60:.1f}min audio into {len(chunks)} chunks")
        return chunks

    def _shift_timestamps(self, text: str, offset_sec: float) -> str:
        """Add offset to [MM:SS] or [HH:MM:SS] timestamps in transcript text."""
        import re
        def shift(m):
            parts = [int(x) for x in m.group(1).split(':')]
            total = sum(p * 60**(len(parts)-1-i) for i, p in enumerate(parts)) + int(offset_sec)
            h, rem = divmod(total, 3600)
            mm, ss = divmod(rem, 60)
            return f"[{h:02d}:{mm:02d}:{ss:02d}]" if h else f"[{mm:02d}:{ss:02d}]"
        return re.sub(r'\[(\d{1,2}(?::\d{2}){1,2})\]', shift, text)

    def process_audio(
        self,
        audio_path: Path,
        custom_prompt: Optional[str] = None
    ) -> GeminiResult:
        """Process an audio file with Gemini, returning transcript and analysis.

        Args:
            audio_path: Path to the audio file (MP3, WAV, etc.)
            custom_prompt: Optional custom prompt (default: LAILIX_ANALYSIS_PROMPT)

        Returns:
            GeminiResult with transcript, metadata, and Lailix feedback
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If audio is long, chunk-and-merge. Each chunk goes through the
        # single-shot path below via recursion, then we stitch.
        total_duration = self._get_duration(audio_path)
        if total_duration > self.CHUNK_THRESHOLD_SEC:
            return self._process_chunked(audio_path, custom_prompt, total_duration)

        start_time = time.time()
        prompt = custom_prompt or LAILIX_ANALYSIS_PROMPT

        file_size = audio_path.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        logger.info(f"Processing audio: {audio_path.name}")
        logger.info(f"File size: {file_size_mb:.1f} MB")

        # Determine MIME type based on extension
        suffix = audio_path.suffix.lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
        }
        mime_type = mime_types.get(suffix, 'audio/mpeg')

        # Always use Files API for audio files (more reliable for uploads)
        # The inline approach can timeout on slower connections
        logger.info(f"Uploading audio to Gemini Files API ({file_size_mb:.1f} MB)...")

        try:
            uploaded_file = self.client.files.upload(
                file=str(audio_path),
                config={"mime_type": mime_type}
            )
            logger.debug(f"Uploaded file: {uploaded_file.name}")

            # Wait for file to be processed
            wait_count = 0
            while uploaded_file.state.name == "PROCESSING":
                wait_count += 1
                if wait_count > 60:  # Max 2 minutes waiting
                    raise RuntimeError("File processing timeout")
                logger.debug(f"Waiting for file processing... ({wait_count})")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(f"File upload failed: {uploaded_file.state.name}")

            audio_content = uploaded_file
            logger.info("Audio uploaded successfully")

        except Exception as e:
            logger.error(f"Files API upload failed: {e}")
            # Fallback to inline data for small files
            if file_size_mb < 5:
                logger.info("Falling back to inline data...")
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                audio_content = self.types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                )
                uploaded_file = None
            else:
                raise

        # Generate content with audio, using streaming + retry to handle
        # server-side disconnects that occur on long audio requests.
        # Note: Do NOT use response_mime_type="application/json" — it causes
        # RemoteProtocolError on audio >15min. The prompt already requests JSON.
        logger.info(f"Generating transcript and analysis with {self.model}...")
        response_text, response = self._generate_with_retry(
            prompt=prompt,
            audio_content=audio_content,
            max_attempts=3,
        )

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.1f}s")

        # Parse response
        result = self._parse_response(response, processing_time, response_text)

        # Clean up uploaded file if we used Files API
        if uploaded_file is not None:
            try:
                self.client.files.delete(name=uploaded_file.name)
                logger.debug("Cleaned up uploaded file")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")

        return result

    def _generate_with_retry(self, prompt: str, audio_content: Any,
                              max_attempts: int = 3) -> tuple:
        """Stream generate_content with retry on transient disconnects.

        Gemini's server disconnects with RemoteProtocolError on ~30% of
        long-audio requests. Streaming + retry makes the pipeline reliable.

        Returns: (text, response) where response has usage_metadata and text.
        """
        last_error = None
        for attempt in range(1, max_attempts + 1):
            chunks = []
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=[prompt, audio_content],
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                final_response = None
                for chunk in stream:
                    if chunk.text:
                        chunks.append(chunk.text)
                    final_response = chunk  # last chunk carries usage_metadata
                text = ''.join(chunks)
                if not text:
                    raise RuntimeError("Empty response from Gemini")
                return text, final_response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini attempt {attempt}/{max_attempts} failed after "
                    f"{len(''.join(chunks))} chars: {e}"
                )
                if attempt < max_attempts:
                    wait = 10 * attempt
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(f"All {max_attempts} attempts failed: {last_error}")

    def _process_chunked(self, audio_path: Path, custom_prompt: Optional[str],
                          total_duration: float) -> GeminiResult:
        """Chunk long audio, transcribe each chunk, merge results.

        Analysis (title/summary/action_items/feedback) comes from the first
        chunk only — reliable enough for routing/logging, and the Claude
        meeting-actions skill redoes its own deeper analysis from the full
        merged transcript anyway.
        """
        start_time = time.time()
        chunks = self._chunk_audio(audio_path)
        logger.info(f"Processing {len(chunks)} chunks of {audio_path.name}")

        merged_transcript_parts = []
        input_tokens_total = 0
        output_tokens_total = 0
        first_result = None
        last_overlap_tail = None  # used to dedupe across chunk boundaries

        for i, (chunk_path, offset) in enumerate(chunks):
            logger.info(f"  Chunk {i+1}/{len(chunks)} at offset {offset/60:.1f}min")
            try:
                # Recurse with single-shot path (chunks are all ≤15min)
                chunk_result = self.process_audio(chunk_path, custom_prompt)
            except Exception as e:
                logger.error(f"  Chunk {i+1} failed: {e}, continuing with others")
                merged_transcript_parts.append(f"\n\n[CHUNK {i+1} FAILED: {e}]\n\n")
                continue

            input_tokens_total += chunk_result.input_tokens
            output_tokens_total += chunk_result.output_tokens
            if first_result is None:
                first_result = chunk_result

            # Shift timestamps in chunk transcript by the chunk's start offset
            shifted = self._shift_timestamps(chunk_result.transcript, offset)
            merged_transcript_parts.append(shifted)

        # Merge transcripts (naive concat — overlap region appears twice but
        # timestamp ordering lets downstream consumers dedupe if needed)
        merged_transcript = '\n\n'.join(t for t in merged_transcript_parts if t.strip())
        processing_time = time.time() - start_time
        logger.info(f"Chunked processing complete in {processing_time:.1f}s")

        # Cleanup chunk files
        import shutil
        if chunks and chunks[0][0] != audio_path:
            try:
                shutil.rmtree(chunks[0][0].parent)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        # Build result: use first chunk's analysis for metadata
        if first_result is None:
            return GeminiResult(
                transcript=merged_transcript,
                language="unknown",
                title="All chunks failed",
                summary="All transcription chunks failed",
                key_points=[], tags=["chunked_processing_error"],
                participants=[], sentiment="neutral", meeting_type="other",
                processing_time_seconds=processing_time,
                error="All chunks failed",
            )

        return GeminiResult(
            transcript=merged_transcript,
            language=first_result.language,
            title=first_result.title,
            summary=first_result.summary,
            key_points=first_result.key_points,
            tags=first_result.tags + ["chunked"],
            participants=first_result.participants,
            sentiment=first_result.sentiment,
            meeting_type=first_result.meeting_type,
            action_items=first_result.action_items,
            decisions_made=first_result.decisions_made,
            lailix_feedback=first_result.lailix_feedback,
            input_tokens=input_tokens_total,
            output_tokens=output_tokens_total,
            audio_duration_seconds=total_duration,
            processing_time_seconds=processing_time,
            model=self.model,
        )

    def _parse_response(self, response: Any, processing_time: float,
                         text: Optional[str] = None) -> GeminiResult:
        """Parse Gemini response into GeminiResult.

        Args:
            response: Gemini API response
            processing_time: Time taken to process
            text: Pre-collected text (from streaming). If None, uses response.text.

        Returns:
            GeminiResult with parsed data
        """
        # Get token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        # Parse JSON from response text
        if text is None:
            text = response.text
        logger.debug(f"Response length: {len(text)} chars")

        # Clean potential markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {text[:500]}...")
            # Return minimal result with raw text
            return GeminiResult(
                transcript=text,
                language="unknown",
                title="Parse Error",
                summary="Failed to parse Gemini response",
                key_points=[],
                tags=["parse_error"],
                participants=[],
                sentiment="neutral",
                meeting_type="other",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                processing_time_seconds=processing_time,
                raw_response={"raw_text": text, "parse_error": str(e)}
            )

        # Extract Lailix feedback
        lailix_data = data.get("lailix_feedback", {})
        lailix_feedback = None
        if lailix_data:
            lailix_feedback = LailixFeedback(
                communication_score=lailix_data.get("communication_score", 0),
                communication_feedback=lailix_data.get("communication_feedback", ""),
                sales_score=lailix_data.get("sales_score", 0),
                sales_feedback=lailix_data.get("sales_feedback", ""),
                strategic_alignment=lailix_data.get("strategic_alignment", ""),
                improvement_areas=lailix_data.get("improvement_areas", []),
                strengths=lailix_data.get("strengths", []),
                overall_assessment=lailix_data.get("overall_assessment", "")
            )

        # Validate enums
        valid_sentiments = ["positive", "neutral", "negative", "mixed"]
        sentiment = data.get("sentiment", "neutral")
        if sentiment not in valid_sentiments:
            sentiment = "neutral"

        valid_types = [
            "client_call", "internal", "sales", "support", "interview",
            "workshop", "presentation", "homeschooling", "other"
        ]
        meeting_type = data.get("meeting_type", "other")
        if meeting_type not in valid_types:
            meeting_type = "other"

        return GeminiResult(
            transcript=data.get("transcript", ""),
            language=data.get("language", "unknown"),
            title=data.get("title", "Untitled Meeting"),
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            tags=data.get("tags", []),
            participants=data.get("participants", []),
            sentiment=sentiment,
            meeting_type=meeting_type,
            action_items=data.get("action_items", []),
            decisions_made=data.get("decisions_made", []),
            lailix_feedback=lailix_feedback,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time_seconds=processing_time,
            model=self.model,
            raw_response=data
        )

    def estimate_cost(self, audio_duration_seconds: float) -> dict:
        """Estimate the cost of processing audio.

        Args:
            audio_duration_seconds: Duration of audio in seconds

        Returns:
            Dict with estimated tokens and cost
        """
        # Gemini: 32 tokens per second of audio
        audio_tokens = int(audio_duration_seconds * 32)

        # Estimate output tokens (varies by meeting length)
        # Rough estimate: ~500 tokens per minute of transcript + 1000 for analysis
        estimated_output = int((audio_duration_seconds / 60) * 500 + 1000)

        # Gemini 2.5 Flash pricing (as of Jan 2025)
        # Input: $0.15 per 1M tokens (text), audio may vary
        # Output: $0.60 per 1M tokens
        input_cost = audio_tokens / 1_000_000 * 0.15
        output_cost = estimated_output / 1_000_000 * 0.60

        return {
            "audio_tokens": audio_tokens,
            "estimated_output_tokens": estimated_output,
            "estimated_input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "estimated_total_cost_usd": input_cost + output_cost
        }


def process_audio_file(
    audio_path: Path,
    api_key: Optional[str] = None
) -> GeminiResult:
    """Convenience function to process an audio file.

    Args:
        audio_path: Path to the audio file
        api_key: Optional API key (default: from environment)

    Returns:
        GeminiResult with transcript and analysis
    """
    processor = GeminiAudioProcessor(api_key=api_key)
    return processor.process_audio(audio_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python gemini_processor.py <audio_file.mp3>")
        print("\nEnvironment: Set GEMINI_API_KEY")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    # Estimate cost first
    from audio_converter import get_audio_duration
    duration = get_audio_duration(audio_file)
    processor = GeminiAudioProcessor()
    cost = processor.estimate_cost(duration)

    print(f"\nAudio duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Estimated audio tokens: {cost['audio_tokens']:,}")
    print(f"Estimated cost: ${cost['estimated_total_cost_usd']:.4f}")

    # Ask for confirmation
    response = input("\nProceed with transcription? [y/N] ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Process
    result = processor.process_audio(audio_file)

    print(f"\n{'='*60}")
    print(f"Title: {result.title}")
    print(f"Language: {result.language}")
    print(f"Meeting type: {result.meeting_type}")
    print(f"Sentiment: {result.sentiment}")
    print(f"\nSummary: {result.summary}")
    print(f"\nKey points:")
    for point in result.key_points:
        print(f"  - {point}")

    if result.lailix_feedback:
        print(f"\n{'='*60}")
        print("LAILIX FEEDBACK")
        print(f"Communication Score: {result.lailix_feedback.communication_score}/10")
        print(f"Sales Score: {result.lailix_feedback.sales_score}/10")
        print(f"\nStrengths:")
        for s in result.lailix_feedback.strengths:
            print(f"  + {s}")
        print(f"\nImprovement Areas:")
        for i in result.lailix_feedback.improvement_areas:
            print(f"  - {i}")

    print(f"\n{'='*60}")
    print(f"Processing time: {result.processing_time_seconds:.1f}s")
    print(f"Input tokens: {result.input_tokens:,}")
    print(f"Output tokens: {result.output_tokens:,}")

    # Save transcript to file
    output_file = audio_file.with_suffix('.transcript.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.raw_response, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full result to: {output_file}")
