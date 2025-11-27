import soundfile as sf
import numpy as np
import os

def audio_to_pcm(path):
    """Load audio file and convert to PCM samples"""
    try:
        # Load audio using soundfile (supports WAV, FLAC, OGG, etc.)
        audio_data, frame_rate = sf.read(path)
        
        # Handle mono vs stereo
        if len(audio_data.shape) == 1:
            channels = 1
        else:
            channels = audio_data.shape[1]
        
        # Convert to int16 for standard PCM format
        if audio_data.dtype != np.int16:
            # If float, scale to int16 range
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        sample_width = 2  # int16 = 2 bytes per sample
        
        return audio_data, sample_width, frame_rate, channels
    
    except Exception as e:
        raise Exception(f"Failed to load audio file {path}: {e}")

def pcm_to_bytes(samples, sample_width):
    """Convert PCM numpy array to raw bytes"""
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    return samples.astype(dtype).tobytes()

def chaos_encode(byte_stream):
    """Placeholder for chaotic encryption - replace with actual encryption"""
    # TODO: Implement Chua/Lorenz/Logistic encryption
    return byte_stream

def chaos_decode(byte_stream):
    """Placeholder for chaotic decryption - replace with actual decryption"""
    # TODO: Implement corresponding decryption
    return byte_stream

def bytes_to_pcm(byte_stream, sample_width, channels):
    """Convert raw bytes back to PCM numpy array"""
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(byte_stream, dtype=dtype)
    
    if channels == 2:
        samples = samples.reshape((-1, 2))
    
    return samples

def pcm_to_audio(samples, sample_width, frame_rate, channels, out_path):
    """Convert PCM samples back to audio file"""
    try:
        # Ensure correct shape
        if channels == 2 and len(samples.shape) == 1:
            samples = samples.reshape((-1, 2))
        
        # Convert int16 back to float32 for soundfile
        if samples.dtype == np.int16:
            samples_float = samples.astype(np.float32) / 32767.0
        else:
            samples_float = samples.astype(np.float32)
        
        # Write to WAV file
        sf.write(out_path, samples_float, frame_rate, subtype='PCM_16')
        
    except Exception as e:
        raise Exception(f"Failed to save audio file {out_path}: {e}")

def pcm_pipeline(path_in, path_out):
    """Full pipeline: load -> encode -> decode -> save"""
    samples, sample_width, frame_rate, channels = audio_to_pcm(path_in)
    raw_bytes = pcm_to_bytes(samples, sample_width)
    encoded = chaos_encode(raw_bytes)
    decoded = chaos_decode(encoded)
    recovered = bytes_to_pcm(decoded, sample_width, channels)
    pcm_to_audio(recovered, sample_width, frame_rate, channels, path_out)

def chaos_encode_audio(path_in):
    """Load and encode audio for transmission"""
    samples, sample_width, frame_rate, channels = audio_to_pcm(path_in)
    raw = pcm_to_bytes(samples, sample_width)
    enc = chaos_encode(raw)
    return enc, sample_width, frame_rate, channels

def chaos_decode_audio(encoded_bytes, sample_width, frame_rate, channels, path_out):
    """Decode received audio and save to file"""
    raw = chaos_decode(encoded_bytes)
    samples = bytes_to_pcm(raw, sample_width, channels)
    pcm_to_audio(samples, sample_width, frame_rate, channels, path_out)