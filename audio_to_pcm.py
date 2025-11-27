import soundfile as sf
import numpy as np

# ============================================================
#  BASIC AUDIO HELPERS
# ============================================================

def audio_to_pcm(path):
    """Load audio file and convert to PCM numpy samples."""
    try:
        audio_data, frame_rate = sf.read(path)

        # Mono or stereo check
        if audio_data.ndim == 1:
            channels = 1
        else:
            channels = audio_data.shape[1]

        # Convert float audio → int16 PCM
        if audio_data.dtype != np.int16:
            if audio_data.dtype in (np.float32, np.float64):
                audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        sample_width = 2  # int16 = 2 bytes

        return audio_data, sample_width, frame_rate, channels

    except Exception as e:
        raise Exception(f"Failed to load audio file {path}: {e}")


def pcm_to_bytes(samples, sample_width):
    """Convert PCM samples → raw bytes."""
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    return samples.astype(dtype).tobytes()


def bytes_to_pcm(byte_stream, sample_width, channels):
    """Convert raw bytes → PCM numpy samples."""
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]

    samples = np.frombuffer(byte_stream, dtype=dtype)

    if channels == 2:
        samples = samples.reshape((-1, 2))

    return samples


def pcm_to_audio(samples, sample_width, frame_rate, channels, out_path):
    """Save PCM numpy buffer → WAV."""
    try:
        # Ensure stereo shape
        if channels == 2 and samples.ndim == 1:
            samples = samples.reshape((-1, 2))

        # Convert PCM int16 → float32 for soundfile
        if samples.dtype == np.int16:
            samples_float = samples.astype(np.float32) / 32767.0
        else:
            samples_float = samples.astype(np.float32)

        sf.write(out_path, samples_float, frame_rate, subtype='PCM_16')

    except Exception as e:
        raise Exception(f"Failed to save WAV file {out_path}: {e}")


# ============================================================
#  SECURE CHAOTIC AUDIO ENCODING
# ============================================================

def chaos_encrypt_pcm(samples, keystream):
    """
    XOR-encrypt PCM samples using chaotic bit keystream.
    samples: np.int16 array (mono or stereo flattened)
    keystream: string of bits "01011011..."
    """

    # Convert keystream into int array
    ks = np.array([int(b) for b in keystream], dtype=np.uint8)

    # Flatten stereo
    flat = samples.flatten()

    # Resize keystream to sample count
    ks = np.resize(ks, flat.shape)

    # XOR per sample
    enc = flat ^ ks

    return enc.astype(np.int16)


def chaos_decrypt_pcm(encoded_samples, keystream):
    """XOR-decrypt PCM samples (same as encode)."""
    ks = np.array([int(b) for b in keystream], dtype=np.uint8)
    ks = np.resize(ks, encoded_samples.shape)

    dec = encoded_samples ^ ks
    return dec.astype(np.int16)


# ============================================================
#  FUNCTIONS USED BY GUI
# ============================================================

def chaos_encode_audio(path_in, keystream):
    """
    Convert audio → PCM → XOR-encrypt → raw bytes.
    Used by SenderGUI.
    """
    samples, sample_width, frame_rate, channels = audio_to_pcm(path_in)

    encrypted_samples = chaos_encrypt_pcm(samples, keystream)
    raw = pcm_to_bytes(encrypted_samples, sample_width)

    return raw, sample_width, frame_rate, channels, samples.size   # return sample count


def chaos_decode_audio(encoded_bytes, sample_width, frame_rate, channels, keystream, path_out):
    """
    XOR-decrypt raw audio bytes → PCM → WAV.
    Used by ReceiverGUI.
    """
    encrypted = bytes_to_pcm(encoded_bytes, sample_width, channels)

    # flatten in case stereo
    encrypted = encrypted.flatten()

    decrypted = chaos_decrypt_pcm(encrypted, keystream)

    # reshape if stereo
    if channels == 2:
        decrypted = decrypted.reshape((-1, 2))

    pcm_to_audio(decrypted, sample_width, frame_rate, channels, path_out)
