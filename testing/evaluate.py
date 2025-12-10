import os
import torch
import pandas as pd
import numpy as np
import whisper
import jiwer
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm
from pathlib import Path

# Path to the folder containing generated wav files
GENERATED_DIR = "/gscratch/realitylab/huyhuynh/TTS/evaluation/lora_results"

# Path to the folder containing the ground-truth reference wav files
REFERENCE_DIR = "/gscratch/realitylab/huyhuynh/TTS/evaluation/test_data/wavs"

# Path to metadata file (for Ground Truth text)
METADATA_PATH = "/gscratch/realitylab/huyhuynh/TTS/evaluation/test_data/metadata.csv"

# Language for Whisper model
LANGUAGE = "zh" 

def load_models():
    print("Loading Whisper Model (for CER)...")
    asr_model = whisper.load_model("base") 
    
    print("Loading Resemblyzer (for SECS)...")
    speaker_encoder = VoiceEncoder()
    
    return asr_model, speaker_encoder

def normalize_text(text):
    """
    Simple normalization for Chinese CER.
    Removes punctuation and spaces to focus purely on character correctness.
    """
    import re
    # Remove punctuation and special characters, keep only Chinese, numbers, and letters
    # This regex matches everything NOT (^) a word character or whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(" ", "").replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    return text.lower()

def compute_secs(encoder, ref_path, gen_path):
    """Computes Cosine Similarity between two audio files."""
    try:
        # Preprocess loads the wav and normalizes volume
        wav_ref = preprocess_wav(ref_path)
        wav_gen = preprocess_wav(gen_path)
        
        # Get embeddings (d-vectors)
        # We assume single speaker segments, so we take the embedding of the whole utterance
        embed_ref = encoder.embed_utterance(wav_ref)
        embed_gen = encoder.embed_utterance(wav_gen)
        
        # Compute Cosine Similarity (Dot product of normalized vectors)
        # Resemblyzer embeddings are already L2-normalized
        similarity = np.inner(embed_ref, embed_gen)
        return similarity
    except Exception as e:
        print(f"Error computing SECS for {gen_path}: {e}")
        return None

def main():
    # 1. Setup
    asr_model, speaker_encoder = load_models()
    
    # 2. Load Metadata (Ground Truth)
    print(f" > Reading metadata from {METADATA_PATH}...")
    try:
        df = pd.read_csv(METADATA_PATH, sep="|", header=None, names=["id", "text", "normalized_text"])
    except:
        df = pd.read_csv(METADATA_PATH, sep="|", header=None, names=["id", "text"])
    
    results = []
    
    # 3. Iterate through generated files
    print("Starting Evaluation...")
    generated_files = list(Path(GENERATED_DIR).glob("*_test.wav"))
    
    if len(generated_files) == 0:
        print(f"ERROR: No files found in {GENERATED_DIR}")
        return

    for gen_file in tqdm(generated_files):
        # Extract File ID (assuming filename is "LJ001-0001_generated.wav")
        file_id = gen_file.name.replace("_test.wav", "")
        
        # Find corresponding row in metadata
        row = df[df['id'] == file_id]
        if row.empty:
            continue
            
        ground_truth_text = str(row.iloc[0]['text'])
        
        # Find Reference Audio
        ref_path = os.path.join(REFERENCE_DIR, f"{file_id}.wav")
        if not os.path.exists(ref_path):
            continue

        # --- METRIC 1: CER (Character Error Rate) ---
        # Transcribe
        transcription = asr_model.transcribe(str(gen_file), language=LANGUAGE)["text"]
        
        # Normalize
        norm_truth = normalize_text(ground_truth_text)
        norm_trans = normalize_text(transcription)
        
        if len(norm_truth) > 0:
            cer_score = jiwer.cer(norm_truth, norm_trans)
        else:
            cer_score = 0.0 # avoid division by zero if text is empty

        # --- METRIC 2: SECS (Speaker Similarity) ---
        secs_score = compute_secs(speaker_encoder, ref_path, gen_file)

        # Log Result
        results.append({
            "id": file_id,
            "cer": cer_score,
            "secs": secs_score,
            "text_truth": norm_truth,
            "text_pred": norm_trans
        })

    # 4. Final Report
    if not results:
        print("No valid comparisons made.")
        return

    results_df = pd.DataFrame(results)
    
    avg_cer = results_df['cer'].mean()
    avg_secs = results_df['secs'].mean()
    
    print("\n" + "="*30)
    print(f" FINAL RESULTS ({len(results_df)} samples)")
    print("="*30)
    print(f" Average CER:  {avg_cer:.4f}  (Lower is better)")
    print(f" Average SECS: {avg_secs:.4f} (Higher is better, range 0.0-1.0)")
    print("="*30)
    
    # Save detailed CSV
    save_path = os.path.join(os.path.dirname(GENERATED_DIR), "evaluation_metrics.csv")
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to: {save_path}")

if __name__ == "__main__":
    main()