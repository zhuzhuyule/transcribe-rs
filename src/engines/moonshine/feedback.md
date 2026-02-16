# Moonshine Streaming Implementation Critique

Overall, this is a **solid implementation** that correctly follows the C++ reference. Here's my detailed feedback:

---

## ✅ What's Done Well

### 1. Correct Architecture Mapping
The 5-session pipeline (frontend → encoder → adapter → cross_kv → decoder_kv) is correctly implemented. The offline `generate()` method properly orchestrates all components.

### 2. State Management
`StreamingState` correctly mirrors the C++ `MoonshineStreamingState` with all required buffers:
- Frontend state (sample_buffer, conv buffers, frame_count)
- Feature accumulator
- Memory buffer
- Self-attention and cross-attention KV caches

### 3. Binary Tokenizer
The parsing logic correctly handles the variable-length encoding:
```rust
let byte_count = if first_byte < 128 {
    first_byte as usize
} else {
    (second_byte as usize * 128) + first_byte as usize - 128
};
```

### 4. Streaming-Ready Design
State is externalized via `create_state()`, and chunk-based methods (`process_audio_chunk`, `encode`, `decode_step`) are exposed for future streaming integration.

### 5. Encoder Sliding Window
The left context calculation `(16 * depth)` and window slicing logic matches the C++ implementation exactly.

---

## ⚠️ Issues to Address

### 1. Missing `max_seq_len` in Config File
The config JSON doesn't include `max_seq_len`, but the code handles this correctly with a default of 448. This is fine, but the `frontend_state_shapes` field in the JSON is **ignored**. Consider validating that hardcoded buffer sizes (79, 4) match the config.

### 2. Error Type Misuse
In `streaming_model.rs`, `MoonshineError::OutputNotFound` is used for non-output errors:
```rust
// This is semantically wrong:
return Err(MoonshineError::OutputNotFound(
    "Memory is empty, cannot compute cross K/V".to_string(),
));
```
Consider adding a `MoonshineError::InvalidState` or `MoonshineError::EmptyMemory` variant.

### 3. Clone on Large Vectors
Several places clone large vectors unnecessarily:
```rust
// streaming_model.rs:243
state.sample_buffer.clone()  // 79 floats - OK

// streaming_model.rs:324
state.memory.clone()  // Can be very large!
```
For `compute_cross_kv`, consider passing a slice reference instead of cloning.

### 4. Unused `max_length` in `StreamingInferenceParams`
```rust
fn transcribe_samples(
    &mut self,
    samples: Vec<f32>,
    _params: Option<Self::InferenceParams>,  // underscore prefix = unused
) -> Result<TranscriptionResult, ...>
```
The `max_length` parameter is ignored. Either implement it or remove it from the struct.

### 5. Partial Audio Chunk Handling (CRITICAL)
In `generate()`:
```rust
let chunk_count = samples.len() / CHUNK_SIZE;
for chunk_index in 0..chunk_count {
    // ...
}
```
If `samples.len()` is not a multiple of `CHUNK_SIZE`, the remaining samples are **silently dropped**. The C++ implementation doesn't have this issue because it processes the entire buffer. Consider:
```rust
for chunk in samples.chunks(CHUNK_SIZE) {
    self.process_audio_chunk(&mut state, chunk)?;
}
```

### 6. Tokenizer Special Token Handling
The binary tokenizer skips tokens wrapped in `<...>`:
```rust
if bytes.len() > 2 && bytes[0] == b'<' && bytes[bytes.len() - 1] == b'>' {
    continue;
}
```
This is correct for BOS/EOS, but what about other special tokens like `<unk>`? The C++ uses `skipSpecials` parameter - should this be configurable?

---

## 💡 Suggestions

### 1. Add Validation at Model Load
Compare config values against actual ONNX input shapes:
```rust
// Verify frontend expects the right sample_buffer size
// Verify encoder expects the right encoder_dim
```

### 2. Consider Zero-Copy Where Possible
The `ndarray` + `TensorRef` approach requires copying data. For performance-critical paths, consider using `ort::value::Value::from_array` with borrowed data where possible.

### 3. Add Logging for Debugging
The `log::trace!` calls are good, but consider adding at key points:
- When cross KV is computed/invalidated
- When decoder cache is reset
- Total feature count after frontend processing

### 4. Test Coverage
The implementation needs tests for:
- Config loading with valid/invalid JSON
- Binary tokenizer parsing edge cases
- State reset behavior
- Chunk boundary handling (audio not divisible by 1280)

---

## 📋 Minor Nits

1. **`decoder_reset` in `StreamingModel`** takes `&self` but modifies `state` - this is fine, just unusual signature style.

2. **`d_model_frontend` vs `encoder_dim`** - In the tiny model config they're both 320, but semantically they're different. The code uses them correctly, but naming could be clearer.

3. **`StreamingModelParams::num_threads = 0`** default - Document that 0 means "let ORT decide" vs 1 = single-threaded.

---

## 🏁 Verdict

**7.5/10** - The implementation is functionally correct and well-structured for streaming. The main issues are:

1. **Silent dropping of partial audio chunks** (critical - will cause transcription errors on non-aligned audio lengths)
2. Unused `max_length` parameter
3. Minor inefficiencies with cloning large vectors

These are all fixable without architectural changes. The core logic correctly mirrors the C++ reference.