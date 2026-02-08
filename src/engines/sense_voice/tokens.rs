use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Symbol table mapping token IDs to strings.
///
/// Loaded from a `tokens.txt` file with the format:
/// ```text
/// <blank> 0
/// <unk> 1
/// a 3
/// ...
/// ```
pub struct SymbolTable {
    id_to_sym: HashMap<i64, String>,
}

impl SymbolTable {
    /// Load symbol table from a tokens.txt file.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Format: "symbol id" or just "id" (space token)
            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() == 2 {
                if let Ok(id) = parts[0].parse::<i64>() {
                    id_to_sym.insert(id, parts[1].to_string());
                }
            }
        }

        log::info!("Loaded {} tokens from {:?}", id_to_sym.len(), path);
        Ok(Self { id_to_sym })
    }

    /// Decode all base64-encoded token values in-place.
    /// Used for FunASR Nano models where tokens are base64-encoded.
    /// Tokens that fail to decode (e.g. `<blank>`, `<unk>`) are left as-is.
    pub fn apply_base64_decode(&mut self) {
        for sym in self.id_to_sym.values_mut() {
            if let Some(decoded) = base64_decode(sym) {
                *sym = decoded;
            }
        }
    }

    /// Look up a symbol by token ID.
    pub fn get(&self, id: i64) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    /// Look up a symbol by token ID, returning empty string if not found.
    pub fn get_or_empty(&self, id: i64) -> &str {
        self.id_to_sym.get(&id).map(|s| s.as_str()).unwrap_or("")
    }
}

/// Simple base64 decoder (standard alphabet with padding).
/// Returns None if the input contains non-base64 characters.
fn base64_decode(input: &str) -> Option<String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let input = input.trim_end_matches('=');
    let chars: Vec<u8> = input.bytes().collect();
    let mut bytes = Vec::with_capacity(chars.len() * 3 / 4 + 1);

    for chunk in chars.chunks(4) {
        let a = val(*chunk.first()?)?;
        let b = val(*chunk.get(1)?)?;
        bytes.push((a << 2) | (b >> 4));

        if let Some(&c_byte) = chunk.get(2) {
            let c = val(c_byte)?;
            bytes.push((b << 4) | (c >> 2));

            if let Some(&d_byte) = chunk.get(3) {
                let d = val(d_byte)?;
                bytes.push((c << 6) | d);
            }
        }
    }

    String::from_utf8(bytes).ok()
}
