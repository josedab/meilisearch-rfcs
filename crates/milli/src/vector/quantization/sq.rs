use serde::{Deserialize, Serialize};

use super::QuantizationError;

/// Scalar Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SQConfig {
    /// Quantization bits (4, 8, or 16)
    pub bits: u8,
}

impl Default for SQConfig {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

pub struct ScalarQuantizer {
    /// Quantization bits (4, 8, or 16)
    bits: u8,
    /// Per-dimension min/max for normalization
    bounds: Vec<(f32, f32)>,
}

impl ScalarQuantizer {
    pub fn train(vectors: &[Vec<f32>], bits: u8) -> Result<Self, QuantizationError> {
        if vectors.is_empty() {
            return Err(QuantizationError::EmptyTrainingSet);
        }

        if bits != 4 && bits != 8 && bits != 16 {
            return Err(QuantizationError::InvalidBitsPerCode);
        }

        let dim = vectors[0].len();
        let mut bounds = Vec::with_capacity(dim);

        // Compute min/max per dimension
        for d in 0..dim {
            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for vec in vectors {
                min = min.min(vec[d]);
                max = max.max(vec[d]);
            }

            bounds.push((min, max));
        }

        Ok(Self { bits, bounds })
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let max_val = (1 << self.bits) - 1;
        let mut codes = Vec::with_capacity(vector.len());

        for (&val, &(min, max)) in vector.iter().zip(self.bounds.iter()) {
            let normalized = if max > min {
                (val - min) / (max - min)
            } else {
                0.0
            };
            let quantized = (normalized * max_val as f32).round().clamp(0.0, max_val as f32) as u8;
            codes.push(quantized);
        }

        codes
    }

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let max_val = (1 << self.bits) - 1;

        codes.iter()
            .zip(self.bounds.iter())
            .map(|(&code, &(min, max))| {
                let normalized = code as f32 / max_val as f32;
                min + normalized * (max - min)
            })
            .collect()
    }

    pub fn euclidean_distance(&self, codes_a: &[u8], codes_b: &[u8]) -> f32 {
        let max_val = (1 << self.bits) - 1;
        let max_val_f32 = max_val as f32;

        let mut distance_sq = 0.0;

        for ((&code_a, &code_b), &(min, max)) in codes_a.iter().zip(codes_b.iter()).zip(self.bounds.iter()) {
            let val_a = min + (code_a as f32 / max_val_f32) * (max - min);
            let val_b = min + (code_b as f32 / max_val_f32) * (max - min);
            let diff = val_a - val_b;
            distance_sq += diff * diff;
        }

        distance_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq_encode_decode() {
        let training_vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..10).map(|j| (i + j) as f32 / 100.0).collect()
            })
            .collect();

        let sq = ScalarQuantizer::train(&training_vectors, 8).unwrap();

        let test_vector: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
        let codes = sq.encode(&test_vector);
        let decoded = sq.decode(&codes);

        assert_eq!(decoded.len(), test_vector.len());
        assert_eq!(codes.len(), test_vector.len());

        // Check that decoded values are close to original
        for (original, decoded) in test_vector.iter().zip(decoded.iter()) {
            let diff = (original - decoded).abs();
            assert!(diff < 0.1, "Decoded value too far from original: {} vs {}", original, decoded);
        }
    }

    #[test]
    fn test_sq_invalid_bits() {
        let training_vectors: Vec<Vec<f32>> = vec![vec![1.0; 10]];
        let result = ScalarQuantizer::train(&training_vectors, 7);
        assert!(matches!(result, Err(QuantizationError::InvalidBitsPerCode)));
    }

    #[test]
    fn test_sq_empty_training_set() {
        let training_vectors: Vec<Vec<f32>> = vec![];
        let result = ScalarQuantizer::train(&training_vectors, 8);
        assert!(matches!(result, Err(QuantizationError::EmptyTrainingSet)));
    }
}
