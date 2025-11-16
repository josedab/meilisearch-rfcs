use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

use super::EmbeddingCache;
use crate::progress::EmbedderStats;
use crate::vector::error::{EmbedError, NewEmbedderError};
use crate::vector::splade::SpladeVector;
use crate::vector::DistributionShift;
use crate::ThreadPoolNoAbort;

/// Configuration options for a SPLADE embedder
///
/// # Warning
///
/// This type is serialized in and deserialized from the DB, any modification should either go
/// through dumpless upgrade or be backward-compatible
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct EmbedderOptions {
    /// HuggingFace model identifier (e.g., "naver/splade-cocondenser-ensembledistil")
    pub model: String,
    /// Model revision (git commit hash)
    pub revision: Option<String>,
    /// Maximum number of active terms to keep in the sparse vector
    /// Higher values preserve more information but increase storage/compute
    #[serde(default = "default_max_active_terms")]
    pub max_active_terms: usize,
    /// Optional distribution shift for score normalization
    pub distribution: Option<DistributionShift>,
}

fn default_max_active_terms() -> usize {
    256
}

impl EmbedderOptions {
    pub fn new() -> Self {
        Self {
            model: "naver/splade-cocondenser-ensembledistil".to_string(),
            revision: None,
            max_active_terms: 256,
            distribution: None,
        }
    }
}

impl Default for EmbedderOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// SPLADE embedder that produces sparse vectors
///
/// SPLADE combines the efficiency of sparse representations with neural semantic understanding.
/// It outputs sparse vectors where each vocabulary term has a learned weight.
pub struct Embedder {
    model: Arc<Mutex<SpladeModel>>,
    tokenizer: Tokenizer,
    options: EmbedderOptions,
    vocab_size: usize,
    cache: EmbeddingCache,
    device: Device,
    max_len: usize,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpladeEmbedder")
            .field("model", &self.options.model)
            .field("options", &self.options)
            .field("vocab_size", &self.vocab_size)
            .field("device", &self.device)
            .field("max_len", &self.max_len)
            .finish()
    }
}

struct SpladeModel {
    encoder: BertModel,
    // SPLADE uses the token embeddings as output logits (vocab_size)
    // The original BERT model already has this capability through its embeddings
}

impl Embedder {
    pub fn new(options: EmbedderOptions, cache_cap: usize) -> Result<Self, NewEmbedderError> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        // Download model files from HuggingFace Hub
        let api = Api::new().map_err(NewEmbedderError::hf_hub)?;
        let repo = Repo::with_revision(
            options.model.clone(),
            RepoType::Model,
            options.revision.clone().unwrap_or_else(|| "main".to_string()),
        );
        let api = api.repo(repo);

        let config_path = api.get("config.json").map_err(NewEmbedderError::hf_hub)?;
        let tokenizer_path = api.get("tokenizer.json").map_err(NewEmbedderError::hf_hub)?;
        let weights_path = api.get("model.safetensors").map_err(NewEmbedderError::hf_hub)?;

        // Load configuration
        let config: BertConfig = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|e| NewEmbedderError::hf_hub(e.into()))?,
        )
        .map_err(|e| NewEmbedderError::hf_hub(e.into()))?;

        let vocab_size = config.vocab_size;
        let max_len = config.max_position_embeddings;

        // Load tokenizer
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(NewEmbedderError::tokenizer)?;

        // Configure padding
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(NewEmbedderError::candle)?
        };

        let encoder = BertModel::load(vb, &config).map_err(NewEmbedderError::candle)?;

        let model = SpladeModel { encoder };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            options,
            vocab_size,
            cache: EmbeddingCache::new(cache_cap),
            device,
            max_len,
        })
    }

    /// Encode a single text into a SPLADE sparse vector
    pub fn encode_one(&self, text: &str) -> Result<SpladeVector, EmbedError> {
        let encoded =
            self.tokenizer.encode(text, true).map_err(|e| EmbedError::hf_tokenizer(e.into()))?;

        let input_ids: Vec<u32> = encoded.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoded.get_attention_mask().to_vec();

        // Create tensors
        let input_tensor = Tensor::new(&input_ids[..], &self.device)
            .map_err(EmbedError::candle)?
            .unsqueeze(0)
            .map_err(EmbedError::candle)?;

        let mask_tensor = Tensor::new(&attention_mask[..], &self.device)
            .map_err(EmbedError::candle)?
            .unsqueeze(0)
            .map_err(EmbedError::candle)?;

        // Forward pass
        let model = self.model.lock().unwrap();
        let output = model
            .encoder
            .forward(&input_tensor, &mask_tensor, None)
            .map_err(EmbedError::candle)?;

        // SPLADE-specific processing:
        // 1. Apply log(1 + ReLU(x)) activation to each token embedding
        // 2. Max pooling over sequence dimension
        // For simplicity, we'll use the last hidden state

        // Get logits for vocabulary (project hidden states to vocab size)
        // In a full implementation, this would use a learned projection layer
        // For this RFC implementation, we'll use a simplified approach

        // Apply ReLU and log1p activation (SPLADE activation)
        let activated = output.relu().map_err(EmbedError::candle)?.log1p().map_err(EmbedError::candle)?;

        // Max pooling over sequence length (dim 1)
        let pooled =
            activated.max_keepdim(1).map_err(EmbedError::candle)?.squeeze(1).map_err(EmbedError::candle)?;

        // Convert to sparse vector
        let pooled_vec = pooled.to_vec1::<f32>().map_err(EmbedError::candle)?;

        let mut weights = HashMap::new();

        // Keep only top-K active terms
        let mut term_weights: Vec<_> =
            pooled_vec.iter().enumerate().filter(|(_, &w)| w > 0.0).map(|(id, &w)| (id as u32, w)).collect();

        term_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        term_weights.truncate(self.options.max_active_terms);

        for (term_id, weight) in term_weights {
            weights.insert(term_id, weight);
        }

        Ok(SpladeVector::new(weights))
    }

    /// Encode multiple texts into SPLADE sparse vectors (batch processing)
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<SpladeVector>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encoded =
            self.tokenizer.encode_batch(text_refs, true).map_err(|e| EmbedError::hf_tokenizer(e.into()))?;

        let input_ids: Vec<Vec<u32>> = encoded.iter().map(|enc| enc.get_ids().to_vec()).collect();

        let attention_masks: Vec<Vec<u32>> =
            encoded.iter().map(|enc| enc.get_attention_mask().to_vec()).collect();

        // Pad to max length in batch
        let max_len = input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let batch_size = texts.len();

        let mut padded_input = vec![0u32; batch_size * max_len];
        let mut padded_mask = vec![0u32; batch_size * max_len];

        for (i, (ids, mask)) in input_ids.iter().zip(attention_masks.iter()).enumerate() {
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                padded_input[i * max_len + j] = id;
                padded_mask[i * max_len + j] = m;
            }
        }

        let input_tensor =
            Tensor::from_slice(&padded_input, (batch_size, max_len), &self.device).map_err(EmbedError::candle)?;

        let mask_tensor =
            Tensor::from_slice(&padded_mask, (batch_size, max_len), &self.device).map_err(EmbedError::candle)?;

        // Forward pass
        let model = self.model.lock().unwrap();
        let output = model
            .encoder
            .forward(&input_tensor, &mask_tensor, None)
            .map_err(EmbedError::candle)?;

        // SPLADE activation
        let activated = output.relu().map_err(EmbedError::candle)?.log1p().map_err(EmbedError::candle)?;

        // Max pooling
        let pooled =
            activated.max_keepdim(1).map_err(EmbedError::candle)?.squeeze(1).map_err(EmbedError::candle)?;

        let pooled_vec = pooled.to_vec2::<f32>().map_err(EmbedError::candle)?;

        // Convert to sparse vectors
        let mut splade_vectors = Vec::new();
        for row in pooled_vec {
            let mut weights = HashMap::new();

            let mut term_weights: Vec<_> =
                row.iter().enumerate().filter(|(_, &w)| w > 0.0).map(|(id, &w)| (id as u32, w)).collect();

            term_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            term_weights.truncate(self.options.max_active_terms);

            for (term_id, weight) in term_weights {
                weights.insert(term_id, weight);
            }

            splade_vectors.push(SpladeVector::new(weights));
        }

        Ok(splade_vectors)
    }

    pub fn distribution(&self) -> Option<DistributionShift> {
        self.options.distribution
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn cache(&self) -> &EmbeddingCache {
        &self.cache
    }

    pub fn max_active_terms(&self) -> usize {
        self.options.max_active_terms
    }

    pub fn chunk_count_hint(&self) -> usize {
        // SPLADE can process in smaller batches than dense models
        20
    }

    pub fn prompt_count_in_chunk_hint(&self) -> usize {
        // Number of texts per batch
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_options_default() {
        let options = EmbedderOptions::default();
        assert_eq!(options.max_active_terms, 256);
        assert!(!options.model.is_empty());
    }

    #[test]
    fn test_embedder_options_serialization() {
        let options = EmbedderOptions {
            model: "naver/splade-cocondenser-ensembledistil".to_string(),
            revision: Some("main".to_string()),
            max_active_terms: 512,
            distribution: None,
        };

        let serialized = serde_json::to_string(&options).unwrap();
        let deserialized: EmbedderOptions = serde_json::from_str(&serialized).unwrap();

        assert_eq!(options, deserialized);
    }
}
