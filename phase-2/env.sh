# env.sh
export GPU=5090x2
export MODEL="unsloth/Qwen3.8-27B-NVFP4"
export PORT=8000

# trie uses AsyncOpenAI; OPENAI_API_KEY MUST be non-empty.
export API_KEY="lab-secret-key"
export OPENAI_API_KEY="$API_KEY"

export HF_HOME="/workspace/hf"  # Use a persistent volume
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRIE_BIN="uv run trie"

# Keep BASE_SAFE/VLLM_ARGS commented out. The wrapper handles the base config.
# export BASE_SAFE="..."
# export VLLM_ARGS="${VLLM_ARGS:-$BASE_SAFE}"
