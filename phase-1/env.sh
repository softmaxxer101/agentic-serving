export GPU=3050

export MODEL=Qwen/Qwen3-0.6B

export PORT=8000

export API_KEY=lab-secret-key

export OPENAI_API_KEY="$API_KEY"     # REQUIRED: trie uses AsyncOpenAI, must match vLLM --api-key

export HF_HOME="$HOME/hf"

export HF_HUB_ENABLE_HF_TRANSFER=1   # deprecated warning but harmless

export TRIE_BIN="uv run trie"
