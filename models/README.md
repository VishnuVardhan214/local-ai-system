# Models folder

Model files are not stored in this project. Ollama keeps them in its own folder:

- Linux service install: `/usr/share/ollama/.ollama/models`
- Windows: `C:\Users\<you>\.ollama\models`

To move them to another drive or folder, uncomment `OLLAMA_MODELS` in the `ollama_server` block of
`config/config.yaml` and run `sudo python scripts/ollama_env.py --systemd`. The script creates the
folder and gives the ollama user access to it. Models already downloaded stay in the old folder until you pull them again.

For llama.cpp, keep your `.gguf` files in any folder you like and point `MODEL_GGUF` at the file.
Do not put them inside this project. `.gitignore` already skips `*.gguf` as a safety net.
