# Local AI System

A private AI agent system that runs on your own laptop. One small language model is shared by four
agents (personal assistant, research, coding and document Q&A), with a local web page for chatting.
Nothing leaves your machine once the models are downloaded.

Built for: Ryzen 7 6800H, 16 GB RAM, RTX 3050 Ti with 4 GB VRAM, Ubuntu or Windows, Ollama.

## What has and has not been tested

Please read this first, because it decides how much to trust the numbers.

This project was built and tested in a cloud sandbox that has no GPU, no Ollama and 2 CPU cores.
So:

- Tested here and passing: 75 automated tests covering the agents, tool approvals, the folder
  sandbox, RAG (PDF, TXT, Markdown, DOCX), memory and summarizing, routing, the web server, the
  streaming API, the model client's fallback logic, and the benchmark script's own code paths.
  The web server was also started for real and checked over HTTP.
- The model client was tested against a fake Ollama server that follows Ollama's API format. It was
  not tested against a real Ollama install.
- Not tested here: real inference. Every speed and memory figure in this file that is not marked
  as measured is an estimate. Tokens per second, peak RAM, peak VRAM, tool-calling reliability of
  the model and the browser page in a real browser all need to be confirmed on your laptop. The
  benchmark script (step 6 below) measures all of this and writes the results to a file.

## Model choice

The first pick is Qwen3 4B in Q4_K_M, because it is the smallest model in the list that is still
good at following instructions and calling tools, and it fits in 4 GB of VRAM with room left for
the context.

| Model | Disk size | Fits in 4 GB VRAM | Notes |
|---|---|---|---|
| qwen3:4b (Q4_K_M) | about 2.5 GB | Yes, with a small margin | Main choice. Has a thinking mode, switched off by default for speed. |
| llama3.2:3b | about 2.0 GB | Yes, comfortably | Automatic fallback. Weaker at multi-step tool use. |
| qwen3:1.7b | under 2 GB | Yes | Last-resort fallback. Fast but noticeably less capable. |
| qwen3.5:4b | about 3.3 GB | Marginal | Newer, but leaves almost no room for the context on 4 GB. Not used by default. |
| qwen3:8b | roughly 5 GB | No | Would be split between GPU and CPU and run slowly. Only pulled with `--compare`. |

Sizes come from a 2026 model listing I could read. I could not open the Ollama library page itself
from the sandbox, so check the SIZE column in `ollama list` for the real numbers. I also could not
verify licenses here. As far as I know Qwen3 is Apache 2.0 and Llama 3.2 uses Meta's community
license, but please read the license on each model page before using them for anything serious.

Rough memory estimate for qwen3:4b at a 4096 context (an estimate from the published model
configuration, not a measurement):

- Weights: about 2.5 GB.
- KV cache: about 0.6 GB at f16, about 0.3 GB with the q8_0 setting used here.
- Working buffers: a few hundred MB.
- Total: around 3.1 to 3.3 GB. That fits in 4 GB, but only if nothing else is using the GPU.

I did not choose a 30B or 70B model. They can technically be loaded with offloading, but on this
laptop they would run at a speed that makes an agent system unusable.

## Install, step by step (Ubuntu)

All commands are run from the project folder.

1. Look at the machine. This only reads information and changes nothing.

       python3 scripts/inspect_hardware.py

   Check that it sees the GPU and 4 GB of VRAM. If it says no NVIDIA driver, install one first
   (`ubuntu-drivers devices`, then `sudo ubuntu-drivers install`, then reboot).

2. Create a Python environment and install the packages.

       python3 -m venv venv
       venv/bin/pip install -r requirements.txt

3. Install and tune Ollama. If Ollama is already installed it is not reinstalled. This writes a
   service settings file from the `ollama_server` block in `config/config.yaml` and restarts Ollama.

       scripts/01_setup_ollama.sh

4. Download the models. Needs internet, once.

       scripts/02_pull_models.sh

5. Optional: copy `.env.example` to `.env` and change anything you like. No keys are needed.

6. Benchmark. Close the browser and other heavy programs first.

       scripts/04_benchmark.sh --auto-tune

   It measures load time, time to first token, generation speed, a long-context test, eight
   repeated requests (to catch memory growth), whether the embedding model pushes the chat model
   out of memory, and three real agent tasks. If a setup is unstable it steps down to half the
   context and then to the next fallback model. When it finds a stable one it writes
   `benchmarks/results/recommended.env`. Copy those lines into `.env`.

7. Start the app and open the address it prints (default http://127.0.0.1:8000).

       scripts/03_start.sh

On Windows, install Ollama from its website, run `python scripts/ollama_env.py --windows`, run the
`setx` lines it prints, restart Ollama, pull the models with `ollama pull`, then use
`scripts\start.ps1`.

## How memory is handled

These four things are different, and mixing them up is the usual cause of confusion.

- Virtual memory mapping (mmap). The model file on the SSD is mapped into the program's address
  space. It costs no RAM by itself. Pages are read from disk only when the model touches them.
- RAM residency. Pages that have been touched are held in RAM by the operating system as cache.
  Under pressure the system may drop them and read them again from the SSD, which is slow but safe.
  llama.cpp uses mmap by default, and Ollama's runner is built on it, so it normally does too
  (I could not confirm in every situation, for example Ollama may turn it off in some cases, so
  check the server log if it matters to you). If you set `generation.use_mmap: false`, the app asks
  for the weights to be read fully into RAM instead.
- GPU VRAM. Layers placed on the GPU are copied into VRAM and stay there. This is the fast part.
  Ollama decides how many layers fit. Set `generation.num_gpu` to force a number, for example 20,
  if the model runs out of VRAM. Layers not on the GPU run on the CPU from RAM.
- Disk offloading. A model that does not fit in RAM can still run through mmap, but every token
  then waits on the SSD. It works, and it is far too slow for agents. This is why the model is
  kept small instead of relying on it.

What is configured, and where:

- Flash attention: `OLLAMA_FLASH_ATTENTION=1`. Required for the quantized KV cache.
- KV cache quantization: `OLLAMA_KV_CACHE_TYPE=q8_0`. Roughly halves the cache memory with a very
  small quality cost. `q4_0` saves more but can hurt answers on long contexts. It applies to every model
  the server loads.
- Context: 4096 tokens. The app also trims history to fit, so the window is not overrun.
- One request at a time: `OLLAMA_NUM_PARALLEL=1`, plus a lock in the orchestrator. Each extra
  parallel slot would multiply the KV cache.
- Keep alive: the model stays loaded for 10 minutes after use, so repeated requests do not
  reload it. The Unload button in the page frees it at once.
- Embeddings: the small embedding model (nomic-embed-text) always runs on the CPU, so it does
  not compete with the chat model for VRAM. If RAG ever feels slow, set `rag.embedder: hashing`,
  which needs no model at all and is less accurate on meaning.
- Mixture-of-experts offloading: not used. The chosen models are dense. llama.cpp supports
  keeping expert weights on the CPU (`--cpu-moe`, `--n-cpu-moe`), which is the right tool if you later try a
  small MoE model, but it needs the llama.cpp route (see the limitations).

The server settings are applied to the Ollama service, not read by the Python app, which is why
they are applied by `scripts/ollama_env.py` and why you restart Ollama after changing them.

## How it works

    web page  ->  FastAPI server  ->  orchestrator  ->  one agent  ->  shared model client  ->  Ollama
                                          |               |
                                       router        tools (files, notes, tasks, search)
                                          |
                                   memory (SQLite) and RAG store (SQLite plus numpy)

- Orchestrator: receives the request, picks an agent, runs it, tracks state (the page shows the
  current agent and step), retries once if the model server fails before producing output, saves
  the conversation, and folds old messages into a short summary when history grows.
- Router: keyword rules first (instant, no model time). Only when nothing matches does it ask the
  model for a one-word choice. You can also pick an agent by hand in the page.
- Agents are plain Python objects with a prompt and a list of allowed tools. They share one model,
  so adding agents costs no memory.
- Personal assistant: tasks, notes and long-term facts, stored in SQLite.
- Research agent: searches your documents several times and writes a report with Summary, Key
  findings, Evidence and Gaps.
- Document Q&A agent: always retrieves first, answers only from the passages, cites them as [1], [2],
  and says so when the documents do not cover the question.
- Coding agent: lists, reads and searches files, writes and edits them, runs tests.
- LangGraph was not used. The flow here is a simple loop, and the extra dependency adds weight
  without a clear benefit on a 16 GB machine.

## Security

- The web server binds to 127.0.0.1 and refuses to start on any other address unless you set
  `LAI_ALLOW_REMOTE=1`. It rejects requests with a foreign Host header and blocks cross-site POSTs.
- File tools can only reach the folders in `paths.workspace_dirs`. Paths are resolved first, so `..`
  and symlinks that point outside are refused.
- Approval is required before: overwriting an existing file, editing, deleting, running tests and
  running any command. The page shows Approve and Deny buttons. No answer within 120 seconds counts
  as Deny.
- Commands run without a shell (so `&&` and pipes do nothing) in the workspace, with a short
  timeout, and a blocked list (sudo, curl, wget, ssh and similar).
- The model never gets a general shell. Only the listed tools exist.
- Uploads accept only PDF, TXT, MD and DOCX, with a size limit.
- No keys are stored anywhere. Put any future keys in environment variables or `.env`.

## Settings

Everything lives in `config/config.yaml`. To change something without editing that file, set an
environment variable or a line in `.env` named `LAI_<SECTION>__<KEY>`, for example
`LAI_GENERATION__NUM_CTX=2048`. See `.env.example`.

## Adding a new agent

1. Create `agents/my_agent.py`.
2. Write a class like this:

       from agents.base import Agent, register_agent

       @register_agent
       class MyAgent(Agent):
           name = "myagent"
           title = "My agent"
           description = "One sentence the router can read."
           keywords = ("word one", "word two")
           tools = ("read_file", "add_note")
           system_prompt = "You are ..."

3. Restart the app. Agents are found automatically, and the new one appears in the page and in
   routing. For custom behavior, override `run(self, run)`; see `agents/research.py` for an example.

## Adding a new tool

1. Create `tools/my_tools.py`.
2. Write a function and decorate it:

       from tools.base import tool

       @tool("word_count", "Count the words in a text.",
             {"text": ("string", "The text to count.", True)})
       def word_count(ctx, text):
           return str(len(text.split()))

   To make a tool ask first, add `approval=lambda args, ctx: "Reason shown to the user"`.
3. Add `my_tools` to the import line in `tools/__init__.py`.
4. Add the tool's name to the `tools` tuple of any agent that should use it.

## If something goes wrong

Out of memory or the model will not load: run `scripts/04_benchmark.sh --auto-tune`. Or set
`LAI_GENERATION__NUM_CTX=2048`, or set `LAI_GENERATION__NUM_GPU` to a lower number, then restart the app.
The app also steps down by itself and tells you in the chat when it does.

Page says the model server is not reachable: run `systemctl status ollama`, then `ollama serve`.

Slow first answer: the first request loads the model, which can take several seconds. Later ones are quick
while it stays loaded.

The GPU is not used: run `ollama ps`. The PROCESSOR column shows the GPU and CPU split. If it says
100 percent CPU, check `nvidia-smi` and the driver.

Another program is using VRAM: `nvidia-smi` lists it. A desktop session on the NVIDIA GPU can take
several hundred MB. Close it or lower the context.

The model's answers look wrong when it uses tools: small models make mistakes with tools. Make the
request more specific, set `LAI_GENERATION__THINK=true` for harder tasks (slower), or pick the
larger model in the page for a single task.

PDF gives "no extractable text": it is probably a scan. This system does not do OCR.

Logs are in `~/local-ai-data/logs/app.log`.

## Known limitations

- No real-model numbers yet. Run the benchmark and trust it over anything written here.
- Small models are less reliable at tool use than large ones. Approvals exist partly for this reason.
- The coding agent can run commands the model suggests, but only after you approve each one. It is not
  a replacement for a full code editor.
- The web page is plain HTML and was only tested with a script, not in a real browser. Expect small
  visual rough edges.
- Only Ollama is supported as the backend. `scripts/run_llamacpp.sh` starts llama.cpp for comparison,
  but the app does not talk to it. If the benchmark ever shows llama.cpp is clearly faster on your
  machine, adding a second client in `core/llm.py` is the place to do it.
- Retrieval quality depends on the embedding model. Scanned PDFs, images and tables inside PDFs are not handled.
- Documents are searched as a whole. There is no per-conversation document filter yet.

## Upgrade ideas

- More RAM (32 GB) or a GPU with 8 GB or more. That opens up 8B models at full speed.
- A reranker for RAG, if answers cite the wrong passage.
- A llama.cpp client for MoE models with expert offloading.
- Speech, calendar and email tools, each added as a tool module.

## Project layout

    local-ai-system/
        config/          config.yaml and the settings loader
        core/            model client, mock model, approvals, system monitor
        agents/          the four agents and the base class
        orchestrator/    router and orchestrator
        rag/             loaders, chunking, embeddings, vector store, service
        memory/          SQLite store for chats, notes, tasks and facts
        tools/           file, command, notes, tasks and search tools
        app/             FastAPI server
        frontend/        web page (HTML, CSS, JavaScript, no outside files)
        benchmarks/      benchmark script and saved results
        scripts/         setup, start, hardware check and helper scripts
        tests/           automated tests (run with: venv/bin/python -m pytest)
        models/          note on where model files live. No model files go here.
        requirements.txt
        .env.example

Run the tests any time with `venv/bin/python -m pytest`.
