# MedAgent — Multi-Agent Clinical Decision Support System

A multi-agent clinical decision support system built with **LangGraph** and **MCP** (Model Context Protocol). MedAgent orchestrates four specialized AI agents — triage, literature search, clinical reasoning, and report generation — to process clinical vignettes and produce structured SOAP notes with differential diagnoses.

## Architecture

```
Patient Vignette
       │
       ▼
┌─────────────┐     ESI 1-2      ┌──────────────┐
│   Triage    │────────────────►  │ Human Review  │
│   Agent     │                   │ (Checkpoint)  │
│  (ESI 1-5)  │                   └──────┬───────┘
└──────┬──────┘                          │
       │ ESI 3-5                         │
       ▼                                 ▼
┌─────────────┐          ┌──────────────────────────┐
│ Literature  │◄─────────┤  MedLlama RAG Pipeline   │
│   Agent     │  MCP     │  (Project 1 - MedLlama)  │
└──────┬──────┘          └──────────────────────────┘
       │
       ▼ (ESI 1-3 only)
┌─────────────┐     ┌───────────────────┐
│ Reasoning   │◄────┤ Drug Interaction  │
│   Agent     │ MCP │ Server (RxNorm)   │
│ (Diff. Dx)  │     └───────────────────┘
└──────┬──────┘     ┌───────────────────┐
       │       ◄────┤ Lab Interpreter   │
       │        MCP │ Server            │
       ▼            └───────────────────┘
┌─────────────┐
│   Report    │
│   Agent     │
│ (SOAP Note) │
└─────────────┘
       │
       ▼
   SOAP Note + Differential Diagnosis
```

## Features

- **4 Specialized Agents**: Triage (ESI classification), Literature (RAG-powered search), Reasoning (chain-of-thought differential diagnosis), Report (SOAP note generation)
- **3 MCP Servers**: Drug interaction checking (NLM RxNorm API), lab value interpretation, MedLlama RAG bridge
- **LangGraph Orchestration**: Conditional routing based on ESI severity, human-in-the-loop for emergency cases
- **Real-time Gradio Demo**: Interactive UI with graph visualization and tabbed clinical outputs
- **Comprehensive Evaluation**: 50 synthetic clinical test cases with automated metrics

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (for agent orchestration)
- [Optional] MedLlama server running at `localhost:8090` (for RAG)

### Installation

```bash
git clone https://github.com/wwhite777/MedAgent.git
cd MedAgent
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run the Demo

```bash
python -m src.ui.medagent-gradio-app
# Open http://localhost:7860
```

### Run with Docker

```bash
cd docker
docker-compose up
# Open http://localhost:7860
```

## MCP Servers

| Server | Tool | Transport | External API |
|--------|------|-----------|-------------|
| `medllama_rag` | `search_literature(query, top_k)` | stdio | MedLlama FastAPI (Project 1) |
| `drug_interaction` | `check_drug_interactions(drugs)` | stdio | NLM RxNorm REST API |
| `lab_interpreter` | `interpret_labs(labs)` | stdio | Local reference ranges DB |

## Evaluation Results

| Metric | Target | Result |
|--------|--------|--------|
| Triage Accuracy (ESI level) | >85% | — |
| Differential Dx Top-3 | >70% | — |
| Tool Calling Accuracy | >90% | — |
| Drug Interaction Detection | >95% | — |
| Report Completeness (SOAP) | >80% | — |
| End-to-End Latency | <30s | — |

Run the evaluation:

```bash
python -m src.eval.medagent-agent-evaluate
python -m src.eval.medagent-trace-analyze
```

## Project Structure

```
medagent/
├── configs/                          # Agent and server configurations
├── src/
│   ├── agents/                       # 4 clinical agents
│   ├── graph/                        # LangGraph state and workflow
│   ├── mcp_servers/                  # 3 MCP tool servers
│   ├── tools/                        # LangChain tool wrappers
│   ├── eval/                         # Evaluation framework
│   └── ui/                           # Gradio demo
├── data/                             # Test cases and reference data
├── test/                             # Unit tests
├── docker/                           # Docker and docker-compose
├── result/                           # Evaluation results and figures
└── scripts/                          # Utility scripts
```

## Tech Stack

- **Agent Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent orchestration
- **Tool Protocol**: [MCP](https://modelcontextprotocol.io/) — Model Context Protocol for tool integration
- **LLM**: OpenAI GPT-4o-mini (orchestration) + MedLlama fine-tuned model (medical RAG)
- **UI**: [Gradio](https://gradio.app/) — interactive demo interface
- **External APIs**: NLM RxNorm (drug interactions)

## Related Projects

- [MedLlama](https://github.com/wwhite777/MedLlama) — LLM fine-tuning (QLoRA + DPO) and RAG pipeline for medical question answering

## License

MIT
