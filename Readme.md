# PDF Document Processing and Q&A System

A Python tool for converting PDFs to markdown and asking questions about document content using local AI.

## Features

- Convert local and online PDFs to markdown format
- Ask questions about document content using free local AI (Ollama)
- Process both local files and URLs
- No API keys required for basic functionality

## Prerequisites

### System Dependencies
```bash
# Install required system libraries
sudo apt-get install libgl1-mesa-glx libglib2.0-0
# PDF Document Processing and Q&A System

A Python tool for converting PDFs to markdown and asking questions about document content using local AI.

## Quick Setup

```bash
# System dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Python dependencies
pip3 install docling requests

# Free AI setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:3b
