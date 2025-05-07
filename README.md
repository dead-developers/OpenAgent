# OpenAgent

<p align="center">
  <img src="assets/logo.jpg" width="200"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

## 🚀 Overview

OpenAgent is a sophisticated dual-model AI agent system that leverages multiple LLMs with smart routing for enhanced reasoning, problem-solving, and autonomous task execution. Built as an evolution of the OpenManus framework, OpenAgent brings advanced capabilities including self-learning, conflict reconciliation, and parallel processing to deliver a powerful, adaptive AI agent solution.

The system features a vector knowledge store for persistent memory and contextual retrieval, enabling continuous improvement through reinforcement learning and knowledge retention across sessions.

## ✨ Key Features

- **Dual-Model Architecture**: Combines a primary code-generating model (deepseek-ai/DeepSeek-V3 via Hugging Face) with a reasoning support model (deepseek-ai/DeepSeek-R1 via Hugging Face) to deliver optimal performance across different task types
- **Smart Model Routing**: Intelligently routes requests between models based on performance metrics
