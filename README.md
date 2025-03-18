# LM Studio Plugin for Dify

This plugin allows you to connect your LM Studio local models to Dify.

## Setup

1. Install [LM Studio](https://lmstudio.ai/) on your computer
2. Run LM Studio and load a local model
3. Configure LM Studio to serve the model via API (in server mode)
4. Add this plugin to your Dify instance
5. Configure the plugin with your LM Studio server URL (default is http://localhost:1234)

## Features

- Support for both Chat and Completion modes
- Stream responses for better UX
- Support for function/tool calling (if your model supports it)
- Support for vision capabilities (if your model supports it)

## Requirements

- LM Studio running with at least one model loaded
- API server mode enabled in LM Studio

## Usage

After setting up the plugin, you can use any loaded LM Studio model in your Dify applications by selecting it in the model dropdown.

## lmstudio

**Author:** stvlynn
**Version:** 0.0.1
**Type:** model

### Description

Connect your local LM Studio models to Dify platform. This plugin allows you to utilize your local LLM models within Dify applications.



