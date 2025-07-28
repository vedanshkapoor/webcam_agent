# Jagga Jasoos: A Context-Driven Multimodal AI Agent

**Jagga Jasoos** is an innovative AI agent that intelligently chooses **when to listen and when to look**, prioritizing **context-driven perception** over constant data collection. Built as a weekend project, this multimodal AI combines **speech recognition**, **reasoning**, optional **vision**, and expressive **text-to-speech** to create a privacy-conscious, autonomous assistant.

---

## 🧠 How It Works

Jagga Jasoos processes user queries through a streamlined pipeline:

### 🎙️ Speech Input
- Captures audio via microphone.
- Uses **Groq Whisper (whisper-large-v3)** for sub-second speech-to-text transcription.

### 🧠 Reasoning
- Transcribed queries are processed by a **LangGraph REACT agent** powered by **Gemini 2.0**.
- The agent decides if the query:
  - **Text-only:** Answers directly.
  - **Vision-required:** Triggers the webcam for image input.

### 👁️ Vision (Optional)
- Captures a single frame using **OpenCV**.
- Encodes the frame as a **Base64 data URI**.
- Sends the query + image to **Groq’s multimodal LLaMA-4 Maverick** for image-based reasoning.

### 🔊 Response
- Generates a response and converts it to expressive speech via **ElevenLabs TTS**.
- Plays the response asynchronously for a seamless experience.

### 🖥️ Interface
- Built with **Gradio** for:
  - Continuous speech listening
  - Real-time webcam display

**Flow:** `Speech → Reasoning → (Optional Vision) → Speech`

---

## 🌟 Why It Matters

Unlike traditional AI systems that are either **"blind"** (text-only) or **"always watching"** (constantly processing visual data), **Jagga Jasoos** adopts a **context-driven approach**:

- **🛡️ Privacy-first:** Activates vision only when needed.
- **♿ Accessibility:** Enables assistive, multimodal reasoning.
- **🚫 Non-invasive:** Ideal for education and smart environments.

> Instead of collecting more data, Jagga Jasoos chooses *less* — a paradigm shift for AI interaction.

---

## 🔮 Implications

This design has potential in:

- Privacy-first AI assistants that respect user boundaries
- Assistive technologies for better accessibility
- Educational and smart environments where **context-aware AI** is key

---

## 🚧 Current Status

This is a **working prototype** with some known issues:

- **Frontend:** Gradio interface has some UI/UX bugs.
- **Audio Recording:** Voice input isn't captured in screen recordings (likely a config issue).

---

## 🛠️ Setup

To run **Jagga Jasoos** locally:

### 1. Clone the Repository

```bash
git clone https://github.com/vedanshkapoor/webcam_agent.git
cd webcam_agent
