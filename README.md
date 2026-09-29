# 🎙️ Real-Time Audio Intent Classification

Capture live voice input, convert speech to text, and classify the intent of the spoken text using open-source NLP models. The system understands user intent in real time and categorizes requests into predefined categories such as **Shopping, Customer Support, Technical Issues, Billing, Returns & Refunds**, and more.

---

## 📌 Objective

- Capture **real-time voice input** from the microphone
- Convert speech to text using **Whisper**
- Classify the **intent** of the transcribed text using a pre-trained NLP model
- Display the spoken text and detected intent instantly

---

## ⚙️ How It Works

### 1. Capture Voice Input
- Records live audio until the user is **silent for 2 seconds**
- Uses `sounddevice` to capture audio and `NumPy` to process it
- The recorded audio is **plotted** for verification

### 2. Speech-to-Text (STT)
- Uses **Hugging Face Whisper** (`openai/whisper-small`)
- Audio is processed **directly in memory** (no file saving required)
- Returns the transcribed text

### 3. Intent Classification
- The transcribed text is passed to the intent classification model (`Serj/intent-classifier`)
- Uses **Few-Shot Learning** to improve accuracy **without fine-tuning**
- Returns the intent that best matches the user's request

### 4. Display Results
- Prints the **spoken text** and the **detected intent**
- Provides real-time classification of user intent

---

## ✨ Key Features

- 🎤 Real-time voice input with automatic silence detection
- 📝 Speech-to-Text conversion using Hugging Face Whisper
- 🧠 Intent classification with a pre-trained NLP model
- 🎯 Few-Shot Learning for better accuracy
- 🚫 No fine-tuning or additional training required

---

## 💡 Example Workflow

### Scenario 1: Online Shopping Query
| Step | Output |
|------|--------|
| **User Speaks** | "I want to buy a new smartphone. Can you help me choose?" |
| **Transcribed Text** | "I want to buy a new smartphone. Can you help me choose?"
