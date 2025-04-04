# Support AISuite - CodeSense & JVM Crash Analyzer + LLM Trace  (v1.0)

This is a Gradio-based application combining two LLM-powered tools:

1.  **Code Q&A Agent:** An agentic RAG system that allows you to ask questions about a Java codebase. It uses ChromaDB for vector storage, Google Gemini models for embeddings and generation, and LangChain for orchestration.
2.  **JVM Crash Analyzer:** A tool to analyze JVM `hs_err_pid<pid>.log` files. It parses key information from the log and uses an LLM to provide insights into potential causes and next steps.

---

## Features

*   **Code Q&A Tab:**
    *   Configure API Key and Java codebase directory.
    *   Conversational interface to ask questions about the code.
    *   Retrieval-Augmented Generation (RAG) using ChromaDB vector store.
    *   Uses LangChain Agents to decide when to use the RAG tool.
    *   Option to "Force RAG Tool Usage" for debugging or specific queries.
    *   Maintains conversation history using LangChain memory.
*   **JVM Crash Analyzer Tab:**
    *   Upload `hs_err_pid<pid>.log` files.
    *   Parses critical sections (Error, Thread, Stack, VM Args, etc.).
    *   Sends structured summary to an LLM (Gemini) for analysis.
    *   Provides formatted analysis results suggesting causes and investigation steps.
*   **Gradio Interface:**
    *   Easy-to-use web UI.
    *   Tabbed layout for different tools.
    *   Generates a temporary shareable URL for demos (`share=True`).
*   **Configuration:**
    *   Supports loading the Google AI API Key via a `.env` file for security.

---

## Prerequisites

*   **Git:** To clone the repository.
*   **Python:** Version 3.10 or higher recommended.
*   **(macOS Optional):** [Homebrew](https://brew.sh/) for easily installing specific Python versions (`brew install python@3.10`).

---

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name` with the actual URL)*

2.  **Create Python Virtual Environment:**
    ```bash
    # Replace 'python3.10' if you installed a different version or use 'python3'/'python'
    python3.10 -m venv venv
    ```

3.  **Activate Virtual Environment:**
    *   **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **Windows (Command Prompt):**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
        *(You might need to adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Configuration

1.  **Google AI API Key:**
    *   You need an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a file named `.env` in the project root directory.
    *   Add your API key to the `.env` file like this:
        ```dotenv
        # .env
        GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   Alternatively, you can paste the key directly into the API Key field in the Gradio UI when configuring the agent, but using `.env` is recommended for security.

2.  **Java Codebase (for Code Q&A):**
    *   Place the Java source code files (`.java`) you want to query inside the `my-java-project` directory (or change the path in the Gradio UI).
    *   The first time you configure the agent, it will index these files into the `./java_vectorstore_gemini` directory (this can take some time).

---

## Running the Application

1.  Make sure your virtual environment is activated.
2.  Run the Gradio app from the project root directory:
    ```bash
    python app.py
    ```
3.  The script will print output to your console, including:
    *   A **Local URL** (usually `http://127.0.0.1:7860` or similar). Open this in your browser.
    *   A **Public URL** (like `https://<random_string>.gradio.live`). This link is temporary (approx. 72 hours) and allows others to access your running application over the internet.
    *   **Warning:** Be mindful of who you share the public URL with, as it exposes your running application.

---

## Usage

1.  **Code Q&A Tab:**
    *   Enter your API Key (if not using `.env`) and the path to your Java code.
    *   Click **"Configure Agent & LLM"**. Wait for the status message to confirm success. Indexing may take time on the first run.
    *   Ask questions about your code in the chat input.
    *   Use the "Force RAG Tool Usage" checkbox if you want every query to retrieve context from the codebase, bypassing the agent's decision.
    *   Use "Clear Chat & Memory" to start a fresh conversation.
2.  **JVM Crash Analyzer Tab:**
    *   **Important:** Ensure the LLM is configured first using the button on the "Code Q&A" tab.
    *   Click the upload area and select your `hs_err_pid<pid>.log` file.
    *   Click **"Analyze Crash Log"**.
    *   Wait for the analysis to appear in the "Analysis Results" section. Check the status messages for progress.

---

## Dependencies

All required Python packages are listed in `requirements.txt`.

---

## `.gitignore`

The `.gitignore` file is configured to exclude:
*   `.env`: Prevents accidentally committing your secret API key.
*   `venv/`: Excludes the Python virtual environment directory.
*   `java_vectorstore_gemini/`: Excludes the generated ChromaDB vector store.
*   Standard Python cache and OS-specific files.

---

*Version: 1.0*
