# app.py (Gradio with Code Q&A and JVM Crash Analyzer Tabs)

import gradio as gr
import os
import shutil
import logging
import re # For parsing hs_err_pid files
from operator import itemgetter

# --- Environment Variable Loading ---
from dotenv import load_dotenv

# --- LangChain & Google Imports ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
import google.generativeai as genai

# --- Configuration ---
DEFAULT_CODE_DIR = "./my-java-project"
VECTORSTORE_DIR = "./java_vectorstore_gemini"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_LLM_MODEL = "gemini-2.5-pro-exp-03-25" # Or "gemini-pro", "gemini-1.5-flash-latest" etc.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Functions (Code Q&A related) ---
# create_or_load_vectorstore, configure_google_api, format_docs,
# create_rag_tool_chain, run_rag_tool_wrapper remain largely the same
# (Ensure error returns in setup_agent_components handle the new return signature)

def create_or_load_vectorstore(
    code_dir: str,
    vectorstore_path: str,
    embedding_model_name: str,
    api_key: str,
    force_reindex: bool = False,
    ui_log_func=logging.info
) -> Chroma | None:
    # ... (Implementation from previous version) ...
    vector_store = None
    embeddings = None

    if not api_key:
        ui_log_func("API Key is missing. Cannot initialize embeddings.", error=True)
        logging.error("API Key is missing in create_or_load_vectorstore.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_key
        )
        logging.info(f"Initialized GoogleGenerativeAIEmbeddings with model: {embedding_model_name}")
    except Exception as e:
        ui_log_func(f"Failed to initialize Google Embeddings. Error: {e}. Make sure API key is valid.", error=True)
        logging.error(f"Failed to initialize Google Embeddings: {e}", exc_info=True)
        return None

    # Load existing vector store
    if os.path.exists(vectorstore_path) and not force_reindex:
        try:
            ui_log_func(f"Loading existing vector store from {vectorstore_path}", info=True)
            vector_store = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            ui_log_func("Vector store loaded.", success=True)
            logging.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            ui_log_func(f"Failed to load existing vector store: {e}. Attempting re-index.", warning=True)
            logging.warning(f"Failed to load existing vector store from {vectorstore_path}: {e}", exc_info=True)
            try:
                shutil.rmtree(vectorstore_path)
                logging.info(f"Removed potentially corrupted directory: {vectorstore_path}")
            except Exception as rm_err:
                ui_log_func(f"Error removing corrupted vector store directory: {rm_err}", error=True)
                logging.error(f"Error removing corrupted directory {vectorstore_path}: {rm_err}", exc_info=True)
                return None

    # Handle force re-index or if loading failed and directory was removed
    if os.path.exists(vectorstore_path) and force_reindex:
         ui_log_func(f"Re-indexing: Removing existing vector store at {vectorstore_path}", info=True)
         logging.info(f"Re-indexing: Removing existing vector store at {vectorstore_path}")
         try:
            shutil.rmtree(vectorstore_path)
         except Exception as e:
            ui_log_func(f"Error removing vector store directory for re-indexing: {e}", error=True)
            logging.error(f"Error removing vector store directory {vectorstore_path} for re-indexing: {e}", exc_info=True)
            return None

    # Create new vector store
    if not os.path.exists(code_dir):
        ui_log_func(f"Error: Code directory '{code_dir}' not found.", error=True)
        logging.error(f"Code directory '{code_dir}' not found.")
        return None

    ui_log_func(f"Creating new vector store from code in {code_dir}", info=True)
    logging.info(f"Creating new vector store from code in {code_dir}")
    try:
        loader = DirectoryLoader(
            code_dir,
            glob="**/*.java",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} raw documents.")

        if not documents:
            ui_log_func("No .java files found in the specified directory. Index will be empty.", warning=True)
            logging.warning(f"No .java files found in {code_dir}.")
            return None

        java_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=1500, chunk_overlap=150
        )
        split_docs = java_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        for doc in split_docs:
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

        ui_log_func(f"Creating embeddings using '{embedding_model_name}' and storing in Chroma (this might take a while)...", info=True)
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=vectorstore_path
        )
        vector_store.persist()
        ui_log_func(f"Vector store created and saved to {vectorstore_path}", success=True)
        logging.info(f"Vector store created and saved to {vectorstore_path}")
        return vector_store

    except Exception as e:
        ui_log_func(f"Error during indexing process: {e}", error=True)
        logging.error(f"Error during indexing process: {e}", exc_info=True)
        return None

def configure_google_api(api_key: str, ui_log_func=logging.info):
    """Configures the Google Generative AI client."""
    # ... (Implementation from previous version) ...
    try:
        if not api_key:
            raise ValueError("API key cannot be empty.")
        genai.configure(api_key=api_key)
        logging.info("Google Generative AI client configured successfully.")
        return True
    except Exception as e:
        ui_log_func(f"Failed to configure Google AI client: {e}. Please check API key.", error=True)
        logging.error(f"Failed to configure Google AI client: {e}", exc_info=True)
        return False

def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into a single string for context."""
    # ... (Implementation from previous version) ...
    return "\n\n".join([f"--- Source: {doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}" for doc in docs])

def create_rag_tool_chain(llm: ChatGoogleGenerativeAI, retriever):
    """Creates the core RAG chain logic for the tool."""
    # ... (Implementation from previous version - prompt asks for inline citations) ...
    tool_prompt_template_str = """You are an expert Java programming assistant. Use the following pieces of context (Java code snippets) ONLY to answer the question accurately and concisely.
If you don't know the answer based *only* on the provided context, state that clearly. Do not make up information.
**IMPORTANT: When using information from the context, you MUST cite the source filename immediately after the information, like this: [Source: MyClass.java].**

Retrieved Context (Code Snippets):
{context}

Question: {question}

Helpful Answer (with inline citations):"""
    RAG_TOOL_PROMPT = PromptTemplate(
        template=tool_prompt_template_str,
        input_variables=["context", "question"]
    )

    rag_tool_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_TOOL_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_tool_chain

def run_rag_tool_wrapper(question: str, rag_chain, retriever) -> str:
    """Runs the RAG chain. Includes logic to retrieve docs separately first."""
    # ... (Implementation from previous version) ...
    try:
        retrieved_docs = retriever.invoke(question)
        logging.info(f"[run_rag_tool_wrapper] Retrieved {len(retrieved_docs)} docs for RAG tool.")
    except Exception as e:
        logging.error(f"[run_rag_tool_wrapper] Error retrieving documents: {e}", exc_info=True)

    logging.info(f"[run_rag_tool_wrapper] Invoking RAG chain for question: '{question[:50]}...'")
    answer = rag_chain.invoke(question)
    logging.info(f"[run_rag_tool_wrapper] RAG chain answer: '{answer[:100]}...'")
    return answer

# --- JVM Crash Analysis Functions ---

def parse_hs_err_log(file_path: str) -> dict | None:
    """Parses essential information from an hs_err_pid log file."""
    parsed_data = {
        "error_summary": None,
        "problematic_frame": None,
        "failing_thread_name": None,
        "failing_thread_stack": [],
        "vm_args": [],
        "timestamp": None,
        "os_info": None,
        "memory_info": None,
        "section_headers": [] # Track seen sections
    }
    current_section = None
    in_thread_stack = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                # Capture Timestamp
                if parsed_data["timestamp"] is None and re.search(r'^\# Time:', line):
                     parsed_data["timestamp"] = line.split(":", 1)[1].strip()

                # Capture Error Summary (often near the top)
                if parsed_data["error_summary"] is None and \
                   (line.startswith("# A fatal error has been detected") or \
                    re.match(r'^# +(EXCEPTION_|SIG|java\.lang\.\w+Error)', line)):
                     parsed_data["error_summary"] = line.lstrip('# ').strip()

                # Capture Problematic Frame
                if line.startswith("Problematic frame:"):
                    parsed_data["problematic_frame"] = line.split(":", 1)[1].strip()

                # Identify Section Headers
                if line.startswith("---------------") and "SECTION" in line:
                    current_section = line
                    parsed_data["section_headers"].append(current_section)
                    in_thread_stack = False # Reset stack capture on new section
                    continue # Move to next line

                # Capture VM Arguments
                if "VM Arguments" in str(current_section):
                     if line.startswith("jvm_args:") or line.startswith("java_command:") : continue # Skip headers
                     if line.startswith("-D") or line.startswith("-X") or line.startswith("="): # Common arg prefixes
                         parsed_data["vm_args"].extend(line.split()) # Simple split, might need refinement

                # Capture Failing Thread Info (usually under THREAD section)
                if "THREAD" in str(current_section):
                    if line.startswith("Current thread"):
                        match = re.search(r'"([^"]+)"', line)
                        if match:
                            parsed_data["failing_thread_name"] = match.group(1)
                        in_thread_stack = True # Start capturing stack from next lines
                    elif in_thread_stack and (line.startswith("Stack:") or line.startswith("\t")) :
                        parsed_data["failing_thread_stack"].append(line)
                    elif in_thread_stack and not line.startswith("\t"): # End of stack?
                        in_thread_stack = False

                # Capture OS Info
                if "Operating System" in str(current_section) and parsed_data["os_info"] is None:
                     parsed_data["os_info"] = line

                # Capture Memory Info
                if "Memory" in str(current_section) and "memory" in line.lower() and parsed_data["memory_info"] is None:
                     parsed_data["memory_info"] = line


        # Basic validation: Check if essential fields were found
        if not parsed_data["error_summary"] or not parsed_data["failing_thread_stack"]:
             logging.warning(f"Parsing hs_err_pid incomplete for {file_path}. Missing essential fields.")
             # Decide if returning partial data is okay or should return None
             # For now, return partial if error summary exists
             if not parsed_data["error_summary"]: return None

        # Join stack lines
        parsed_data["failing_thread_stack"] = "\n".join(parsed_data["failing_thread_stack"])

        logging.info(f"Parsed hs_err_pid file: {file_path}")
        return parsed_data

    except FileNotFoundError:
        logging.error(f"File not found during parsing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error parsing hs_err_pid file {file_path}: {e}", exc_info=True)
        return None


def analyze_crash_data_with_llm(parsed_data: dict, llm: ChatGoogleGenerativeAI) -> str:
    """Generates an analysis of the parsed JVM crash data using an LLM."""

    if not llm:
        return "Error: LLM is not available for analysis."
    if not parsed_data:
        return "Error: No valid parsed crash data provided."

    # Construct the prompt
    prompt_lines = [
        "You are an expert Java Virtual Machine (JVM) performance and crash analysis assistant.",
        "Analyze the following summarized data extracted from a JVM hs_err_pid crash log:",
        "--- Crash Data Summary ---",
        f"Timestamp: {parsed_data.get('timestamp', 'N/A')}",
        f"Error Summary: {parsed_data.get('error_summary', 'N/A')}",
        f"Problematic Frame: {parsed_data.get('problematic_frame', 'N/A')}",
        f"Failing Thread Name: {parsed_data.get('failing_thread_name', 'N/A')}",
        f"Operating System Info: {parsed_data.get('os_info', 'N/A')}",
        f"Memory Info: {parsed_data.get('memory_info', 'N/A')}",
        f"Relevant VM Arguments: {', '.join(parsed_data.get('vm_args', [])) if parsed_data.get('vm_args') else 'N/A'}",
        "Failing Thread Stack Trace:",
        "```",
        f"{parsed_data.get('failing_thread_stack', 'N/A')}",
        "```",
        "--- Analysis Request ---",
        "Based *only* on the provided summary data:",
        "1. Provide a brief summary of the likely crash event.",
        "2. Explain the potential significance of the 'Problematic Frame'. Indicate if it suggests an issue in Native code (C/C++), JVM internal code (libjvm), or Java application code (J).",
        "3. Describe the likely activity of the 'Failing Thread' based on its name and stack trace.",
        "4. List the most plausible potential root causes given the error summary and problematic frame (e.g., native library bug, JVM bug, JNI issue, resource exhaustion like OutOfMemoryError, specific Java code error, hardware issue).",
        "5. Suggest concrete next steps for deeper investigation (e.g., check OS system logs near the timestamp, analyze core dump if generated and available, check related application logs, search for known bugs related to the problematic frame/components, review relevant Java code, consider adjusting specific VM arguments).",
        "6. Comment on the potential relevance of any provided VM Arguments to the crash.",
        "7. Clearly state if the root cause cannot be determined from this summary alone and specify what additional information (e.g., full log, core dump, specific logs) would be most helpful.",
        "Format the output clearly using Markdown."
    ]
    prompt_text = "\n".join(prompt_lines)

    try:
        logging.info("Sending parsed crash data to LLM for analysis.")
        response = llm.invoke(prompt_text)
        analysis = response.content # Adjust if using a different LLM structure
        logging.info("Received analysis from LLM.")
        return analysis
    except Exception as e:
        logging.error(f"Error invoking LLM for crash analysis: {e}", exc_info=True)
        return f"Error during LLM analysis: {e}"


# --- Shared Agent Setup ---
def setup_agent_components(api_key: str, code_dir: str, vectorstore_path: str, embedding_model: str, llm_model: str, ui_log_func=logging.info):
    """Initializes and returns LLM, Agent Executor, Retriever, AND the base RAG chain."""
    llm = None
    vector_store = None
    retriever = None
    rag_chain_for_tool = None
    agent_executor = None

    # 1. Configure API
    if not configure_google_api(api_key, ui_log_func):
        ui_log_func("Google API Key configuration failed.", error=True)
        return None, None, None, None # llm, executor, retriever, rag_chain

    # 3. Initialize LLM (Moved earlier, needed directly now)
    try:
        llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.1,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        logging.info(f"Initialized ChatGoogleGenerativeAI with model: {llm_model}")
    except Exception as e:
        ui_log_func(f"Error initializing Gemini LLM ({llm_model}): {e}", error=True)
        logging.error(f"Error initializing Gemini LLM ({llm_model}): {e}", exc_info=True)
        return None, None, None, None

    # 2. Load/Create Vector Store (Depends on API key being valid)
    vector_store = create_or_load_vectorstore(
        code_dir, vectorstore_path, embedding_model, api_key, force_reindex=False, ui_log_func=ui_log_func
    )
    if not vector_store:
        # Logged by create_or_load_vectorstore
        # ui_log_func("Failed to initialize vector store.", error=True) # Allow proceeding without vector store? Maybe for crash tool only?
        # For now, let's assume vector store is needed for the full app config to succeed.
        ui_log_func("Failed to initialize vector store. Code Q&A tool inactive.", warning=True)
        # Return None for components that depend on it
        retriever = None
        rag_chain_for_tool = None
        agent_executor = None # Agent needs the tool, tool needs the chain/retriever
        # We *can* still return the LLM
        # return llm, None, None, None
    else:
        # 4. Initialize Retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # 5. Create RAG Tool Chain & Tool
        rag_chain_for_tool = create_rag_tool_chain(llm, retriever)
        code_rag_tool = Tool(
            name="JavaCodeQATool",
            func=lambda q: run_rag_tool_wrapper(q, rag_chain_for_tool, retriever),
            description="Use this tool ONLY to answer specific questions about the Java codebase context provided. Input must be the user's full question about the code.",
        )
        tools = [code_rag_tool]

        # 6. Agent Prompt
        try:
            prompt = hub.pull("hwchase17/react-chat")
            logging.info(f"Loaded agent prompt from hub: hwchase17/react-chat")
        except Exception as e:
            ui_log_func(f"Could not load agent prompt: {e}. Using fallback.", warning=True)
            # ... (fallback prompt definition) ...
            prompt = ChatPromptTemplate.from_messages([ # Fallback
                ("system", "You are a helpful assistant... Use JavaCodeQATool for code questions... Cite sources as [Source: filename.java]..."),
                MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")
            ])


        # 7. Create Agent & Executor
        try:
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms to the required ReAct format.",
                max_iterations=5
            )
            logging.info("Agent executor created successfully.")
        except Exception as e:
            ui_log_func(f"Failed to create agent executor: {e}", error=True)
            logging.error(f"Failed to create agent executor: {e}", exc_info=True)
            agent_executor = None # Ensure it's None on failure

    # Return all components (some might be None if vector store failed)
    return llm, agent_executor, retriever, rag_chain_for_tool


# --- Gradio UI ---

# Global variables for Gradio state
gradio_llm = None # <<< Store the LLM instance directly
gradio_agent_executor = None
gradio_retriever = None
gradio_rag_chain_for_tool = None
gradio_api_key_configured = False # Track configuration status

# Wrapper for Gradio logging/notifications
def gr_log(message, info=False, success=False, warning=False, error=False):
    """Logs to console and provides user feedback via Gradio warnings/errors."""
    # ... (Implementation from previous version) ...
    level = "INFO"
    if error: level = "ERROR"
    elif warning: level = "WARNING"
    elif success: level = "SUCCESS"
    prefix = f"[{level}] "
    print(prefix + message) # Log to console
    if error: gr.Error(f"{message}")
    elif warning: gr.Warning(message)


def run_gradio_app():
    global gradio_llm, gradio_agent_executor, gradio_retriever, gradio_rag_chain_for_tool, gradio_api_key_configured

    with gr.Blocks(theme=gr.themes.Soft(), title="CodeSense & Crash Analyzer") as demo:
        gr.Markdown("# 🧠 CodeSense & JVM Crash Analyzer")
        gr.Markdown("Use the tabs below to interact with the Code Q&A agent or analyze JVM crash logs.")

        # --- Tabbed Interface ---
        with gr.Tabs():

            # --- Tab 1: Code Q&A ---
            with gr.TabItem("Code Q&A Agent"):
                gr.Markdown("Ask questions about a Java codebase. Configure API Key and Code Path first.")
                # Configuration Inputs for Code Q&A
                with gr.Row():
                    qna_api_key_input = gr.Textbox(
                        label="🔑 Google AI API Key", type="password",
                        placeholder="Enter key or leave blank if set in .env",
                        value=os.environ.get("GOOGLE_API_KEY", "")
                    )
                    qna_code_dir_input = gr.Textbox(label="📁 Path to Java Codebase", value=DEFAULT_CODE_DIR)
                    qna_config_button = gr.Button("Configure Agent & LLM")

                qna_status_display = gr.Markdown("Status: Waiting for configuration...")

                with gr.Row():
                     qna_force_rag_checkbox = gr.Checkbox(label="Force RAG Tool Usage", info="Check this to force ALL queries through the codebase RAG tool.")

                qna_chatbot = gr.Chatbot(label="Agent Chat", height=450)
                qna_msg_input = gr.Textbox(label="Your Question", placeholder="Type your question and press Enter...")
                qna_clear_button = gr.Button("Clear Chat & Memory")

                qna_agent_memory_state = gr.State(ConversationBufferMemory(
                     memory_key="chat_history", return_messages=True, input_key="input"
                ))

            # --- Tab 2: JVM Crash Analyzer ---
            with gr.TabItem("JVM Crash Analyzer"):
                gr.Markdown("Upload an `hs_err_pid<pid>.log` file to get an LLM-powered analysis.")
                with gr.Row():
                    crash_file_input = gr.File(label="Upload hs_err_pid<pid>.log File", type="filepath") # Get filepath directly
                    crash_analyze_button = gr.Button("Analyze Crash Log")
                crash_status_display = gr.Markdown("Status: Ready to analyze.")
                crash_analysis_output = gr.Markdown(label="Analysis Results")


        # --- Event Handlers ---

        # 1. Configure Agent & LLM (for both tabs)
        def configure_agent_and_llm(api_key_from_input, code_dir):
            """Handles the configuration button click. Configures LLM and Agent components."""
            global gradio_llm, gradio_agent_executor, gradio_retriever, gradio_rag_chain_for_tool, gradio_api_key_configured

            effective_api_key = api_key_from_input if api_key_from_input else os.environ.get("GOOGLE_API_KEY")

            if not effective_api_key:
                 # Provide feedback on the Q&A tab status
                 qna_status_display.value = "<p style='color:red;'>❌ Error: API Key is required.</p>"
                 gradio_api_key_configured = False
                 gr.Warning("API Key is required (either in the input field or in a .env file).")
                 return "<p style='color:red;'>❌ Error: API Key is required.</p>" # Return status for Q&A tab

            # Define logger for this configuration attempt
            def _gr_setup_log(msg, error=False, warning=False, success=False, info=False):
                 prefix = "[INFO]"
                 color = "blue"
                 if error: prefix = "[ERROR]"; color = "red"; gr.Error(msg)
                 elif warning: prefix = "[WARN]"; color = "orange"; gr.Warning(msg)
                 elif success: prefix = "[SUCCESS]"; color = "green"
                 print(f"{prefix} {msg}")
                 # Update Q&A status display during config
                 qna_status_display.value = f"<p style='color:{color};'>{prefix} {msg}</p>"

            _gr_setup_log("Configuring LLM and Agent Components...", info=True)
            # Get all four components
            llm_comp, executor_comp, retriever_comp, rag_chain_comp = setup_agent_components(
                effective_api_key, code_dir, VECTORSTORE_DIR, GEMINI_EMBEDDING_MODEL, GEMINI_LLM_MODEL,
                ui_log_func=_gr_setup_log
            )

            # Store components globally
            gradio_llm = llm_comp # Store the LLM instance
            gradio_agent_executor = executor_comp
            gradio_retriever = retriever_comp
            gradio_rag_chain_for_tool = rag_chain_comp

            # Determine overall configuration success based on LLM availability
            if gradio_llm:
                gradio_api_key_configured = True
                success_msg = "✅ LLM Configured. "
                if gradio_agent_executor:
                     success_msg += "Code Q&A Agent Ready."
                else:
                     success_msg += "Code Q&A Agent Failed (check logs/vector store)."
                     gr.Warning("Code Q&A Agent setup failed, but LLM is ready for Crash Analyzer.")
                _gr_setup_log(success_msg, success=True)
                return f"<p style='color:green;'>{success_msg}</p>"
            else:
                gradio_api_key_configured = False
                # Reset all globals on LLM failure
                gradio_agent_executor = None
                gradio_retriever = None
                gradio_rag_chain_for_tool = None
                fail_msg = "❌ LLM Configuration Failed. Cannot proceed."
                _gr_setup_log(fail_msg, error=True)
                return f"<p style='color:red;'>{fail_msg}</p>"

        # 2. Respond Handler (Code Q&A Tab)
        def respond_qna(message, chat_history, memory_from_state, force_rag):
            """Handles user message submission in the Code Q&A tab."""
            global gradio_agent_executor, gradio_api_key_configured, gradio_rag_chain_for_tool, gradio_retriever

            agent_ready = gradio_agent_executor and gradio_api_key_configured
            forced_rag_ready = gradio_rag_chain_for_tool and gradio_retriever and gradio_api_key_configured

            if not agent_ready and not force_rag:
                gr.Warning("Code Q&A Agent not configured. Please click 'Configure Agent & LLM'.")
                return "", chat_history, memory_from_state
            if not forced_rag_ready and force_rag:
                gr.Warning("RAG components not configured for forced mode. Please click 'Configure Agent & LLM'.")
                return "", chat_history, memory_from_state

            chat_history.append([message, None])
            answer = "Error: Could not process Q&A request."
            updated_memory = memory_from_state

            try:
                if force_rag:
                    logging.info(f"Gradio Q&A FORCING RAG for: '{message[:50]}...'")
                    print("[INFO] Q&A Tab: Executing forced RAG tool path.")
                    gr.Info("Forcing RAG Tool Usage...")

                    answer = run_rag_tool_wrapper(message, gradio_rag_chain_for_tool, gradio_retriever)
                    logging.info(f"Gradio Q&A Forced RAG response: '{answer[:100]}...'")

                    if isinstance(memory_from_state, ConversationBufferMemory):
                        memory_from_state.save_context({"input": message}, {"output": answer})
                        updated_memory = memory_from_state
                        logging.info("Manually updated memory for Q&A forced RAG path.")
                    else:
                        logging.error("Invalid memory object in Q&A state during forced RAG save.")
                        gr.Error("Failed to save interaction to Q&A memory.")

                else:
                    logging.info(f"Gradio Q&A Invoking Agent for: '{message[:50]}...'")
                    print("[INFO] Q&A Tab: Executing standard agent path.")

                    gradio_agent_executor.memory = memory_from_state
                    response = gradio_agent_executor.invoke({"input": message})
                    answer = response.get("output", "Agent did not return a valid answer.")
                    logging.info(f"Gradio Q&A Agent response: '{answer[:100]}...'")
                    updated_memory = memory_from_state

                chat_history[-1][1] = answer

            except Exception as e:
                logging.error(f"Error during Q&A response generation: {e}", exc_info=True)
                error_msg = f"An error occurred: {str(e)}"
                chat_history[-1][1] = f"⚠️ Error: {error_msg}"
                gr.Error(f"Q&A Processing failed: {e}")
                updated_memory = memory_from_state

            return "", chat_history, updated_memory

        # 3. Clear Chat Handler (Code Q&A Tab)
        def clear_qna_chat():
            """Clears the Code Q&A chat history and resets its memory state."""
            new_memory = ConversationBufferMemory(
                 memory_key="chat_history", return_messages=True, input_key="input"
            )
            gr.Info("Code Q&A chat history and memory cleared.")
            return [], new_memory

        # 4. Analyze Crash Handler (JVM Crash Analyzer Tab)
        def handle_crash_analysis(uploaded_file_obj):
            """Handles the crash log analysis button click."""
            global gradio_llm, gradio_api_key_configured

            crash_status_display.value = "Status: Processing..." # Update status

            if not uploaded_file_obj:
                gr.Warning("Please upload an hs_err_pid log file.")
                crash_status_display.value = "Status: Error - No file uploaded."
                return "Error: No file uploaded." # Return error to output

            if not gradio_llm or not gradio_api_key_configured:
                gr.Error("LLM not configured. Please configure the API Key on the 'Code Q&A Agent' tab first.")
                crash_status_display.value = "Status: Error - LLM not configured."
                return "Error: LLM not configured. Use the 'Code Q&A Agent' tab to configure."

            file_path = uploaded_file_obj # Gradio File component with type="filepath" returns path directly
            logging.info(f"Received crash file for analysis: {file_path}")

            # Parse the file
            crash_status_display.value = "Status: Parsing hs_err_pid log..."
            parsed_data = parse_hs_err_log(file_path)

            if not parsed_data:
                gr.Error(f"Failed to parse the uploaded file: {os.path.basename(file_path)}. Is it a valid hs_err_pid log?")
                crash_status_display.value = "Status: Error - Failed to parse file."
                return f"Error: Could not parse the file '{os.path.basename(file_path)}'. Please ensure it is a valid hs_err_pid log."

            # Analyze with LLM
            crash_status_display.value = "Status: Analyzing data with LLM..."
            analysis_result = analyze_crash_data_with_llm(parsed_data, gradio_llm)

            # Display result
            if "Error:" in analysis_result:
                crash_status_display.value = f"Status: Error during analysis."
                gr.Error(f"LLM analysis failed: {analysis_result}")
            else:
                 crash_status_display.value = "Status: Analysis complete."
                 gr.Info("Crash analysis complete.")

            return analysis_result # Return result to the output Markdown


        # --- Wire Components ---

        # Wiring for Code Q&A Tab
        qna_config_button.click(
            configure_agent_and_llm,
            inputs=[qna_api_key_input, qna_code_dir_input],
            outputs=[qna_status_display] # Update status on the Q&A tab
        )
        qna_msg_input.submit(
            respond_qna,
            inputs=[qna_msg_input, qna_chatbot, qna_agent_memory_state, qna_force_rag_checkbox],
            outputs=[qna_msg_input, qna_chatbot, qna_agent_memory_state]
        )
        qna_clear_button.click(
            clear_qna_chat,
            inputs=None,
            outputs=[qna_chatbot, qna_agent_memory_state]
        )

        # Wiring for JVM Crash Analyzer Tab
        crash_analyze_button.click(
            handle_crash_analysis,
            inputs=[crash_file_input],
            outputs=[crash_analysis_output] # Output analysis result
        )


    # --- Launch Gradio App ---
    print("\nLaunching Gradio App with Code Q&A and JVM Crash Analyzer...")
    print("Attempting to create a Share URL...")
    print("NOTE: The public Share URL is temporary and exposes this running app.")
    demo.queue().launch(share=True)


# --- Main Execution Logic ---
if __name__ == "__main__":
    load_dotenv()
    print("Attempted to load environment variables from .env file.")

    # Basic check if default code dir exists
    if not os.path.exists(DEFAULT_CODE_DIR):
        try:
            os.makedirs(DEFAULT_CODE_DIR)
            logging.warning(f"Default code directory '{DEFAULT_CODE_DIR}' created. Add Java files.")
            # ... (placeholder file creation) ...
        except OSError as e:
            logging.error(f"Failed to create default code directory '{DEFAULT_CODE_DIR}': {e}")

    # Check if key is likely available
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not found in env vars or .env file. Provide in UI if needed.")

    # Directly run the Gradio app
    run_gradio_app()