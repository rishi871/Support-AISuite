# app.py (Unified UI, Agent Decides + Tool Force + SDC Log Analysis + Phoenix OTEL - OpenInference Fix V4)

from __future__ import annotations # Must be the very first line

import gradio as gr
import os
import shutil
import logging
import re
import collections # For defaultdict
from operator import itemgetter
from typing import Any, Dict, Optional, Tuple, List, ContextManager # Added ContextManager

# --- Environment Variable Loading ---
from dotenv import load_dotenv

# --- LangChain & Google Imports ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
import google.generativeai as genai

# --- Phoenix OpenTelemetry Integration (NEW APPROACH) ---
phoenix_integration_enabled = False
# tracer_provider = None # Local to __main__ now
try:
    from phoenix.otel import register as register_phoenix_otel
    from openinference.instrumentation.langchain import LangChainInstrumentor
    import opentelemetry.trace
    # Import specific types needed for annotations INSIDE functions if used explicitly
    from opentelemetry.trace import Span, Status, StatusCode, Tracer

    phoenix_integration_enabled = True
    print("INFO: Arize Phoenix, OpenInference, and OpenTelemetry modules loaded successfully.")

except ImportError as e:
    print(f"ERROR: Failed to import required tracing modules: {e}")
    print("WARNING: Ensure 'arize-phoenix', 'openinference-instrumentation-langchain', etc. are installed.")
    opentelemetry = None # type: ignore
    # tracer will be set to None in __main__
    Span = None # type: ignore
    Status = None # type: ignore
    StatusCode = None # type: ignore
    Tracer = None # type: ignore


# Dummy context manager
class _dummy_context(ContextManager):
    def __enter__(self): return self
    def __exit__(self, *args): return False

# --- Configuration ---
DEFAULT_CODE_DIR = "./my-java-project"
VECTORSTORE_DIR = "./java_vectorstore_gemini"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_LLM_MODEL = "gemini-2.5-pro-exp-03-25"

# --- Log Parsing Limits ---
MAX_ERRORS_TO_STORE = 20
MAX_WARNINGS_TO_STORE = 30
MAX_OTHER_EVENTS = 20

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
shared_llm: Optional[ChatGoogleGenerativeAI] = None
shared_vector_store: Optional[Chroma] = None
shared_retriever: Optional[Any] = None
shared_code_rag_chain: Optional[Runnable] = None
shared_agent_executor: Optional[AgentExecutor] = None
is_configured: bool = False

# --- Phoenix/OTEL Globals ---
# REMOVED module-level type annotation for tracer
# Initialize to None, will be assigned in __main__ if setup succeeds
#tracer = None

# --- Core Functions ---

def configure_google_api(api_key: str, ui_log_func=logging.info):
    try:
        genai.configure(api_key=api_key)
        logging.info("Google Generative AI client configured successfully.")
        return True
    except Exception as e:
        ui_log_func(f"Failed to configure Google AI client: {e}. Check API key.", error=True)
        logging.error(f"Failed to configure Google AI client: {e}", exc_info=True)
        return False

def format_docs(docs: list[Document]) -> str:
     return "\n\n".join([f"--- Source: {doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}" for doc in docs])

# create_or_load_vectorstore (Uses manual tracing)
def create_or_load_vectorstore(code_dir, vs_path, embed_model, api_key, force_reindex=False, ui_log_func=logging.info):
    global tracer # Access the global tracer
    vector_store = None
    embeddings = None

    if not api_key:
        ui_log_func("API Key missing for vector store.", error=True)
        return None

    span_context = tracer.start_as_current_span("create_or_load_vectorstore") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    result_vector_store = None # Define outside with

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording() # Cache is_recording check

        if is_rec:
           span.set_attribute("code_dir", code_dir)
           span.set_attribute("vs_path", vs_path)
           span.set_attribute("force_reindex", force_reindex)

        try:
            ui_log_func("Initializing embeddings...", info=True)
            embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)

            # --- Load/Reindex Logic ---
            if os.path.exists(vs_path) and not force_reindex:
                try:
                    ui_log_func(f"Loading existing vector store from {vs_path}", info=True)
                    vector_store = Chroma(persist_directory=vs_path, embedding_function=embeddings)
                    ui_log_func("Vector store loaded.", success=True)
                    if is_rec: span.set_attribute("action", "load_existing")
                    result_vector_store = vector_store # Assign result
                except Exception as e:
                    ui_log_func(f"Failed loading vector store: {e}. Re-indexing.", warning=True)
                    logging.warning(f"Failed loading vector store: {e}", exc_info=True)
                    try: shutil.rmtree(vs_path)
                    except Exception as rm_e: ui_log_func(f"Failed removing corrupted store: {rm_e}", error=True); # Stay inside 'with'
            elif os.path.exists(vs_path) and force_reindex:
                ui_log_func(f"Re-indexing requested: Removing {vs_path}", info=True)
                try: shutil.rmtree(vs_path)
                except Exception as e: ui_log_func(f"Error removing store for re-index: {e}", error=True); # Stay inside 'with'

            # --- Create New Logic ---
            # Proceed only if not loaded/re-indexed successfully above
            if result_vector_store is None:
                if not os.path.exists(code_dir):
                    ui_log_func(f"Code directory not found: {code_dir}", error=True)
                    if is_rec: span.set_status(Status(StatusCode.ERROR, "Code directory not found"))
                elif not os.listdir(code_dir):
                    ui_log_func(f"Code directory is empty: {code_dir}. Skipping indexing.", warning=True)
                    if is_rec: span.set_attribute("action", "skip_empty_dir")
                else:
                    # Create new
                    ui_log_func(f"Creating new vector store from {code_dir}", info=True)
                    if is_rec: span.set_attribute("action", "create_new")
                    loader = DirectoryLoader(code_dir, glob="**/*.java", loader_cls=TextLoader, show_progress=True, use_multithreading=True, loader_kwargs={'autodetect_encoding': True})
                    docs = loader.load()
                    if not docs:
                        ui_log_func(f"No .java files found in {code_dir}.", warning=True)
                        if is_rec: span.set_attribute("num_docs_loaded", 0)
                    else:
                        if is_rec: span.set_attribute("num_docs_loaded", len(docs))
                        splitter = RecursiveCharacterTextSplitter.from_language(Language.JAVA, chunk_size=1500, chunk_overlap=150)
                        split_docs = splitter.split_documents(docs)
                        for doc in split_docs:
                            if 'source' in doc.metadata:
                                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

                        ui_log_func(f"Creating embeddings for {len(split_docs)} documents (takes time)...", info=True)
                        if is_rec: span.set_attribute("num_docs_split", len(split_docs))
                        vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=vs_path)
                        vector_store.persist()
                        ui_log_func("Vector store created successfully.", success=True)
                        if is_rec:
                            span.set_attribute("status", "success")
                            span.set_status(Status(StatusCode.OK))
                        result_vector_store = vector_store # Assign result
        except Exception as e:
            ui_log_func(f"Error during vector store process: {e}", error=True)
            logging.error(f"Error during vector store creation/loading: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error during vector store process: {e}"))
            # Attempt cleanup only if creation failed
            if os.path.exists(vs_path) and 'vector_store' not in locals() and result_vector_store is None:
                try: shutil.rmtree(vs_path)
                except Exception as rm_e: logging.error(f"Failed to clean up partial vector store {vs_path}: {rm_e}")

    return result_vector_store # Return outside the 'with' block

# create_code_rag_chain (No callbacks needed)
def create_code_rag_chain(llm: ChatGoogleGenerativeAI, retriever):
    tool_prompt_template_str = """You are an expert Java programming assistant tasked with answering questions about a specific codebase.
Use the following retrieved code snippets to answer the question.
If the context doesn't contain the answer, state that the information is not available in the provided snippets.
Keep your answers concise and focused on the code. Cite the source file for each piece of information used.

Example Citation Format:
"The `processData` method in `DataProcessor.java` handles the input stream..."
"... which is defined in the `Constants.java` file."

Retrieved Context (Code Snippets):
{context}

Question: {question}

Helpful Answer (with inline citations):"""
    RAG_PROMPT = PromptTemplate(template=tool_prompt_template_str, input_variables=["context", "question"])

    # Define the chain - LangChainInstrumentor will handle tracing this chain execution
    rag_chain_sequence = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm # LLM calls within are instrumented
        | StrOutputParser()
    )

    # No explicit callback configuration needed anymore
    rag_chain = rag_chain_sequence
    logging.info("Created Code RAG Chain (tracing handled by OpenInference Instrumentor).")
    return rag_chain

# run_code_rag_chain_wrapper (Uses manual tracing)
def run_code_rag_chain_wrapper(question: str) -> str:
    """Directly executes the Code RAG chain with tracing."""
    global shared_code_rag_chain, shared_retriever, tracer
    if not shared_code_rag_chain or not shared_retriever:
        return "Error: Code Q&A RAG components are not configured."

    span_context = tracer.start_as_current_span("run_code_rag_chain_wrapper") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    result = "Error: Code Q&A processing failed unexpectedly." # Default error

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording()

        if is_rec:
            span.set_attribute("question_length", len(question))
            span.set_attribute("question_preview", question[:100] + "..." if len(question)>100 else question)

        try:
            logging.info(f"[run_code_rag_chain_wrapper] Invoking for question: '{question[:50]}...'")
            # Chain invocation happens here. LangChainInstrumentor traces the internal steps.
            answer = shared_code_rag_chain.invoke(question)
            logging.info(f"[run_code_rag_chain_wrapper] Answer: '{answer[:100]}...'")
            if is_rec:
                span.set_attribute("answer_length", len(answer))
                span.set_attribute("answer_preview", answer[:100] + "..." if len(answer)>100 else answer)
                span.set_status(Status(StatusCode.OK))
            result = answer # Set success result
        except Exception as e:
            logging.error(f"Error running Code RAG Chain Wrapper: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error in Code RAG Chain: {e}"))
            result = f"Error during Code Q&A processing: {e}" # Set error result

    return result # Return result outside the 'with' block

# --- JVM Crash Log Parsing and Analysis ---

# parse_hs_err_log (Uses manual tracing)
def parse_hs_err_log(file_path: str) -> dict | None:
    global tracer
    span_context = tracer.start_as_current_span("parse_hs_err_log") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    parsed_data_result = None # Default result

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording()

        if is_rec: span.set_attribute("file_path", file_path)

        parsed_data = { "error_summary": None, "problematic_frame": None, "failing_thread_name": None, "failing_thread_stack": [], "vm_args": [], "timestamp": None, "os_info": None, "memory_info": None, "section_headers": [] }
        current_section = None
        in_thread_stack = False
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    # Using regex for more robust matching
                    if parsed_data["timestamp"] is None and re.search(r'^#\s+Time:', line): parsed_data["timestamp"] = line.split(":", 1)[1].strip()
                    if parsed_data["error_summary"] is None and (line.startswith("# A fatal error") or re.match(r'^#\s+(EXCEPTION_|SIG[A-Z]+|java\.lang\.\w+Error)', line)): parsed_data["error_summary"] = line.lstrip('# ').strip()
                    if line.startswith("Problematic frame:"): parsed_data["problematic_frame"] = line.split(":", 1)[1].strip()
                    if line.startswith("---------------") and "SECTION" in line:
                         current_section = line;
                         parsed_data["section_headers"].append(current_section);
                         in_thread_stack = ("THREAD" in current_section) # Reset stack flag based on section
                         continue # Don't process the header line itself

                    # Section-specific parsing
                    if current_section:
                        if "VM Arguments" in current_section:
                             vm_arg_match = re.match(r'^\s*(?:java_command:|jvm_args:)?\s*(-[XD][^\s]+(?:=[^\s]+)?)\s*(.*)', line)
                             if vm_arg_match:
                                 parsed_data["vm_args"].append(vm_arg_match.group(1))
                                 remaining_line = vm_arg_match.group(2).strip()
                                 if remaining_line: parsed_data["vm_args"].extend(remaining_line.split())
                             elif line.startswith('='):
                                 parsed_data["vm_args"].extend(line.lstrip('=').split())
                        elif "THREAD" in current_section:
                            if line.startswith("Current thread"):
                                match = re.search(r'"([^"]+)"', line); parsed_data["failing_thread_name"] = match.group(1) if match else 'Unknown'; in_thread_stack = True
                            elif in_thread_stack and (line.startswith("Stack:") or re.match(r'^\s*(\tat |J |j |V |v ).*', line)): # Include J/j/V/v prefixes
                                 parsed_data["failing_thread_stack"].append(line)
                            elif in_thread_stack and not (line.startswith("Stack:") or re.match(r'^\s*(\tat |J |j |V |v ).*', line)):
                                 in_thread_stack = False
                        elif "Operating System" in current_section and parsed_data["os_info"] is None:
                             if not line.startswith("#") and len(line) > 5: parsed_data["os_info"] = line
                        elif "Memory" in current_section and "memory" in line.lower() and parsed_data["memory_info"] is None:
                             if not line.startswith("#") and len(line) > 5: parsed_data["memory_info"] = line

            if not parsed_data["error_summary"] and not parsed_data["problematic_frame"]:
                 logging.warning(f"Parsing hs_err_pid incomplete or file not recognized as hs_err: {file_path}.")
                 if is_rec: span.set_attribute("parse_status", "incomplete_or_not_hs_err")
            else:
                parsed_data["failing_thread_stack"] = "\n".join(parsed_data["failing_thread_stack"])
                logging.info(f"Parsed hs_err_pid file: {file_path}")
                if is_rec:
                    span.set_attribute("parse_status", "success")
                    span.set_attribute("error_summary_found", bool(parsed_data["error_summary"]))
                    span.set_attribute("problematic_frame_found", bool(parsed_data["problematic_frame"]))
                    span.set_status(Status(StatusCode.OK))
                parsed_data_result = parsed_data # Set success result

        except Exception as e:
            logging.error(f"Error parsing hs_err_pid file {file_path}: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error parsing hs_err_pid: {e}"))

    return parsed_data_result

# analyze_crash_data_with_llm (LLM call automatically instrumented)
def analyze_crash_data_with_llm(parsed_data: dict, llm: ChatGoogleGenerativeAI) -> str:
    if not llm: return "Error: LLM is not available for analysis."
    if not parsed_data: return "Error: No valid parsed crash data provided."
    # Construct prompt
    prompt_lines = [
        "You are an expert Java Virtual Machine (JVM) crash log analyzer.",
        "Analyze the following information extracted from a JVM HotSpot Error Log (hs_err_pid).",
        "--- JVM Crash Data ---"
    ]
    if parsed_data.get("timestamp"): prompt_lines.append(f"Timestamp: {parsed_data['timestamp']}")
    if parsed_data.get("error_summary"): prompt_lines.append(f"Error Summary: {parsed_data['error_summary']}")
    if parsed_data.get("problematic_frame"): prompt_lines.append(f"Problematic Frame: {parsed_data['problematic_frame']}")
    if parsed_data.get("failing_thread_name"): prompt_lines.append(f"Failing Thread: {parsed_data['failing_thread_name']}")
    if parsed_data.get("os_info"): prompt_lines.append(f"OS Info: {parsed_data['os_info']}")
    if parsed_data.get("memory_info"): prompt_lines.append(f"Memory Info: {parsed_data['memory_info']}")
    if parsed_data.get("vm_args"): prompt_lines.append(f"VM Arguments: {' '.join(parsed_data['vm_args'])}")

    if parsed_data.get("failing_thread_stack"):
        prompt_lines.append("\n--- Failing Thread Stack Trace ---")
        stack_trace = parsed_data['failing_thread_stack']
        max_stack_len = 2000
        prompt_lines.append(f"```\n{(stack_trace[:max_stack_len] + '...') if len(stack_trace) > max_stack_len else stack_trace}\n```")
    else:
        prompt_lines.append("\n--- Failing Thread Stack Trace ---")
        prompt_lines.append("Stack trace not found or parsing failed.")

    prompt_lines.extend([
        "\n--- Analysis Request ---",
        "Provide a concise analysis including:",
        "1.  **Root Cause Hypothesis:** ...",
        "2.  **Key Evidence:** ...",
        "3.  **Impact of VM Arguments:** ...",
        "4.  **Problematic Code Area:** ...",
        "5.  **Troubleshooting Steps:** ...",
        "6.  **Information Gaps:** ...",
        "Format the output clearly using Markdown."
    ])
    prompt_text = "\n".join(prompt_lines)

    try:
        logging.info("Sending parsed JVM crash data to LLM for analysis.")
        # LangChainInstrumentor handles tracing this invoke call
        response = llm.invoke(prompt_text)
        analysis = response.content if hasattr(response, 'content') else str(response)
        logging.info("Received JVM crash analysis from LLM.")
        return analysis
    except Exception as e:
        logging.error(f"Error invoking LLM for crash analysis: {e}", exc_info=True)
        # LangChainInstrumentor should record this exception in the LLM span
        return f"Error during LLM analysis: {e}"

# run_jvm_crash_analysis (Uses manual tracing for wrapper)
def run_jvm_crash_analysis(file_path: str) -> str:
    """Parses and analyzes a JVM crash log file using the LLM with tracing."""
    global shared_llm, tracer
    if not shared_llm: return "Error: LLM is not configured..."
    if not file_path or not os.path.exists(file_path): return f"Error: Invalid file path..."

    span_context = tracer.start_as_current_span("run_jvm_crash_analysis") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    final_analysis_result = f"Error: JVM crash analysis failed unexpectedly." # Default error

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording()

        if is_rec: span.set_attribute("file_path", file_path)

        try:
            logging.info(f"Starting forced JVM crash analysis for: {file_path}")
            parsed_data = parse_hs_err_log(file_path) # Traced internally
            if not parsed_data:
                result = f"Error: Failed to parse the crash log file: {os.path.basename(file_path)}."
                if is_rec: span.set_status(Status(StatusCode.ERROR, "Parsing failed"))
                final_analysis_result = result
            else:
                analysis = analyze_crash_data_with_llm(parsed_data, shared_llm) # LLM call traced internally
                if is_rec:
                     span.set_attribute("analysis_length", len(analysis))
                     span.set_status(Status(StatusCode.OK))
                final_analysis_result = analysis
        except Exception as e:
            logging.error(f"Error during JVM crash analysis execution: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error in JVM analysis execution: {e}"))
            final_analysis_result = f"An error occurred during JVM crash analysis: {e}"

    return final_analysis_result


# --- SDC Log Parsing and Analysis ---

# Regexes needed for parse_sdc_log
STACK_TRACE_REGEX = re.compile(r"^\s*(\tat |Caused by: |Suppressed: |\.{3}\s+\d+\s+more).*", re.IGNORECASE)
LOG_LINE_REGEX = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+" # 1: Timestamp
    r"\[user:(.*?)\]\s+"                            # 2: User
    r"\[pipeline:(.*?)\]\s+"                       # 3: Pipeline ID/Name
    r"\[runner:(.*?)\]\s+"                         # 4: Runner ID
    r"\[thread:(.*?)\]\s+"                         # 5: Thread Name
    r"(?:\[stage:(.*?)\]\s+)?"                      # 6: Stage Name (Optional)
    r"(INFO|WARN|ERROR|DEBUG|TRACE)\s+"            # 7: Log Level
    r"([^\s]+)\s+-\s+"                             # 8: Class Name
    r"(.*)"                                        # 9: Message
)

def _is_stack_trace_line(line: str) -> bool:
    """Checks if a line looks like a Java stack trace element."""
    return bool(STACK_TRACE_REGEX.match(line))

# parse_sdc_log (Uses manual tracing)
def parse_sdc_log(file_path: str) -> Optional[Dict]:
    """ Parses SDC log with tracing """
    global tracer
    span_context = tracer.start_as_current_span("parse_sdc_log") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    parsed_data_result = None # Default result

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording()

        if is_rec: span.set_attribute("file_path", file_path)

        parsed_data = {
            "metadata": {"versions": [], "build_info": [], "java_version": None, "sdc_id": None, "directories": {}, "jvm_args": None,},
            "errors": [], "warnings": [], "pipeline_activity": collections.defaultdict(list),
            "stage_activity": collections.defaultdict(lambda: collections.defaultdict(list)),
            "performance_issues": [],
            "thread_summary": collections.defaultdict(lambda: {"count": 0, "errors": 0, "warnings": 0}),
            "other_notable_events": [], "log_parsing_errors": 0
        }
        active_entry_ref: Optional[Tuple[List[str], int]] = None
        line_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    line_count = line_num + 1
                    line = line.strip()
                    if not line: continue

                    match = LOG_LINE_REGEX.match(line)
                    if match:
                        active_entry_ref = None # New distinct event
                        timestamp, user, pipeline_id, runner_id, thread, stage, level, class_name, message = match.groups()
                        pipeline_id = pipeline_id if (pipeline_id and pipeline_id != '*') else 'GLOBAL'
                        log_entry_line = f"[{timestamp}] [{level}] [{thread}] {class_name} - {message}"

                        # --- Metadata Extraction ---
                        if class_name == "Main":
                            if "Version:" in message and len(parsed_data["metadata"]["versions"]) < 3: parsed_data["metadata"]["versions"].append(message.strip())
                            elif "Built by" in message: parsed_data["metadata"]["build_info"].append(message.strip())
                            elif "Java version" in message: parsed_data["metadata"]["java_version"] = message.split(":", 1)[1].strip()
                            elif "SDC ID" in message: parsed_data["metadata"]["sdc_id"] = message.split(":", 1)[1].strip()
                            elif "Runtime dir" in message: parsed_data["metadata"]["directories"]["runtime"] = message.split(":", 1)[1].strip().split('(')[0].strip()
                            elif "Config dir" in message: parsed_data["metadata"]["directories"]["config"] = message.split(":", 1)[1].strip().split('(')[0].strip()
                            elif "Data dir" in message: parsed_data["metadata"]["directories"]["data"] = message.split(":", 1)[1].strip().split('(')[0].strip()
                            elif "Log dir" in message: parsed_data["metadata"]["directories"]["log"] = message.split(":", 1)[1].strip().split('(')[0].strip()
                            elif "Non-SDC JVM Args:" in message: parsed_data["metadata"]["jvm_args"] = message.split(":", 1)[1].strip()

                        # --- Thread Summary ---
                        if thread:
                             parsed_data["thread_summary"][thread]["count"] += 1
                             if level == "ERROR": parsed_data["thread_summary"][thread]["errors"] += 1
                             elif level == "WARN": parsed_data["thread_summary"][thread]["warnings"] += 1

                        # --- Store Significant Entries ---
                        target_list, limit = (None, 0)
                        if level == "ERROR" and len(parsed_data["errors"]) < MAX_ERRORS_TO_STORE:
                            target_list, limit = parsed_data["errors"], MAX_ERRORS_TO_STORE
                            if "OutOfMemoryError" in message or "Java heap space" in message:
                                if len(parsed_data["performance_issues"]) < limit: parsed_data["performance_issues"].append(f"Potential OOM: {log_entry_line}")
                        elif level == "WARN" and len(parsed_data["warnings"]) < MAX_WARNINGS_TO_STORE:
                            target_list, limit = parsed_data["warnings"], MAX_WARNINGS_TO_STORE
                            if "TimeoutException" in message or "timed out" in message.lower():
                                 if len(parsed_data["performance_issues"]) < limit: parsed_data["performance_issues"].append(f"Potential Timeout: {log_entry_line}")

                        if target_list is not None:
                            target_list.append(log_entry_line)
                            active_entry_ref = (target_list, len(target_list) - 1)

                        # --- Pipeline/Stage Activity ---
                        if pipeline_id != 'GLOBAL':
                            if stage:
                                if "Could not obtain a driver" in message or "Failed to get driver instance" in message:
                                     if len(parsed_data["stage_activity"][pipeline_id][stage]) < 5:
                                          parsed_data["stage_activity"][pipeline_id][stage].append(f"[{timestamp}] [{level}] {message[:200]}...")
                            else:
                                if any(phrase in message for phrase in ["Starting pipeline", "Stopping pipeline", "Pipeline is in terminal state", "Stopped due to validation error"]):
                                     if len(parsed_data["other_notable_events"]) < MAX_OTHER_EVENTS:
                                         parsed_data["pipeline_activity"][pipeline_id].append(log_entry_line)
                                         parsed_data["other_notable_events"].append(log_entry_line)

                    elif active_entry_ref and _is_stack_trace_line(line):
                        try:
                            entry_list, entry_index = active_entry_ref
                            if entry_index < len(entry_list):
                                 entry_list[entry_index] += f"\n{line}"
                            else: active_entry_ref = None; parsed_data["log_parsing_errors"] += 1
                        except Exception: active_entry_ref = None; parsed_data["log_parsing_errors"] += 1
                    else:
                        active_entry_ref = None
                        if not _is_stack_trace_line(line): parsed_data["log_parsing_errors"] += 1

            # --- Final Cleanup ---
            parsed_data["metadata"]["versions"] = list(set(parsed_data["metadata"]["versions"]))
            parsed_data["pipeline_activity"] = {k: v for k, v in parsed_data["pipeline_activity"].items() if v}
            parsed_data["stage_activity"] = {
                pid: {stage: msgs for stage, msgs in stages.items() if msgs}
                for pid, stages in parsed_data["stage_activity"].items() if any(stages.values())
            }
            parsed_data["thread_summary"] = dict(sorted(parsed_data["thread_summary"].items(), key=lambda item: item[1]['count'], reverse=True)[:10])

            logging.info(f"Parsed SDC log file: {file_path}. Found {len(parsed_data['errors'])} errors, {len(parsed_data['warnings'])} warnings. Parsing errors: {parsed_data['log_parsing_errors']}")
            if is_rec:
                span.set_attribute("parse_status", "success")
                span.set_attribute("line_count", line_count)
                span.set_attribute("errors_found", len(parsed_data['errors']))
                span.set_attribute("warnings_found", len(parsed_data['warnings']))
                span.set_attribute("parsing_errors", parsed_data['log_parsing_errors'])
                span.set_status(Status(StatusCode.OK))
            parsed_data_result = parsed_data

        except FileNotFoundError:
             logging.error(f"SDC log file not found: {file_path}")
             if is_rec: span.set_status(Status(StatusCode.ERROR, "File not found"))
        except Exception as e:
            logging.error(f"Error parsing SDC log file {file_path}: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error parsing SDC log: {e}"))

    return parsed_data_result

# analyze_sdc_log_with_llm (LLM call automatically instrumented)
def analyze_sdc_log_with_llm(parsed_data: dict, llm: ChatGoogleGenerativeAI) -> str:
    if not llm: return "Error: LLM is not available for SDC log analysis."
    if not parsed_data: return "Error: No valid parsed SDC log data provided."
    # --- Prepare concise summary for LLM ---
    summary_lines = [
        "You are an expert StreamSets Data Collector (SDC) administrator and troubleshooter.",
        "Analyze the following summarized information extracted from an SDC log file (`sdc.log`). Focus only on the provided information.",
        "\n--- SDC Metadata Summary ---"
    ]
    meta = parsed_data.get('metadata', {})
    if meta.get('versions'): summary_lines.append(f"SDC Version(s) Detected: {', '.join(meta['versions'])}")
    if meta.get('java_version'): summary_lines.append(f"Java Version: {meta['java_version']}")
    if meta.get('sdc_id'): summary_lines.append(f"SDC ID: {meta['sdc_id']}")
    if meta.get('jvm_args'):
        xmx = re.search(r'-Xmx(\S+)', meta['jvm_args'])
        xms = re.search(r'-Xms(\S+)', meta['jvm_args'])
        heap_dump = re.search(r'-XX:HeapDumpPath=(\S+)', meta['jvm_args'])
        summary_lines.append(f"JVM Max Heap (-Xmx): {xmx.group(1) if xmx else 'Not Specified/Found'}")
        summary_lines.append(f"JVM Initial Heap (-Xms): {xms.group(1) if xms else 'Not Specified/Found'}")
        summary_lines.append(f"Heap Dump on OOM: {'Yes' if heap_dump else 'No/Not Found'}")
    else: summary_lines.append("JVM Arguments: Not Found/Parsed")

    summary_lines.append(f"\n--- Key Errors Summary (Top {MAX_ERRORS_TO_STORE} shown) ---")
    if parsed_data.get('errors'):
        for i, error in enumerate(parsed_data['errors']):
            summary_lines.append(f"{i+1}. {error[:600]}{'...' if len(error) > 600 else ''}")
    else: summary_lines.append("No significant errors captured in summary.")

    summary_lines.append(f"\n--- Key Warnings Summary (Top {MAX_WARNINGS_TO_STORE} shown) ---")
    if parsed_data.get('warnings'):
        for i, warning in enumerate(parsed_data['warnings']):
            summary_lines.append(f"{i+1}. {warning[:600]}{'...' if len(warning) > 600 else ''}")
    else: summary_lines.append("No significant warnings captured in summary.")

    summary_lines.append(f"\n--- Pipeline Activity Highlights (Top {MAX_OTHER_EVENTS} events) ---")
    if parsed_data.get('other_notable_events'):
         for i, event in enumerate(parsed_data['other_notable_events']):
             match = re.search(r'\[pipeline:(.*?)\] .* - (.*)', event)
             if match: summary_lines.append(f"- Pipeline '{match.group(1).strip()}': {match.group(2).strip()}")
             else: summary_lines.append(f"- {event}")
    else: summary_lines.append("No major pipeline start/stop/failure events captured.")

    summary_lines.append("\n--- Potential Performance Issues Detected ---")
    if parsed_data.get('performance_issues'):
        for i, issue in enumerate(parsed_data['performance_issues']): summary_lines.append(f"{i+1}. {issue}")
    else: summary_lines.append("No specific OOM or Timeout messages captured.")

    if parsed_data.get('log_parsing_errors', 0) > 0:
         summary_lines.append(f"\nNote: Encountered {parsed_data['log_parsing_errors']} log lines that could not be fully parsed.")

    # --- Analysis Request ---
    summary_lines.extend([
        "\n--- Analysis Request ---",
        "Based *only* on the summarized log data provided above:",
        "1.  **Overall Status:** ...", "2.  **Critical Errors:** ...", "3.  **Significant Warnings:** ...",
        "4.  **Pipeline Behavior:** ...", "5.  **Performance:** ...", "6.  **Actionable Recommendations:** ...",
        "7.  **Information Gaps:** ...", "Format the output clearly using Markdown..."
    ])
    prompt_text = "\n".join(summary_lines)
    MAX_PROMPT_CHARS = 25000

    try:
        logging.info("Sending parsed SDC log data to LLM for analysis.")
        if len(prompt_text) > MAX_PROMPT_CHARS:
            logging.warning(f"SDC analysis prompt length exceeds limit. Truncating.")
            prompt_text = prompt_text[:MAX_PROMPT_CHARS] + "\n... [PROMPT TRUNCATED]"

        # LangChainInstrumentor handles tracing this invoke call
        response = llm.invoke(prompt_text)
        analysis = response.content if hasattr(response, 'content') else str(response)
        logging.info("Received SDC log analysis from LLM.")
        return analysis
    except Exception as e:
        logging.error(f"Error invoking LLM for SDC log analysis: {e}", exc_info=True)
        return f"Error during LLM analysis for SDC log: {str(e)}"

# run_sdc_log_analysis (Uses manual tracing for wrapper)
def run_sdc_log_analysis(file_path: str) -> str:
    """ Parses and analyzes an SDC log file using the LLM with tracing. """
    global shared_llm, tracer
    if not shared_llm: return "Error: LLM is not configured..."
    if not file_path or not os.path.exists(file_path): return f"Error: Invalid file path..."

    span_context = tracer.start_as_current_span("run_sdc_log_analysis") if tracer else None
    active_context = span_context if span_context else _dummy_context()
    final_analysis_result = f"Error: SDC log analysis failed unexpectedly." # Default error

    with active_context:
        span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
        is_rec = span and span.is_recording()

        if is_rec: span.set_attribute("file_path", file_path)

        try:
            logging.info(f"Starting SDC log analysis for: {file_path}")
            parsed_data = parse_sdc_log(file_path) # Traced internally
            if not parsed_data:
                result = f"Error: Failed to parse the SDC log file: {os.path.basename(file_path)}."
                if is_rec: span.set_status(Status(StatusCode.ERROR, "Parsing failed"))
                final_analysis_result = result
            else:
                analysis = analyze_sdc_log_with_llm(parsed_data, shared_llm) # LLM call traced internally
                if is_rec:
                     span.set_attribute("analysis_length", len(analysis))
                     span.set_status(Status(StatusCode.OK))
                final_analysis_result = analysis
        except Exception as e:
            logging.error(f"Error during SDC log analysis execution: {e}", exc_info=True)
            if is_rec:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Error in SDC analysis execution: {e}"))
            final_analysis_result = f"An error occurred during SDC log analysis: {e}"

    return final_analysis_result

# --- Tool Definitions ---
# Ensure the funcs used here are the traced versions defined above
code_qna_tool = Tool(
    name="JavaCodeQATool",
    func=run_code_rag_chain_wrapper, # Uses traced version
    description="Use this tool ONLY to answer specific questions about the Java codebase context provided. Input must be the user's full question about the code. Do NOT use for log file analysis.",
)
jvm_crash_analyzer_tool = Tool(
    name="JVMCrashLogAnalyzer",
    func=run_jvm_crash_analysis, # Uses traced version
    description="Use this tool ONLY when the user uploads a JVM crash log file (typically named hs_err_pid*.log) AND asks for an analysis of the JVM *crash*. This analyzes fatal JVM errors, signals, and native stack traces. The input MUST be the full path to the uploaded hs_err_pid file provided in the user message.",
)
sdc_log_analyzer_tool = Tool(
    name="SDCLogAnalyzer",
    func=run_sdc_log_analysis, # Uses traced version
    description="Use this tool ONLY when the user uploads a standard StreamSets Data Collector log file (e.g., sdc.log, collector.log) AND asks for an analysis of SDC *runtime* behavior, application errors, warnings, pipeline issues, or configuration problems found *within that application log*. Do NOT use this for hs_err_pid files or general code questions. The input MUST be the full path to the uploaded sdc.log file provided in the user message.",
)


# --- Gradio UI and Logic ---

def gr_log(message, info=False, success=False, warning=False, error=False):
    level = "INFO"
    if error: level = "ERROR"; gr.Error(f"{message}")
    elif warning: level = "WARNING"; gr.Warning(message)
    elif success: level = "SUCCESS"; gr.Info(f"{message}")
    elif info: level = "INFO"; gr.Info(message)
    else: gr.Info(message)
    prefix = f"[{level}] "
    print(prefix + message)


def run_gradio_app():
    global shared_llm, shared_vector_store, shared_retriever, shared_code_rag_chain, shared_agent_executor, is_configured
    # No need to declare global tracer here, it's accessed directly in handlers

    # --- UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Support Suite") as demo:
        gr.Markdown("# 🚀 AI Support Suite")
        gr.Markdown("Unified interface for Code Q&A, JVM Crash Analysis, and SDC Log Analysis.")

        with gr.Accordion("Configuration", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="🔑 Google AI API Key", type="password",
                    placeholder="Enter key or leave blank if set in .env",
                    value=os.environ.get("GOOGLE_API_KEY", "")
                )
                code_dir_input = gr.Textbox(label="📁 Path to Java Codebase (for Code Q&A)", value=DEFAULT_CODE_DIR)
            config_button = gr.Button("Configure LLM & Tools")
            config_status_display = gr.Markdown("Status: Waiting for configuration...")

        gr.Markdown("---")

        chatbot = gr.Chatbot(label="Conversation", height=550, show_copy_button=True)
        chat_memory_state = gr.State(ConversationBufferMemory(
             memory_key="chat_history", return_messages=True, input_key="input"
        ))

        with gr.Row():
            with gr.Column(scale=10):
                text_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask about code, or describe the log file you are uploading (hs_err_pid or sdc.log)...",
                    lines=3
                )
                file_input = gr.File(label="Upload Log File (Optional - hs_err_pid*.log or sdc.log)", type="filepath")
            with gr.Column(scale=3):
                tool_choice = gr.Dropdown(
                    label="Execution Mode",
                    choices=["Agent Decides", "Force Code Q&A", "Force JVM Crash Analysis", "Force SDC Log Analysis"],
                    value="Agent Decides"
                )
                submit_button = gr.Button("➡️ Send", variant="primary")
                clear_button = gr.Button("🗑️ Clear Chat")

        # --- Event Handlers ---

        def handle_configuration(api_key, code_dir):
            """Configures LLM, vector store, tools, agent executor."""
            global shared_llm, shared_vector_store, shared_retriever, shared_code_rag_chain, shared_agent_executor, is_configured

            status_updates = []
            def _log(msg, error=False, warning=False, success=False, info=False):
                 status_updates.append(f"{'[Error] ' if error else '[Warn] ' if warning else '[Ok] ' if success else '[Info] '}{msg}")
                 gr_log(msg, error=error, warning=warning, success=success, info=info)

            _log("Starting configuration...", info=True)
            effective_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not effective_api_key:
                 _log("Google AI API Key is required.", error=True); is_configured = False; shared_llm=None; return "<br>".join(status_updates)

            # 1. Configure API & LLM (No callbacks passed)
            llm_ok = configure_google_api(effective_api_key, _log)
            if not llm_ok: is_configured = False; shared_llm = None; return "<br>".join(status_updates)
            try:
                shared_llm = ChatGoogleGenerativeAI(
                    model=GEMINI_LLM_MODEL,
                    google_api_key=effective_api_key,
                    temperature=0.1,
                    convert_system_message_to_human=True,
                    )
                _log(f"LLM ({GEMINI_LLM_MODEL}) initialized (tracing via OpenInference).", success=True)
            except Exception as e:
                 _log(f"LLM initialization failed: {e}", error=True); logging.error("LLM Init Error", exc_info=True); is_configured = False; shared_llm = None; return "<br>".join(status_updates)

            # 2. Configure Vector Store & RAG
            _log("Attempting to configure Code Q&A tool...", info=True)
            shared_vector_store = create_or_load_vectorstore(code_dir, VECTORSTORE_DIR, GEMINI_EMBEDDING_MODEL, effective_api_key, ui_log_func=_log)
            code_tool_available = False
            if shared_vector_store:
                try:
                    shared_retriever = shared_vector_store.as_retriever(search_kwargs={"k": 4})
                    shared_code_rag_chain = create_code_rag_chain(shared_llm, shared_retriever)
                    _log("Code Q&A components ready.", success=True)
                    code_tool_available = True
                except Exception as e:
                     _log(f"Failed RAG setup: {e}", warning=True); logging.warning("RAG Setup Error", exc_info=True); shared_retriever = None; shared_code_rag_chain = None
            else:
                _log("Code Q&A vector store failed.", warning=True); shared_retriever = None; shared_code_rag_chain = None

            # 3. Define available tools
            tools_list = [jvm_crash_analyzer_tool, sdc_log_analyzer_tool]
            if code_tool_available: tools_list.insert(0, code_qna_tool)

            # 4. Configure Agent Executor
            _log("Configuring Agent Executor...", info=True)
            if not shared_llm:
                 _log("LLM not configured, cannot create Agent Executor.", error=True); shared_agent_executor = None
            else:
                try:
                    try: agent_prompt = hub.pull("hwchase17/react-chat")
                    except Exception as e_hub:
                         _log(f"Hub prompt pull failed: {e_hub}. Using fallback.", warning=True)
                         agent_prompt = ChatPromptTemplate.from_messages([...]) # Define fallback prompt here
                    agent = create_react_agent(shared_llm, tools_list, agent_prompt)
                    shared_agent_executor = AgentExecutor(
                        agent=agent,
                        tools=tools_list,
                        verbose=True,
                        handle_parsing_errors="Check your output and make sure it conforms to the Action/Action Input format.",
                        max_iterations=5,
                    )
                    _log(f"Agent Executor configured (tracing via OpenInference).", success=True)
                except Exception as e:
                     _log(f"Agent Executor config failed: {e}", error=True); logging.error("Agent Config Error", exc_info=True); shared_agent_executor = None

            # Final Status
            is_configured = True
            status_message = "Configuration complete."
            if not shared_agent_executor: status_message += " Agent Executor failed."
            if not code_tool_available: status_message += " Code Q&A disabled."
            _log(status_message, info=True)
            return "<br>".join(status_updates)


        # Use the generator version for submit/click handlers
        def handle_chat_submission_generator(text_msg: str, file_obj: Optional[Any], history: list, memory: ConversationBufferMemory, tool_choice: str):
             global is_configured, shared_agent_executor, shared_llm, shared_code_rag_chain, tracer # Reference global tracer

             # Prepare inputs
             file_path = file_obj
             filename = os.path.basename(file_path) if file_path else None
             display_input = text_msg + (f" (File: {filename})" if filename else "")
             augmented_input = text_msg + (f"\n\n[File Attached: {filename} Path: {file_path}]" if file_path else "")

             # --- Yield User Message ---
             history.append([display_input, None])
             yield history, memory, None, "" # Yield 1: Show user msg, clear inputs

             # --- Start Processing Span ---
             span_context = tracer.start_as_current_span("handle_chat_submission_processing") if tracer else None
             active_context = span_context if span_context else _dummy_context()
             final_answer = "Error: Could not process request." # Default

             with active_context:
                 span: Optional[Span] = opentelemetry.trace.get_current_span() if opentelemetry else None
                 is_rec = span and span.is_recording()

                 if is_rec:
                     span.set_attribute("tool_choice", tool_choice)
                     span.set_attribute("message_length", len(text_msg))
                     span.set_attribute("has_file_upload", file_obj is not None)
                     if filename: span.set_attribute("filename", filename)
                     span.set_attribute("message_preview", text_msg[:100] + ("..." if len(text_msg)>100 else ""))

                 # Config check
                 if not is_configured or not shared_llm:
                     final_answer = "Error: System not configured. Please Configure first."
                     gr.Warning(final_answer.split(': ')[1])
                     if is_rec: span.set_status(Status(StatusCode.ERROR, "System not configured"))
                 else:
                     # Main Processing Block
                     try:
                         status_code = StatusCode.UNSET # Default status code for this block

                         if tool_choice == "Force Code Q&A":
                              if not shared_code_rag_chain: final_answer = "Error: Code Q&A Tool not configured."; gr.Error(final_answer.split(': ')[1]); status_code = StatusCode.FAILED_PRECONDITION
                              else: final_answer = run_code_rag_chain_wrapper(text_msg); status_code = StatusCode.OK if "Error:" not in final_answer else StatusCode.ERROR
                         elif tool_choice == "Force JVM Crash Analysis":
                              if not file_path: final_answer = "Error: Please upload hs_err_pid file."; gr.Error(final_answer.split(': ')[1]); status_code = StatusCode.FAILED_PRECONDITION
                              else: final_answer = run_jvm_crash_analysis(file_path); status_code = StatusCode.OK if "Error:" not in final_answer else StatusCode.ERROR
                         elif tool_choice == "Force SDC Log Analysis":
                              if not file_path: final_answer = "Error: Please upload sdc.log file."; gr.Error(final_answer.split(': ')[1]); status_code = StatusCode.FAILED_PRECONDITION
                              else: final_answer = run_sdc_log_analysis(file_path); status_code = StatusCode.OK if "Error:" not in final_answer else StatusCode.ERROR
                         elif tool_choice == "Agent Decides":
                              if not shared_agent_executor: final_answer = "Error: Agent Executor not configured."; gr.Error(final_answer.split(': ')[1]); status_code = StatusCode.FAILED_PRECONDITION
                              else:
                                  gr.Info("Agent processing...")
                                  agent_input = {"input": augmented_input, "chat_history": memory.chat_memory.messages}
                                  # Agent execution is auto-instrumented
                                  response = shared_agent_executor.invoke(agent_input)
                                  final_answer = response.get("output", "Agent did not provide output.")
                                  memory.save_context({"input": display_input}, {"output": final_answer})
                                  status_code = StatusCode.OK # Assume agent handles internal errors for trace status
                         else:
                              final_answer = "Error: Invalid tool choice selected."; status_code = StatusCode.INVALID_ARGUMENT

                         # Set span status only if it's still UNSET
                         if is_rec and span.status.status_code == StatusCode.UNSET:
                              if status_code != StatusCode.UNSET: # Check if we determined a status
                                   span.set_status(Status(status_code))
                              else: # If logic didn't set a status, assume OK if no exception
                                   span.set_status(Status(StatusCode.OK))

                     except Exception as e:
                         logging.error(f"Error during chat submission processing: {e}", exc_info=True)
                         final_answer = f"An error occurred: {e}"
                         gr.Error(f"Processing Error: {e}") # Show error notification
                         if is_rec:
                             span.record_exception(e)
                             span.set_status(Status(StatusCode.ERROR, f"Error in submission handler: {e}"))
                         try: memory.save_context({"input": display_input}, {"output": f"Error: {str(e)}"})
                         except Exception as mem_e: logging.error(f"Failed to save error context to memory: {mem_e}")

             # --- End Span ---

             # --- Update history & Final Yield ---
             history[-1][1] = final_answer # Update the None placeholder
             if is_rec: span.set_attribute("final_answer_length", len(final_answer))

             # Yield 2: Update chatbot with final answer
             # The outputs list in .click/.submit handles which components get the yielded values
             yield history, memory, None, None


        def clear_chat_history():
            """Clears chat display and memory."""
            new_memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, input_key="input"
            )
            gr.Info("Chat history and memory cleared.")
            return [], new_memory


        # --- Wire Components ---
        config_button.click(
            handle_configuration,
            inputs=[api_key_input, code_dir_input],
            outputs=[config_status_display]
        )
        # Wire generator to UI events
        text_input.submit(
            handle_chat_submission_generator,
            inputs=[text_input, file_input, chatbot, chat_memory_state, tool_choice],
            outputs=[chatbot, chat_memory_state, file_input, text_input]
        )
        submit_button.click(
             handle_chat_submission_generator,
             inputs=[text_input, file_input, chatbot, chat_memory_state, tool_choice],
             outputs=[chatbot, chat_memory_state, file_input, text_input]
        )
        clear_button.click(
            clear_chat_history,
            inputs=None,
            outputs=[chatbot, chat_memory_state]
        )

    # --- Launch Gradio App ---
    print("\nLaunching AI Support Suite...")
    print("Check console for agent thoughts & tool outputs.")
    print("Ensure Phoenix API Key is set in .env for cloud tracing via OpenInference.")
    demo.queue().launch(share=True, server_name="0.0.0.0")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Declare tracer as global FIRST, so assignment below works
    global tracer
    tracer = None

    load_dotenv()
    print("INFO: Attempted to load environment variables from .env file.")

    # --- Phoenix / OpenTelemetry Setup (NEW APPROACH) ---
    phoenix_api_key = os.environ.get("PHOENIX_API_KEY")
    tracer_provider = None # Initialize locally

    if phoenix_integration_enabled and phoenix_api_key:
        print("INFO: PHOENIX_API_KEY found, configuring OTEL via phoenix.otel.register for Phoenix Cloud.")
        try:
            os.environ['PHOENIX_PROJECT_NAME'] = "Support AI Suite"
            os.environ['PHOENIX_COLLECTOR_ENDPOINT'] = "https://app.phoenix.arize.com/v1/traces"
            os.environ['PHOENIX_CLIENT_HEADERS'] = f"api_key={phoenix_api_key}"

            tracer_provider = register_phoenix_otel()
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            print("INFO: LangChainInstrumentor configured.")

            # Assign the obtained tracer to the GLOBAL tracer variable
            tracer = opentelemetry.trace.get_tracer(__name__, tracer_provider=tracer_provider)
            print("INFO: OpenTelemetry tracer obtained and assigned globally.")
            print(f"      Traces will be sent to project '{os.environ['PHOENIX_PROJECT_NAME']}' at {os.environ['PHOENIX_COLLECTOR_ENDPOINT']}")

        except Exception as e:
            print(f"ERROR: Failed to configure Phoenix/OpenInference OTEL: {e}")
            logging.error("Failed to configure Phoenix/OpenInference OTEL", exc_info=True)
            phoenix_integration_enabled = False
            tracer = None # Ensure global tracer is None if setup fails
            tracer_provider = None
    else:
        # Ensure tracer is None if tracing is disabled for any reason
        tracer = None
        tracer_provider = None # Keep local variable consistent
        if phoenix_integration_enabled: # Case where key was missing
             print("INFO: PHOENIX_API_KEY not found. Tracing disabled.")
        else: # Case where modules weren't loaded
             print("INFO: Required tracing modules not loaded. Tracing disabled.")


    # --- Default Directory and API Key Checks ---
    if not os.path.exists(DEFAULT_CODE_DIR):
        try:
            os.makedirs(DEFAULT_CODE_DIR)
            logging.warning(f"Default code directory '{DEFAULT_CODE_DIR}' created.")
            print(f"INFO: Default code directory '{DEFAULT_CODE_DIR}' created.")
        except OSError as e:
            logging.error(f"Failed to create default code directory '{DEFAULT_CODE_DIR}': {e}")
            print(f"ERROR: Failed to create default code directory '{DEFAULT_CODE_DIR}': {e}")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("--------------------------------------------------------------------")
        print("WARNING: GOOGLE_API_KEY environment variable not found.")
        print("         Please provide the API key in the Configuration UI below")
        print("         or create a '.env' file with GOOGLE_API_KEY='your_key'.")
        print("--------------------------------------------------------------------")

    # --- Run the App ---
    run_gradio_app()