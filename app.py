"""Streamlit UI: sidebar (model settings, scenario picker, document upload) +
in-character chat loop + end-of-session feedback."""

import time
import uuid

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import CLOUD_PROVIDERS, EMBED_MODEL_NAME, LOCAL_SERVER_TYPES
from ingest import ingest_files, ingest_text
from llm import build_model_string, fetch_local_models, generate_response
from persona import (
    build_context_block,
    build_feedback_messages,
    build_retrieval_query,
    build_scene_block,
    build_stop_sequences,
    build_system_prompt,
    trim_to_single_turn,
)
from retrieve import retrieve_chunks
from scenarios import CUSTOM_LOCATION_OPTION, DEFAULT_MOOD, MOOD_OPTIONS, SCENARIO_ORDER, SCENARIOS, get_scenario

st.set_page_config(page_title="Roleplay RAG", page_icon="🎭")

MANUAL_NOTES_FILENAME = "manual notes"


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_chroma_client():
    # In-memory only: nothing persists once the process stops, unlike a PersistentClient.
    return chromadb.EphemeralClient()


def new_conversation_state(scenario: dict) -> dict:
    return {
        "messages": [{"role": "assistant", "content": scenario["opening_message"]}],
        "ingested_files": set(),
        "feedback": None,
        "uploader_key": 0,
        "manual_key": 0,
    }


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "active_scenario_id" not in st.session_state:
        st.session_state.active_scenario_id = SCENARIO_ORDER[0]
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "local_models_cache" not in st.session_state:
        st.session_state.local_models_cache = {}


def get_conversation(scenario_id: str) -> dict:
    if scenario_id not in st.session_state.conversations:
        st.session_state.conversations[scenario_id] = new_conversation_state(get_scenario(scenario_id))
    return st.session_state.conversations[scenario_id]


def get_collection(scenario_id: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=f"session_{st.session_state.session_id}_{scenario_id}")


def render_technical_details(details: dict):
    st.markdown(f"**Model used:** `{details['model_string']}`")
    st.markdown(
        f"**Latency:** retrieval {details['retrieval_ms']:.0f} ms, "
        f"generation {details['generation_ms']:.0f} ms"
    )
    if details["chunks"]:
        st.markdown("**Retrieved chunks grounding this reply (by distance, lower = more similar):**")
        st.table([
            {
                "filename": c["filename"],
                "page": c["page"],
                "distance": round(c["distance"], 4),
                "text": c["text"][:120] + ("..." if len(c["text"]) > 120 else ""),
            }
            for c in details["chunks"]
        ])
    else:
        st.caption("No document uploaded — this reply was improvised.")


init_session_state()
embedder = get_embedder()

# ---------------- Sidebar ----------------
SOURCE_OPTIONS = ["Local server"] + list(CLOUD_PROVIDERS.keys())

with st.sidebar:
    st.title("RAG RoleApp app")
    st.caption("Built by [Sai Thejas Manjunath](https://saithejas.com)")

    api_key = None
    api_base = None
    needs_key = False

    with st.expander("Settings", expanded=False):
        source = st.selectbox("LLM Source", SOURCE_OPTIONS, index=0)

        if source == "Local server":
            server_type = st.selectbox("Server type", list(LOCAL_SERVER_TYPES.keys()), index=0)
            server_config = LOCAL_SERVER_TYPES[server_type]
            prefix = server_config["litellm_prefix"]

            base_url = st.text_input(
                "Server URL",
                value=server_config["default_base_url"],
                key=f"base_url_{server_type}",
            )
            api_base = base_url

            cache_key = f"{server_type}|{base_url}"
            col_refresh, _ = st.columns([1, 2])
            refresh_clicked = col_refresh.button("Refresh models")

            if cache_key not in st.session_state.local_models_cache or refresh_clicked:
                try:
                    st.session_state.local_models_cache[cache_key] = fetch_local_models(server_type, base_url)
                except RuntimeError as e:
                    st.session_state.local_models_cache[cache_key] = []
                    st.error(str(e))

            available_models = st.session_state.local_models_cache.get(cache_key, [])
            if available_models:
                model_name = st.selectbox("Model", available_models)
            else:
                model_name = st.text_input("Model name (couldn't fetch list — enter manually)")

            api_key = st.text_input("API key (only if your server requires one)", type="password") or None
            if prefix == "openai" and not api_key:
                # litellm's OpenAI client requires a non-empty key even when the local
                # server (LM Studio, llama.cpp, vLLM) doesn't actually check it.
                api_key = "not-needed"
        else:
            prefix = CLOUD_PROVIDERS[source]["prefix"]
            needs_key = True
            api_key = st.text_input("API Key", type="password")
            model_name = st.text_input("Model name", value=CLOUD_PROVIDERS[source]["default_model"])

    model_string = build_model_string(prefix, model_name) if model_name else None

    st.divider()

    scenario_labels = {sid: f"{SCENARIOS[sid]['icon']} {SCENARIOS[sid]['display_name']}" for sid in SCENARIO_ORDER}
    selected_label = st.selectbox(
        "Scenario",
        [scenario_labels[sid] for sid in SCENARIO_ORDER],
        index=SCENARIO_ORDER.index(st.session_state.active_scenario_id),
    )
    st.session_state.active_scenario_id = next(
        sid for sid in SCENARIO_ORDER if scenario_labels[sid] == selected_label
    )

    scenario_id = st.session_state.active_scenario_id
    scenario = get_scenario(scenario_id)
    conversation = get_conversation(scenario_id)
    collection = get_collection(scenario_id)

    st.markdown("**Scene**")
    location_choice = st.selectbox(
        "Location / setting",
        scenario["location_options"] + [CUSTOM_LOCATION_OPTION],
        key=f"location_choice_{scenario_id}",
    )
    if location_choice == CUSTOM_LOCATION_OPTION:
        location = st.text_input(
            "Describe the location", key=f"location_custom_{scenario_id}"
        ).strip() or scenario["location_options"][0]
    else:
        location = location_choice

    mood = st.selectbox(
        "Mood / tone",
        MOOD_OPTIONS,
        index=MOOD_OPTIONS.index(DEFAULT_MOOD),
        key=f"mood_{scenario_id}",
    )

    st.divider()

    uploaded_files = st.file_uploader(
        scenario["doc_upload_label"], type=["pdf", "txt"], accept_multiple_files=True,
        key=f"uploader_{scenario_id}_{conversation['uploader_key']}",
    )
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in conversation["ingested_files"]]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                n_chunks = ingest_files(new_files, collection, embedder)
            for f in new_files:
                conversation["ingested_files"].add(f.name)
            st.success(f"Indexed {n_chunks} chunks from {len(new_files)} file(s).")

    manual_text = st.text_area(
        scenario["manual_input_label"],
        key=f"manual_{scenario_id}_{conversation['manual_key']}",
        height=140,
    )
    if st.button("Add details", key=f"save_manual_{scenario_id}"):
        text = manual_text.strip()
        with st.spinner("Indexing your notes..."):
            n_chunks = ingest_text(text, MANUAL_NOTES_FILENAME, collection, embedder)
        if text:
            conversation["ingested_files"].add(MANUAL_NOTES_FILENAME)
            st.success(f"Indexed {n_chunks} chunk(s) from your notes.")
        else:
            conversation["ingested_files"].discard(MANUAL_NOTES_FILENAME)
            st.info("Cleared your typed notes.")

    st.caption(f"{collection.count()} chunks indexed from {len(conversation['ingested_files'])} source(s).")

    st.divider()

    if st.button("Start new session", key=f"reset_{scenario_id}"):
        get_chroma_client().delete_collection(name=f"session_{st.session_state.session_id}_{scenario_id}")
        fresh = new_conversation_state(scenario)
        fresh["uploader_key"] = conversation["uploader_key"] + 1
        fresh["manual_key"] = conversation["manual_key"] + 1
        st.session_state.conversations[scenario_id] = fresh
        st.rerun()

# ---------------- Main area ----------------
st.title(f"{scenario['icon']} {scenario['display_name']}")
st.caption(f"AI plays: **{scenario['ai_role']}**  |  You play: **{scenario['user_role']}**")
st.caption(f"📍 {location}  |  🎭 Mood: {mood}")
st.markdown(scenario["description"])

for message in conversation["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("technical_details"):
            with st.expander("Technical Details"):
                render_technical_details(message["technical_details"])

# Captured here so it's handled and rendered immediately below, in the natural
# chat flow, above the feedback section -- st.chat_input still visually pins
# to the bottom of the page regardless of where it's called in source order.
query = st.chat_input(f"Respond as the {scenario['user_role']}...")
if query:
    if not model_name:
        st.warning("Pick or enter a model name in the sidebar first.")
        st.stop()

    if needs_key and not api_key:
        st.warning(f"Enter an API key for {source}, or switch to a Local server.")
        st.stop()

    conversation["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieval_start = time.perf_counter()
            retrieval_query = build_retrieval_query(scenario, conversation["messages"])
            chunks, _ = retrieve_chunks(retrieval_query, collection, embedder)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            context_block = build_context_block(chunks)
            scene_block = build_scene_block(location, mood)
            system_prompt = build_system_prompt(scenario, context_block, scene_block)
            history = [{"role": m["role"], "content": m["content"]} for m in conversation["messages"]]
            messages = [{"role": "system", "content": system_prompt}, *history]

            generation_start = time.perf_counter()
            try:
                answer = generate_response(
                    model_string, messages, api_key=api_key, api_base=api_base,
                    stop=build_stop_sequences(scenario),
                )
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            answer = trim_to_single_turn(answer, scenario)
            generation_ms = (time.perf_counter() - generation_start) * 1000

        st.markdown(answer)
        technical_details = {
            "model_string": model_string,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "chunks": chunks,
        }
        with st.expander("Technical Details"):
            render_technical_details(technical_details)

    conversation["messages"].append({
        "role": "assistant",
        "content": answer,
        "technical_details": technical_details,
    })

st.divider()
feedback_label = "Regenerate feedback" if conversation["feedback"] else "End session & get feedback"
if st.button(feedback_label):
    has_user_turn = any(m["role"] == "user" for m in conversation["messages"])
    if not has_user_turn:
        st.info("Chat a bit first before requesting feedback.")
    elif not model_name:
        st.warning("Pick or enter a model name in the sidebar first.")
    elif needs_key and not api_key:
        st.warning(f"Enter an API key for {source}, or switch to a Local server.")
    else:
        with st.spinner("Evaluating your responses..."):
            feedback_messages = build_feedback_messages(scenario, conversation["messages"])
            try:
                conversation["feedback"] = generate_response(
                    model_string, feedback_messages, api_key=api_key, api_base=api_base
                )
            except RuntimeError as e:
                st.error(str(e))

if conversation["feedback"]:
    with st.container(border=True):
        st.markdown(conversation["feedback"])
