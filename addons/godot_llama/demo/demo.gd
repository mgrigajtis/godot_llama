extends Control

var _llama: GodotLlama
var _streaming_active := false
var _model_path := ""
var _pending_user_text := ""
var _chat_history: Array[Dictionary] = []

const DEFAULT_SYSTEM_PROMPT := "You are Bran, a blacksmith in Oakridge. Stay in character, be concise, and be helpful. Never mention being an AI."
const DEFAULT_WORLD_STATE_PROMPT := "World state: The bridge to Northpass is broken. The player completed quest 'Rat Cellar'."
const DEFAULT_MAX_HISTORY_TURNS := 4

@onready var _select_model_button: Button = $MarginContainer/Scroll/VBox/ModelPathRow/SelectModelButton
@onready var _model_path_value: Label = $MarginContainer/Scroll/VBox/ModelPathRow/ModelPathValue
@onready var _model_file_dialog: FileDialog = $ModelFileDialog
@onready var _load_button: Button = $MarginContainer/Scroll/VBox/ConfigRow/LoadButton
@onready var _create_context_button: Button = $MarginContainer/Scroll/VBox/ConfigRow/CreateContextButton
@onready var _n_ctx_spin: SpinBox = $MarginContainer/Scroll/VBox/ConfigRow/NCtxSpin
@onready var _threads_spin: SpinBox = $MarginContainer/Scroll/VBox/ConfigRow/ThreadsSpin
@onready var _system_prompt_edit: TextEdit = $MarginContainer/Scroll/VBox/SystemPromptEdit
@onready var _world_state_edit: TextEdit = $MarginContainer/Scroll/VBox/WorldStateEdit
@onready var _history_turns_spin: SpinBox = $MarginContainer/Scroll/VBox/MemoryRow/HistoryTurnsSpin
@onready var _clear_memory_button: Button = $MarginContainer/Scroll/VBox/MemoryRow/ClearMemoryButton
@onready var _prompt_edit: TextEdit = $MarginContainer/Scroll/VBox/PromptEdit
@onready var _generate_button: Button = $MarginContainer/Scroll/VBox/GenerateRow/GenerateButton
@onready var _max_tokens_spin: SpinBox = $MarginContainer/Scroll/VBox/GenerateRow/MaxTokensSpin
@onready var _temperature_spin: SpinBox = $MarginContainer/Scroll/VBox/GenerateRow/TemperatureSpin
@onready var _top_p_spin: SpinBox = $MarginContainer/Scroll/VBox/GenerateRow/TopPSpin
@onready var _streaming_check: CheckBox = $MarginContainer/Scroll/VBox/GenerateRow/StreamingCheck
@onready var _output_text: RichTextLabel = $MarginContainer/Scroll/VBox/OutputText
@onready var _status_label: Label = $MarginContainer/Scroll/VBox/StatusLabel

func _ready() -> void:
    _llama = GodotLlama.new()
    _llama.context.token_generated.connect(_on_token_generated)
    _llama.context.generation_finished.connect(_on_generation_finished)
    _llama.context.generation_error.connect(_on_generation_error)

    _select_model_button.pressed.connect(_on_select_model_pressed)
    _model_file_dialog.file_selected.connect(_on_model_file_selected)
    _load_button.pressed.connect(_on_load_model_pressed)
    _create_context_button.pressed.connect(_on_create_context_pressed)
    _clear_memory_button.pressed.connect(_on_clear_memory_pressed)
    _generate_button.pressed.connect(_on_generate_pressed)

    _system_prompt_edit.text = DEFAULT_SYSTEM_PROMPT
    _world_state_edit.text = DEFAULT_WORLD_STATE_PROMPT
    _history_turns_spin.value = DEFAULT_MAX_HISTORY_TURNS
    _set_status("ready")

func _on_select_model_pressed() -> void:
    _model_file_dialog.popup_centered_ratio(0.75)

func _on_model_file_selected(path: String) -> void:
    _model_path = path
    _model_path_value.text = path
    _set_status("model selected")

func _on_load_model_pressed() -> void:
    if _model_path.is_empty():
        _set_status("select a model first")
        return

    _set_status("loading model...")
    var err := _llama.load_model(_model_path)
    if err != OK:
        _set_status("model load failed: %s" % error_string(err))
        return

    _set_status("model loaded")

func _on_create_context_pressed() -> void:
    var params := {
        "n_ctx": int(_n_ctx_spin.value),
        "threads": int(_threads_spin.value),
        "threads_batch": int(_threads_spin.value),
    }
    _set_status("creating context...")
    var err := _llama.create_context(params)
    if err != OK:
        _set_status("create context failed: %s" % error_string(err))
        return

    _set_status("context ready")

func _on_generate_pressed() -> void:
    var user_text := _prompt_edit.text.strip_edges()
    if user_text.is_empty():
        _set_status("prompt is empty")
        return

    var prompt := _build_chat_prompt(user_text)
    var max_tokens := int(_max_tokens_spin.value)
    var params := {
        "temperature": float(_temperature_spin.value),
        "top_p": float(_top_p_spin.value),
    }

    _output_text.clear()
    _pending_user_text = user_text
    if _streaming_check.button_pressed:
        _streaming_active = true
        _set_status("generating (streaming)...")
        _llama.context.set_prompt(prompt)
        _llama.context.generate_stream(max_tokens, params)
        return

    _set_status("generating...")
    var result := _llama.generate(prompt, max_tokens, params)
    var clean_result := _clean_assistant_output(result)
    _output_text.text = clean_result
    _append_chat_turn("user", user_text)
    _append_chat_turn("assistant", clean_result)
    _set_status("generation finished")

func _on_clear_memory_pressed() -> void:
    _chat_history.clear()
    _pending_user_text = ""
    _set_status("memory cleared")

func _on_token_generated(token_text: String, _token_id: int) -> void:
    if _streaming_active:
        _output_text.text += token_text

func _on_generation_finished(_full_text: String) -> void:
    if _streaming_active:
        var clean_text := _clean_assistant_output(_full_text)
        _output_text.text = clean_text
        _append_chat_turn("user", _pending_user_text)
        _append_chat_turn("assistant", clean_text)
        _streaming_active = false
    _set_status("generation finished")

func _on_generation_error(message: String) -> void:
    _streaming_active = false
    _set_status("generation error: %s" % message)

func _set_status(message: String) -> void:
    _status_label.text = "Status: %s" % message

func _build_chat_prompt(user_text: String) -> String:
    var parts := PackedStringArray()
    var clean_system := _system_prompt_edit.text.strip_edges()
    var clean_world := _world_state_edit.text.strip_edges()

    if not clean_system.is_empty():
        parts.append("<|im_start|>system")
        parts.append(clean_system)
        parts.append("<|im_end|>")

    if not clean_world.is_empty():
        parts.append("<|im_start|>system")
        parts.append(clean_world)
        parts.append("<|im_end|>")

    for turn in _chat_history:
        var role := String(turn.get("role", ""))
        var content := String(turn.get("content", "")).strip_edges()
        if role.is_empty() or content.is_empty():
            continue
        parts.append("<|im_start|>%s" % role)
        parts.append(content)
        parts.append("<|im_end|>")

    parts.append("<|im_start|>user")
    parts.append(user_text.strip_edges())
    parts.append("<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)

func _append_chat_turn(role: String, content: String) -> void:
    var clean := content.strip_edges()
    if clean.is_empty():
        return
    _chat_history.append({
        "role": role,
        "content": clean,
    })
    var max_items : int = max(0, int(_history_turns_spin.value)) * 2
    if max_items == 0:
        _chat_history.clear()
        return
    while max_items > 0 and _chat_history.size() > max_items:
        _chat_history.remove_at(0)

func _clean_assistant_output(text: String) -> String:
    var out := text
    var end_idx := out.find("<|im_end")
    if end_idx >= 0:
        out = out.substr(0, end_idx)
    out = out.replace("<|im_start|>assistant", "")
    out = out.replace("<|im_start>assistant", "")
    out = out.replace("<|im_start|> assistant", "")
    out = out.replace("<|im_start> assistant", "")
    out = out.replace("<|im_start|>", "")
    out = out.replace("<|im_start>", "")
    out = out.replace("<|im_end|>", "")
    out = out.replace("<|im_end>", "")
    out = out.replace("<|endoftext|>", "")
    return out.strip_edges()
