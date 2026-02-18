class_name GodotLlama
extends RefCounted

var model: LlamaModel
var context: LlamaContext

func _init() -> void:
    model = LlamaModel.new()
    context = LlamaContext.new()

func load_model(path: String, params: Dictionary = {}) -> Error:
    return model.load(path, params)

func create_context(params: Dictionary = {}) -> Error:
    return context.create(model, params)

func generate(prompt: String, max_tokens: int = 128, params: Dictionary = {}) -> String:
    context.set_prompt(prompt)
    return context.generate(max_tokens, params)
