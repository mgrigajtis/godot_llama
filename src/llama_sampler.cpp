#include "llama_sampler.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void LlamaSampler::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &LlamaSampler::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaSampler::get_temperature);
    ClassDB::bind_method(D_METHOD("set_top_p", "top_p"), &LlamaSampler::set_top_p);
    ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaSampler::get_top_p);
    ClassDB::bind_method(D_METHOD("set_seed", "seed"), &LlamaSampler::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &LlamaSampler::get_seed);

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");
}

void LlamaSampler::set_temperature(double p_temperature) {
    temperature = p_temperature;
}

double LlamaSampler::get_temperature() const {
    return temperature;
}

void LlamaSampler::set_top_p(double p_top_p) {
    top_p = p_top_p;
}

double LlamaSampler::get_top_p() const {
    return top_p;
}

void LlamaSampler::set_seed(int p_seed) {
    seed = p_seed;
}

int LlamaSampler::get_seed() const {
    return seed;
}
