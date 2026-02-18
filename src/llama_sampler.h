#ifndef GODOT_LLAMA_SAMPLER_H
#define GODOT_LLAMA_SAMPLER_H

#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {

class LlamaSampler : public RefCounted {
    GDCLASS(LlamaSampler, RefCounted);

private:
    double temperature = 0.8;
    double top_p = 0.95;
    int seed = -1;

protected:
    static void _bind_methods();

public:
    void set_temperature(double p_temperature);
    double get_temperature() const;

    void set_top_p(double p_top_p);
    double get_top_p() const;

    void set_seed(int p_seed);
    int get_seed() const;
};

} // namespace godot

#endif
