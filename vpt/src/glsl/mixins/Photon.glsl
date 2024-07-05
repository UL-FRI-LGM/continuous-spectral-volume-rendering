// #part /glsl/mixins/Photon

struct Photon {
    vec3 position;
    vec3 direction;
    vec3 transmittance;
    vec3 radiance; // only one element
    uint bounces;
    uint samples;
    uint wavelength;
    bool interacted;
};
