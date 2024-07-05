// #part /glsl/shaders/renderers/SR/integrate/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/SR/integrate/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;
precision mediump usampler2D;
precision mediump sampler3D;


#define EPS 1e-5

// #link /glsl/mixins/Photon
@Photon
// #link /glsl/mixins/intersectCube
@intersectCube

@constants
@random/hash/pcg
@random/hash/squashlinear
@random/distribution/uniformdivision
@random/distribution/square
@random/distribution/disk
@random/distribution/sphere
@random/distribution/exponential

@unprojectRand

uniform sampler2D uPosition;
uniform sampler2D uDirection;
uniform sampler2D uTransmittance;
uniform sampler2D uRadiance;
uniform usampler2D uWavelength;

uniform sampler3D uVolume;
uniform sampler2D uTransferFunction;
uniform sampler2D uEnvironment;

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

uniform float uExtinction;
uniform float uAnisotropy;
uniform uint uMaxBounces;
uniform uint uSteps;
uniform float uBrightness;
uniform float uLightPos;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;
// layout (location = 4) out uint oWavelength; 

const int numWavelengths = 97; 
const float wavelengths[numWavelengths] = float[](
    300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
    350.0, 355.0, 360.0, 365.0, 370.0, 375.0, 380.0, 385.0, 390.0, 395.0,
    400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0,
    450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0,
    500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0,
    550.0, 555.0, 560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0,
    600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0, 645.0,
    650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0,
    700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0,
    750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0
);
const float values[numWavelengths] = float[](
    0.00, 0.00, 0.00, 0.00, 0.03, 0.33, 1.60, 4.17, 8.76, 15.61,
    24.24, 31.93, 37.94, 43.65, 47.99, 47.23, 46.01, 49.19, 52.63, 67.11,
    81.45, 85.97, 90.20, 90.96, 91.75, 88.40, 85.08, 93.98, 102.94, 109.14,
    115.49, 116.30, 117.08, 115.89, 114.62, 115.30, 115.97, 112.55, 109.12, 109.48,
    109.82, 109.12, 108.41, 106.93, 105.42, 106.84, 108.25, 106.54, 104.83, 104.57,
    104.29, 102.16, 100.00, 98.03, 96.01, 95.50, 94.97, 91.19, 87.44, 87.84,
    88.27, 87.86, 87.41, 86.16, 84.90, 82.50, 80.13, 80.05, 79.97, 77.97,
    75.97, 75.83, 75.69, 76.44, 77.15, 74.94, 72.77, 68.58, 64.42, 65.04,
    65.62, 66.60, 67.57, 61.51, 55.50, 59.00, 62.46, 64.50, 66.48, 61.13,
    55.83, 48.09, 40.41, 49.09, 57.70, 56.00, 54.30
);
uint sampleWavelengthD65(inout uint state) {
    // Step 1: Compute a prefix sum or cumulative sum array of the values to form a CDF
    float cdf[numWavelengths];
    cdf[0] = values[0];
    for (int i = 1; i < numWavelengths; i++) {
        cdf[i] = cdf[i - 1] + values[i];
    }

    // Step 2: Normalize the CDF so that the last element is 1.0
    for (int i = 0; i < numWavelengths; i++) {
        cdf[i] = cdf[i] / cdf[numWavelengths - 1];
    }

    // Step 3: Use the uniform random value to pick a wavelength based on the CDF
    float randomSample = random_uniform(state);
    for (int i = 0; i < numWavelengths; i++) {
        if (randomSample < cdf[i]) {
            return uint(wavelengths[i]);
        }
    }
    return uint(wavelengths[numWavelengths - 1]); // Return the last one if nothing else was picked
}


uint sampleWavelength(inout uint state) {
    return uint(random_uniform(state) * 320.0) + 380u; 
}

void resetPhoton(inout uint state, inout Photon photon) {
    vec3 from, to;
    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    photon.direction = normalize(to - from);
    photon.bounces = 0u;
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);

    photon.wavelength = sampleWavelength(state);
    
    // photon.interacted = false;
}

vec4 sampleEnvironmentMap(vec3 d) {
    vec2 texCoord = vec2(atan(d.x, -d.z), asin(-d.y) * 2.0) * INVPI * 0.5 + 0.5;
    return texture(uEnvironment, texCoord);
}

vec4 sampleVolumeColor(vec3 position) {

    vec2 volumeSample = texture(uVolume, position).rg;
    vec4 transferSample = texture(uTransferFunction, volumeSample);

    return transferSample;
    // if (volumeSample.r > 200.0)
    // {
    //     return vec4(0.5, 0.5, 0.5, 1);
    // }
    // return vec4(1, 1, 1, 1);
}

float sampleHenyeyGreensteinAngleCosine(inout uint state, float g) {
    float g2 = g * g;
    float c = (1.0 - g2) / (1.0 - g + 2.0 * g * random_uniform(state));
    return (1.0 + g2 - c * c) / (2.0 * g);
}

vec3 sampleHenyeyGreenstein(inout uint state, float g, vec3 direction) {
    // generate random direction and adjust it so that the angle is HG-sampled
    vec3 u = random_sphere(state);
    if (abs(g) < EPS) {
        return u;
    }
    float hgcos = sampleHenyeyGreensteinAngleCosine(state, g);
    vec3 circle = normalize(u - dot(u, direction) * direction);
    return sqrt(1.0 - hgcos * hgcos) * circle + hgcos * direction;
}

float max3(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float mean3(vec3 v) {
    return dot(v, vec3(1.0 / 3.0));
}

float gaussian(float lambda, float peak, float width) {
    return exp(-0.5 * pow((lambda - peak) / width, 2.0));
}

float gaussianLn(float lambda, float peak, float width) {
    return exp(-0.5 * pow((log(lambda) - log(peak)) / width, 2.0));
}

// float xBar(uint wavelength) {
//     float lambda = float(wavelength);
//     float peak = 590.0; // Peak response for the red component
//     float width = 40.0; // Width of the Gaussian
//     return gaussian(lambda, peak, width);
// }

// float yBar(uint wavelength) {
//     float lambda = float(wavelength); // Convert the wavelength to a float for calculation
//     float peak = 550.0; // Peak response for the green component
//     float width = 40.0; // Width of the Gaussian
//     return gaussian(lambda, peak, width);
// }

// float zBar(uint wavelength) {
//     float lambda = float(wavelength); // Convert the wavelength to a float for calculation
//     float peak = 450.0; // Peak response for the blue component
//     float width = 40.0; // Width of the Gaussian
//     return gaussian(lambda, peak, width);
// }

float normalizationFactor = 106.34512326191066;

float xBar(uint wavelength) {

    if (wavelength < 380u || wavelength > 700u) {
        return 0.0;
    }
    float lambda = float(wavelength);
    float peak = 595.8; // Peak response for the red component
    float width = 33.33; // Width of the Gaussian
    float peak2 = 446.8; // Peak response for the red component
    float width2 = 19.44; // Width of the Gaussian
    return 1.065 * gaussian(lambda, peak, width) + 0.366 * gaussian(lambda, peak2, width2);
    // return gaussian(lambda, peak, width);

}

float yBar(uint wavelength) {

    
    if (wavelength < 380u || wavelength > 700u) {
        return 0.0;
    }

    float lambda = float(wavelength); // Convert the wavelength to a float for calculation
    float peak = 556.3; // Peak response for the green component
    float width = 0.075; // Width of the Gaussian
    return 1.014 * gaussianLn(lambda, peak, width);
}

float zBar(uint wavelength) {
    
    
    if (wavelength < 380u || wavelength > 700u) {
        return 0.0;
    }

    float lambda = float(wavelength); // Convert the wavelength to a float for calculation
    float peak = 449.8; // Peak response for the blue component
    float width = 0.051; // Width of the Gaussian
    return 1.839 * gaussianLn(lambda, peak, width);
}


vec3 RGBtoXYZ(vec3 RGB) {
    // First, apply inverse gamma correction to convert sRGB to Linear RGB
    vec3 linearRGB = vec3(
        RGB.r > 0.04045 ? pow((RGB.r + 0.055) / 1.055, 2.4) : RGB.r / 12.92,
        RGB.g > 0.04045 ? pow((RGB.g + 0.055) / 1.055, 2.4) : RGB.g / 12.92,
        RGB.b > 0.04045 ? pow((RGB.b + 0.055) / 1.055, 2.4) : RGB.b / 12.92
    );

    // Transformation matrix from sRGB to XYZ (D65 white point)
    mat3 M = mat3(
        0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505
    );

    // Apply the matrix to get XYZ values
    return M * linearRGB;
}

vec3 XYZtoRGB(vec3 XYZ) {
    // Transformation matrix from XYZ to linear RGB, assuming D65 white point (sRGB)
    mat3 M = mat3( 3.2406, -1.5372, -0.4986,
                  -0.9689,  1.8758,  0.0415,
                   0.0557, -0.2040,  1.0570 );

    vec3 RGB = M * XYZ; // Convert to linear RGB

    // Apply gamma correction (sRGB)
    RGB = vec3(RGB.r <= 0.0031308 ? 12.92 * RGB.r : 1.055 * pow(RGB.r, 1.0 / 2.4) - 0.055,
               RGB.g <= 0.0031308 ? 12.92 * RGB.g : 1.055 * pow(RGB.g, 1.0 / 2.4) - 0.055,
               RGB.b <= 0.0031308 ? 12.92 * RGB.b : 1.055 * pow(RGB.b, 1.0 / 2.4) - 0.055);

    RGB = clamp(RGB, 0.0, 1.0);

    return RGB; // These are the sRGB values suitable for display
}

float getAnisotropy(uint wavelength) {
    float normalizedWavelength = (float(wavelength) - 380.0) / 320.0;

    // return 0.8 - 0.6 * normalizedWavelength;
    // return 0.0;

    // return pow((normalizedWavelength), 4.0);

    if (wavelength > 555u && wavelength < 610u) {
        return 1.0;
    } else if (wavelength > 470u && wavelength < 500u) {
        return 0.0;
    }
    
    // if (wavelength > 620u && wavelength < 670u) {
    //     return 0.8;
    // }

    // if (wavelength > 425u && wavelength < 450u) {
    //     return -0.8;
    // }

    // More forward scattering for reds, less for blues
    // return 0.5 + 0.5 * (normalizedWavelength); // Range from 0.5 to 1.0
}


float getAlbedo(uint wavelength, float density) {
    float normalizedWavelength = (float(wavelength) - 380.0) / 320.0;

    // Simple spectral albedo model: higher albedo at lower wavelengths
    float albedo = 1.0 - normalizedWavelength;

    albedo = 0.0;


    if (wavelength > 555u && wavelength < 610u) {
        return 1.0;
    } else if (wavelength > 470u && wavelength < 500u) {
        return 0.5;
    }
    

    // if (density > 100.0 && wavelength > 300)
    // {

    // }
    // float albedo = 0.075;

    // if (density > 0.5) {
    //     if (wavelength > 600u && wavelength < 680u) {
    //         return 1.0;
    //     } else if (wavelength > 400u && wavelength < 440u)
    //     {
    //         return 0.6;
    //     }
    // } else {
    //     if (wavelength > 400u && wavelength < 440u) {
    //         return 1.0;
    //     }
    // }

    return albedo;
}

vec2 sampleSpectralTF(vec3 position, uint wavelength) {

    vec2 volumeSample = texture(uVolume, position).rg;
    vec4 transferSample = texture(uTransferFunction, volumeSample);

    // float density = texture(uVolume, position).r;  // Assuming .r is the correct channel for density
    float density = transferSample.a;
    
    float albedo = getAlbedo(wavelength, density);

    if (density > 0.0) {
         density = 0.8;
    }

    return vec2(albedo, density);  // Properly return a vec2 with both albedo and density
}

float sampleLight(vec3 photonDirection, vec3 lightDirection) {
    return max(dot(photonDirection, lightDirection), 0.0);
}

// This function computes the contribution from a point light at a given position with a specified radius
float samplePointLight(vec3 photonPosition, vec3 photonDirection, vec3 lightPosition, float lightRadius) {
    vec3 lightVector = lightPosition - photonPosition;
    float distanceSquared = dot(lightVector, lightVector);
    float radiusSquared = lightRadius * lightRadius;
    
    if (distanceSquared > radiusSquared) return 0.0; // No contribution if outside the light radius

    vec3 lightDirection = normalize(lightVector);
    float attenuation = max(0.2, 1.0 - distanceSquared / radiusSquared);
    return max(dot(photonDirection, lightDirection), 0.0) * attenuation;
}

void main() {
    // 288
    Photon photon;
    vec2 mappedPosition = vPosition * 0.5 + 0.5;
    photon.position = texture(uPosition, mappedPosition).xyz;
    vec4 directionAndBounces = texture(uDirection, mappedPosition);
    photon.direction = directionAndBounces.xyz;
    photon.bounces = uint(directionAndBounces.w + 0.5);
    photon.transmittance = texture(uTransmittance, mappedPosition).rgb;
    vec4 radianceAndSamples = texture(uRadiance, mappedPosition);
    photon.radiance = radianceAndSamples.rgb;
    photon.samples = uint(radianceAndSamples.w + 0.5);
    photon.wavelength = uint(texture(uWavelength, mappedPosition).r);
    photon.interacted = false;
    // vec3 lightPosition = vec3(0.5, 1.0, 0.5);
    // float lightRadius = 0.5;
    // vec3 lightDirection = normalize(vec3(1.0 - uLightPos, 0.0, uLightPos));
    float angle = (uLightPos + 1.0) * 3.14159265; // angle in radians

    // Calculate the directional vector for the light
    vec3 lightDirection = normalize(vec3(0.0, cos(angle), sin(angle)));

    uint state = hash(uvec3(floatBitsToUint(mappedPosition.x), floatBitsToUint(mappedPosition.y), floatBitsToUint(uRandSeed)));
    photon.wavelength = sampleWavelength(state);
    for (uint i = 0u; i < uSteps; i++) {
        
        float dist = random_exponential(state, uExtinction);
        // float dist = 1.0;
        vec3 oldPosition = photon.position;
        photon.position += dist * photon.direction;

        // vec4 volumeSample = sampleVolumeColor(photon.position);

        vec2 volumeSample = sampleSpectralTF(photon.position, photon.wavelength);
        float albedo = volumeSample.x;
        float trueExtinction = volumeSample.y;

        // float PNull = 1.0 - volumeSample.a;
        float PNull = 1.0 - trueExtinction;
        // float PNull = 1.0;
        float PScattering;
        if (photon.bounces >= uMaxBounces) {
            PScattering = 0.0;
        } else {
            // PScattering = volumeSample.a * max3(volumeSample.rgb);
            PScattering = trueExtinction * albedo;
            // PScattering = 0.0; 
        }
        float PAbsorption = 1.0 - PNull - PScattering;

        float fortuneWheel = random_uniform(state);
        // float fortuneWheel = 1.0;
        if (any(greaterThan(photon.position, vec3(1))) || any(lessThan(photon.position, vec3(0)))) {
            // out of bounds

            photon.samples++;

            // vec3 xyz = RGBtoXYZ(photon.radiance.rgb);
            
            // xyz.x = ((xyz.x * float(photon.samples - 1u)) + xBar(photon.wavelength)) / float(photon.samples);
            // xyz.y = ((xyz.y * float(photon.samples - 1u)) + yBar(photon.wavelength)) / float(photon.samples);
            // xyz.z = ((xyz.z * float(photon.samples - 1u)) + zBar(photon.wavelength)) / float(photon.samples);
            
            // float lightContribution = samplePointLight(photon.position, photon.direction, lightPosition, lightRadius);
            float lightContribution = sampleLight(photon.direction, lightDirection);


            // ui skaliranja
            vec3 radiance = lightContribution * (uBrightness + 3.0) * vec3(xBar(photon.wavelength), yBar(photon.wavelength), zBar(photon.wavelength));
            // vec3 radiance = vec3(xBar(photon.wavelength), yBar(photon.wavelength), zBar(photon.wavelength));

            // radiance = XYZtoRGB(radiance)

            photon.radiance += (radiance - photon.radiance) / float(photon.samples);

            // float rand = random_uniform(state);
            // photon.radiance = vec3(rand, rand, rand);

            // photon.radiance = vec3(0.24860, 0.13255, 0.04863);
   
            // vec3 rgb = XYZtoRGB(xyz);

            // photon.radiance.rgb = rgb;

            // float x = xBar(photon.wavelength);
            // float y = yBar(photon.wavelength);
            // float z = zBar(photon.wavelength);  
            
            // vec3 rgb = XYZtoRGB(vec3(x, y, z));

            // photon.radiance.rgb = rgb;

            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption) {
            // absorption
            photon.samples++;
            photon.interacted = true;
            vec3 radiance = vec3(0); // No emission

            // if (photon.wavelength < 380u) {
            //     uint floroWavelength = 550u;
            //     float sampledFloroWavelength = float(floroWavelength) * gaussian(float(photon.wavelength), 330.0, 60.0);
            //     // float sampledFloroWavelength = 400.0;
            //     radiance =  4.0 * vec3(xBar(uint(sampledFloroWavelength)), yBar(uint(sampledFloroWavelength)), zBar(uint(sampledFloroWavelength)));
            // }

            photon.radiance += (radiance - photon.radiance) / float(photon.samples);

            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption + PScattering) {
            // scattering
            photon.bounces++;
            photon.interacted = true;

            // photon.transmittance.r 
            float anisotropy = getAnisotropy(photon.wavelength);
            photon.direction = sampleHenyeyGreenstein(state, anisotropy, photon.direction);
            // photon.direction = scatterDirection(state, photon.direction, photon.wavelength);
        } else {
            // null collision
        }
    }

    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, 0);
    oRadiance = vec4(photon.radiance, float(photon.samples));
    // oWavelength = photon.wavelength;
}

// #part /glsl/shaders/renderers/SR/render/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/SR/render/fragment

#version 300 es
precision highp float;
precision mediump usampler2D; // For unsigned integer samplers

uniform sampler2D uRadiance;
uniform usampler2D uWavelength; // Changed from sampler2D to usampler2D

in vec2 vPosition;
out vec4 oColor;

vec3 RGBtoXYZ(vec3 RGB) {
    // First, apply inverse gamma correction to convert sRGB to Linear RGB
    vec3 linearRGB = vec3(
        RGB.r > 0.04045 ? pow((RGB.r + 0.055) / 1.055, 2.4) : RGB.r / 12.92,
        RGB.g > 0.04045 ? pow((RGB.g + 0.055) / 1.055, 2.4) : RGB.g / 12.92,
        RGB.b > 0.04045 ? pow((RGB.b + 0.055) / 1.055, 2.4) : RGB.b / 12.92
    );

    // Transformation matrix from sRGB to XYZ (D65 white point)
    mat3 M = mat3(
        0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505
    );

    // Apply the matrix to get XYZ values
    return M * linearRGB;
}

vec3 XYZtoRGB(vec3 XYZ) {
    // Transformation matrix from XYZ to linear RGB, assuming D65 white point (sRGB)
    mat3 M = transpose(mat3(3.2406, -1.5372, -0.4986,
                        -0.9689,  1.8758,  0.0415,
                         0.0557, -0.2040,  1.0570));

    vec3 RGB = M * XYZ; // Convert to linear RGB
    
    // tone mapper naredi to
    // Apply gamma correction (sRGB)
    // RGB = vec3(RGB.r <= 0.0031308 ? 12.92 * RGB.r : 1.055 * pow(RGB.r, 1.0 / 2.4) - 0.055,
    //            RGB.g <= 0.0031308 ? 12.92 * RGB.g : 1.055 * pow(RGB.g, 1.0 / 2.4) - 0.055,
    //            RGB.b <= 0.0031308 ? 12.92 * RGB.b : 1.055 * pow(RGB.b, 1.0 / 2.4) - 0.055);

    RGB = clamp(RGB, 0.0, 1.0);

    return RGB; // These are the sRGB values suitable for display
}

vec3 xyz2rgb(vec3 xyz) {
    const mat3 XYZ2RGB = mat3(
         3.240481, -1.537152, -0.498536,
        -0.969255,  1.875990,  0.041556,
         0.055647, -0.204041,  1.057311
    );

    return XYZ2RGB * xyz;
}


void main() {
    vec3 radiance = texture(uRadiance, vPosition).rgb;
    uint wavelengthBin = texture(uWavelength, vPosition).r; // Now using uint

    // vec3 outputColor = vec3(0.0, 0.0, 0.0);

    // if (wavelengthBin == 0u) {
    //     outputColor.r = radiance.r;
    // } else if (wavelengthBin == 1u) {
    //     outputColor.g = radiance.g;
    // } else if (wavelengthBin == 2u) {
    //     outputColor.b = radiance.b;
    // }

    // Debug by visualizing the wavelength bins as colors
    // if (wavelengthBin== 0u) {
    //     oColor = vec4(1, 0, 0, 1); // Red for bin 0
    // } else if (wavelengthBin == 1u) {
    //     oColor = vec4(0, 1, 0, 1); // Green for bin 1
    // } else if (wavelengthBin == 2u) {
    //     oColor = vec4(0, 0, 1, 1); // Blue for bin 2
    // } else {
    //     oColor = vec4(1, 1, 1, 1); // White for unexpected values
    // }

    oColor = vec4(XYZtoRGB(radiance), 1.0);
    // oColor = vec4(radiance, 1.0);

}

// #part /glsl/shaders/renderers/SR/reset/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/SR/reset/fragment

#version 300 es
precision mediump float;

// #link /glsl/mixins/Photon
@Photon
// #link /glsl/mixins/intersectCube
@intersectCube

@constants
@random/hash/pcg
@random/hash/squashlinear
@random/distribution/uniformdivision
@random/distribution/square
@random/distribution/disk
@random/distribution/sphere
@random/distribution/exponential

@unprojectRand

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;
layout (location = 4) out uint oWavelength;

const int numWavelengths = 97; 
const float wavelengths[numWavelengths] = float[](
    300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
    350.0, 355.0, 360.0, 365.0, 370.0, 375.0, 380.0, 385.0, 390.0, 395.0,
    400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0,
    450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0,
    500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0,
    550.0, 555.0, 560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0,
    600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0, 645.0,
    650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0,
    700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0,
    750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0
);
const float values[numWavelengths] = float[](
    0.00, 0.00, 0.00, 0.00, 0.03, 0.33, 1.60, 4.17, 8.76, 15.61,
    24.24, 31.93, 37.94, 43.65, 47.99, 47.23, 46.01, 49.19, 52.63, 67.11,
    81.45, 85.97, 90.20, 90.96, 91.75, 88.40, 85.08, 93.98, 102.94, 109.14,
    115.49, 116.30, 117.08, 115.89, 114.62, 115.30, 115.97, 112.55, 109.12, 109.48,
    109.82, 109.12, 108.41, 106.93, 105.42, 106.84, 108.25, 106.54, 104.83, 104.57,
    104.29, 102.16, 100.00, 98.03, 96.01, 95.50, 94.97, 91.19, 87.44, 87.84,
    88.27, 87.86, 87.41, 86.16, 84.90, 82.50, 80.13, 80.05, 79.97, 77.97,
    75.97, 75.83, 75.69, 76.44, 77.15, 74.94, 72.77, 68.58, 64.42, 65.04,
    65.62, 66.60, 67.57, 61.51, 55.50, 59.00, 62.46, 64.50, 66.48, 61.13,
    55.83, 48.09, 40.41, 49.09, 57.70, 56.00, 54.30
);
uint sampleWavelengthD65(inout uint state) {
    // Step 1: Compute a prefix sum or cumulative sum array of the values to form a CDF
    float cdf[numWavelengths];
    cdf[0] = values[0];
    for (int i = 1; i < numWavelengths; i++) {
        cdf[i] = cdf[i - 1] + values[i];
    }

    // Step 2: Normalize the CDF so that the last element is 1.0
    for (int i = 0; i < numWavelengths; i++) {
        cdf[i] = cdf[i] / cdf[numWavelengths - 1];
    }

    // Step 3: Use the uniform random value to pick a wavelength based on the CDF
    float randomSample = random_uniform(state);
    for (int i = 0; i < numWavelengths; i++) {
        if (randomSample < cdf[i]) {
            return uint(wavelengths[i]);
        }
    }
    return uint(wavelengths[numWavelengths - 1]); // Return the last one if nothing else was picked
}

uint sampleWavelength(inout uint state) {
    return uint(random_uniform(state) * 320.0) + 380u; 
}

void main() {
    Photon photon;
    vec3 from, to;
    uint state = hash(uvec3(floatBitsToUint(vPosition.x), floatBitsToUint(vPosition.y), floatBitsToUint(uRandSeed)));
    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    photon.direction = normalize(to - from);
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
    photon.radiance = vec3(0);
    photon.bounces = 0u;
    photon.samples = 0u;
    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, 0);
    oRadiance = vec4(photon.radiance, float(photon.samples));
    oWavelength = sampleWavelength(state);
}
