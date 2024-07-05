import { mat4 } from "../../lib/gl-matrix-module.js";

import { WebGL } from "../WebGL.js";
import { AbstractRenderer } from "./AbstractRenderer.js";

import { PerspectiveCamera } from "../PerspectiveCamera.js";

const [SHADERS, MIXINS] = await Promise.all(
  ["shaders.json", "mixins.json"].map((url) =>
    fetch(url).then((response) => response.json())
  )
);

export class SpectralRenderer extends AbstractRenderer {
  constructor(gl, volume, camera, environmentTexture, options = {}) {
    super(gl, volume, camera, environmentTexture, options);

    this.ran = Math.random();
    this.registerProperties([
      // {
      //   name: "bins",
      //   label: "Bins",
      //   type: "spinner",
      //   value: 3,
      //   min: 1,
      //   max: 20,
      // },
      {
        name: "extinction",
        label: "Extinction",
        type: "spinner",
        value: 6,
        min: 0,
      },
      {
        name: "anisotropy",
        label: "Anisotropy",
        type: "slider",
        value: 0,
        min: -1,
        max: 1,
      },
      {
        name: "bounces",
        label: "Max bounces",
        type: "spinner",
        value: 8,
        min: 0,
      },
      {
        name: "steps",
        label: "Steps",
        type: "spinner",
        value: 10,
        min: 0,
      },
      {
        name: "lightPos",
        label: "Light Direction",
        type: "slider",
        value: 1,
        min: -1,
        max: 1,
      },
      {
        name: "brightness",
        label: "Light Intensity",
        type: "spinner",
        value: 1,
        // min: 0,
      },
      // {
      //   name: "transferFunction",
      //   label: "Transfer function",
      //   type: "transfer-function",
      //   value: new Uint8Array(256),
      // },
    ]);

    this.addEventListener("change", (e) => {
      const { name, value } = e.detail;

      if (name === "transferFunction") {
        this.setTransferFunction(this.transferFunction);
      }

      if (
        [
          "extinction",
          "bins",
          "anisotropy",
          "bounces",
          "transferFunction",
          "lightPos",
          "brightness",
        ].includes(name)
      ) {
        this.reset();
      }
    });

    this._programs = WebGL.buildPrograms(gl, SHADERS.renderers.SR, MIXINS);
  }

  destroy() {
    const gl = this._gl;
    Object.keys(this._programs).forEach((programName) => {
      gl.deleteProgram(this._programs[programName].program);
    });

    super.destroy();
  }
  _resetFrame() {
    const gl = this._gl;

    const { program, uniforms } = this._programs.reset;
    gl.useProgram(program);

    gl.uniform2f(
      uniforms.uInverseResolution,
      1 / this._resolution,
      1 / this._resolution
    );
    gl.uniform1f(uniforms.uRandSeed, Math.random());
    gl.uniform1f(uniforms.uBlur, 0);

    const centerMatrix = mat4.fromTranslation(
      mat4.create(),
      [-0.5, -0.5, -0.5]
    );
    const modelMatrix = this._volumeTransform.globalMatrix;
    const viewMatrix = this._camera.transform.inverseGlobalMatrix;
    const projectionMatrix =
      this._camera.getComponent(PerspectiveCamera).projectionMatrix;

    const matrix = mat4.create();
    mat4.multiply(matrix, centerMatrix, matrix);
    mat4.multiply(matrix, modelMatrix, matrix);
    mat4.multiply(matrix, viewMatrix, matrix);
    mat4.multiply(matrix, projectionMatrix, matrix);
    mat4.invert(matrix, matrix);
    gl.uniformMatrix4fv(uniforms.uMvpInverseMatrix, false, matrix);

    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
      gl.COLOR_ATTACHMENT4,
    ]);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  _generateFrame() {}

  _integrateFrame() {
    const gl = this._gl;

    const { program, uniforms } = this._programs.integrate;
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[0]
    );
    gl.uniform1i(uniforms.uPosition, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[1]
    );
    gl.uniform1i(uniforms.uDirection, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[2]
    );
    gl.uniform1i(uniforms.uTransmittance, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[3]
    );
    gl.uniform1i(uniforms.uRadiance, 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[4]
    );
    gl.uniform1i(uniforms.uWavelength, 4);

    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_3D, this._volume.getTexture());
    gl.uniform1i(uniforms.uVolume, 5);

    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, this._environmentTexture);
    gl.uniform1i(uniforms.uEnvironment, 6);

    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, this._transferFunction);
    gl.uniform1i(uniforms.uTransferFunction, 7);

    gl.uniform2f(
      uniforms.uInverseResolution,
      1 / this._resolution,
      1 / this._resolution
    );
    gl.uniform1f(uniforms.uRandSeed, Math.random());
    gl.uniform1f(uniforms.uBlur, 0);

    gl.uniform1f(uniforms.uExtinction, this.extinction);
    gl.uniform1f(uniforms.uAnisotropy, this.anisotropy);
    gl.uniform1ui(uniforms.uMaxBounces, this.bounces);
    gl.uniform1ui(uniforms.uSteps, this.steps);
    gl.uniform1f(uniforms.uBrightness, this.brightness);
    gl.uniform1f(uniforms.uLightPos, this.lightPos);

    const centerMatrix = mat4.fromTranslation(
      mat4.create(),
      [-0.5, -0.5, -0.5]
    );
    const modelMatrix = this._volumeTransform.globalMatrix;
    const viewMatrix = this._camera.transform.inverseGlobalMatrix;
    const projectionMatrix =
      this._camera.getComponent(PerspectiveCamera).projectionMatrix;

    const matrix = mat4.create();
    mat4.multiply(matrix, centerMatrix, matrix);
    mat4.multiply(matrix, modelMatrix, matrix);
    mat4.multiply(matrix, viewMatrix, matrix);
    mat4.multiply(matrix, projectionMatrix, matrix);
    mat4.invert(matrix, matrix);
    gl.uniformMatrix4fv(uniforms.uMvpInverseMatrix, false, matrix);

    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
      // gl.COLOR_ATTACHMENT4,
    ]);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Bind the framebuffer we're reading from
    // gl.readBuffer(gl.COLOR_ATTACHMENT3); // Specify which color attachment to read from

    // // Create a buffer to hold the pixel data and read the pixels
    // const pixels = new Float32Array(512 * 512 * 4); // Create a buffer for the data
    // gl.readPixels(0, 0, 512, 512, gl.RGBA, gl.FLOAT, pixels);

    // // Log the RGBA values of the first pixel to see what's there
    // console.log(
    //   "First pixel RGBA: " + pixels[0] + ", " + pixels[1] + ", " + pixels[2]
    // );
    // console.log(
    //   "First pixel RGBA255: " +
    //     255 * pixels[0] +
    //     ", " +
    //     255 * pixels[1] +
    //     ", " +
    //     255 * pixels[2]
    // );
  }

  _renderFrame() {
    const gl = this._gl;

    const { program, uniforms } = this._programs.render;
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[3]
    );
    gl.uniform1i(uniforms.uColor, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(
      gl.TEXTURE_2D,
      this._accumulationBuffer.getAttachments().color[4]
    );
    gl.uniform1i(uniforms.uWavelength, 1);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Bind the default framebuffer
    // gl.readBuffer(gl.BACK); // Ensure we are reading from the default draw buffer

    // const pixels = new Uint8Array(512 * 512 * 4); // 4 components (RGBA), each 1 byte
    // gl.readPixels(0, 0, 512, 512, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // // Log the RGBA values of the first pixel to see what's there
    // console.log(
    //   "First pixel RGBA: " + pixels[0] + ", " + pixels[1] + ", " + pixels[2]
    // );
  }

  _getFrameBufferSpec() {
    const gl = this._gl;
    return [
      {
        width: this._resolution,
        height: this._resolution,
        min: gl.NEAREST,
        mag: gl.NEAREST,
        format: gl.RGBA,
        iformat: gl.RGBA32F,
        type: gl.FLOAT,
      },
    ];
  }

  _getAccumulationBufferSpec() {
    const gl = this._gl;

    const positionBufferSpec = {
      width: this._resolution,
      height: this._resolution,
      min: gl.NEAREST,
      mag: gl.NEAREST,
      format: gl.RGBA,
      iformat: gl.RGBA32F,
      type: gl.FLOAT,
    };

    const directionBufferSpec = {
      width: this._resolution,
      height: this._resolution,
      min: gl.NEAREST,
      mag: gl.NEAREST,
      format: gl.RGBA,
      iformat: gl.RGBA32F,
      type: gl.FLOAT,
    };

    const transmittanceBufferSpec = {
      width: this._resolution,
      height: this._resolution,
      min: gl.NEAREST,
      mag: gl.NEAREST,
      format: gl.RGBA,
      iformat: gl.RGBA32F,
      type: gl.FLOAT,
    };

    const radianceBufferSpec = {
      width: this._resolution,
      height: this._resolution,
      min: gl.NEAREST,
      mag: gl.NEAREST,
      format: gl.RGBA,
      iformat: gl.RGBA32F,
      type: gl.FLOAT,
    };

    const wavelengthBufferSpec = {
      width: this._resolution,
      height: this._resolution,
      min: gl.NEAREST,
      mag: gl.NEAREST,
      format: gl.RED_INTEGER,
      iformat: gl.R32UI,
      type: gl.UNSIGNED_INT,
    };

    return [
      positionBufferSpec,
      directionBufferSpec,
      transmittanceBufferSpec,
      radianceBufferSpec,
      wavelengthBufferSpec,
    ];
  }
}
