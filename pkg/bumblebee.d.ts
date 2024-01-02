/* tslint:disable */
/* eslint-disable */
/**
* @param {string} name
*/
export function greet(name: string): void;
/**
*/
export class WasmModel {
  free(): void;
/**
* @returns {WasmModel}
*/
  static new(): WasmModel;
/**
* @param {Uint8Array} weights
*/
  constructor(weights: Uint8Array);
/**
* @param {Uint8Array} image
* @returns {Float32Array}
*/
  predict_image(image: Uint8Array): Float32Array;
/**
* @returns {string}
*/
  info(): string;
}
