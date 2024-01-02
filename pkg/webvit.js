import * as wasm from "./webvit_bg.wasm";
import { __wbg_set_wasm } from "./webvit_bg.js";
__wbg_set_wasm(wasm);
export * from "./webvit_bg.js";
