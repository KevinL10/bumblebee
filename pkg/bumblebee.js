import * as wasm from "./bumblebee_bg.wasm";
import { __wbg_set_wasm } from "./bumblebee_bg.js";
__wbg_set_wasm(wasm);
export * from "./bumblebee_bg.js";
