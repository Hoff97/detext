let _wasmSupport;

export function wasmSupport(): boolean {
  if (_wasmSupport) {
    return _wasmSupport;
  }

  try {
      if (typeof WebAssembly === "object"
          && typeof WebAssembly.instantiate === "function") {
          const module = new WebAssembly.Module(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
          if (module instanceof WebAssembly.Module) {
            _wasmSupport = new WebAssembly.Instance(module) instanceof WebAssembly.Instance;
            return _wasmSupport;
          }
      }
  } catch (e) {
  }
  _wasmSupport = false;
  return _wasmSupport;
}
