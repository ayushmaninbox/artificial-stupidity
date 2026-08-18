/* One list, used by the picker and by the worker's routing. Sizes are the
   real download for each — the tiny ones are ONNX exports of the from-scratch
   checkpoints, which is why AS-4 is 3.8 MB here but 169 KB "packed": ONNX
   stores dequantized fp32 values, exactly as safetensors does. */

export type ModelId =
  | "AS-F" | "AS-0" | "AS-1" | "AS-2" | "AS-3" | "AS-4" | "AS-5"
  | "AS-I" | "AS-I-300" | "AS-IF";

export type ModelInfo = {
  id: ModelId;
  family: "text" | "image";
  ready: boolean;
  size: string;
  blurb: string;
};

export const MODELS: ModelInfo[] = [
  { id: "AS-F",  family: "text", ready: true,  size: "164 MB", blurb: "GPT-2 fine-tune — fluent, and wrong about everything" },
  { id: "AS-0",  family: "text", ready: true,  size: "3.5 MB", blurb: "From scratch, full precision — the control group" },
  { id: "AS-1",  family: "text", ready: true,  size: "3.8 MB", blurb: "8-bit weights · 256 settings each" },
  { id: "AS-2",  family: "text", ready: true,  size: "3.8 MB", blurb: "4-bit weights · 16 settings each" },
  { id: "AS-3",  family: "text", ready: true,  size: "3.8 MB", blurb: "Ternary · −1, 0, +1" },
  { id: "AS-4",  family: "text", ready: true,  size: "3.8 MB", blurb: "1-bit · packs to 169 KB" },
  { id: "AS-5",  family: "text", ready: true,  size: "2.0 MB", blurb: "1-bit, smaller brain · packs to 83 KB" },
  { id: "AS-I",     family: "image", ready: true,  size: "17 MB",  blurb: "Text-to-image, from scratch · 1,254 emoji" },
  { id: "AS-I-300", family: "image", ready: true,  size: "17 MB",  blurb: "Same model, 300 glyphs · sharper" },
  { id: "AS-IF",    family: "image", ready: false, size: "1.2 GB", blurb: "Quantized SD-Turbo · draws anything" },
];

export const byId = (id: string) => MODELS.find((m) => m.id === id) ?? MODELS[0];
