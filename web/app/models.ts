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
  /** Shown in the thread on switching. The image models only understand a
      closed caption grammar, and a user typing "a dragon" at them deserves to
      be told that before they conclude the model is broken. */
  hint?: string;
  examples?: string[];
};

export const MODELS: ModelInfo[] = [
  { id: "AS-F",  family: "text", ready: true,  size: "164 MB", blurb: "GPT-2 fine-tune — fluent, and wrong about everything" },
  { id: "AS-0",  family: "text", ready: true,  size: "3.5 MB", blurb: "From scratch, full precision — the control group" },
  { id: "AS-1",  family: "text", ready: true,  size: "3.8 MB", blurb: "8-bit weights · 256 settings each" },
  { id: "AS-2",  family: "text", ready: true,  size: "3.8 MB", blurb: "4-bit weights · 16 settings each" },
  { id: "AS-3",  family: "text", ready: true,  size: "3.8 MB", blurb: "Ternary · −1, 0, +1" },
  { id: "AS-4",  family: "text", ready: true,  size: "3.8 MB", blurb: "1-bit · packs to 169 KB" },
  { id: "AS-5",  family: "text", ready: true,  size: "2.0 MB", blurb: "1-bit, smaller brain · packs to 83 KB" },
  { id: "AS-I",     family: "image", ready: true,  size: "17 MB",  blurb: "Text-to-image, from scratch · 1,254 emoji",
    hint: "It only draws emoji, and only in the grammar it was trained on: a name, optionally with a size, a position and a background colour.", examples: ["red heart", "pizza", "a small rocket in the top left on a navy background", "a large grinning face in the center on a cream background"] },
  { id: "AS-I-300", family: "image", ready: true,  size: "17 MB",  blurb: "Same model, 300 glyphs · sharper",
    hint: "It only draws emoji, and only in the grammar it was trained on: a name, optionally with a size, a position and a background colour. It knows 300 of the most distinct glyphs, so it draws them more sharply.",
    examples: ["cat face", "strawberry", "a large pizza in the center on a white background"] },
  { id: "AS-IF",    family: "image", ready: true,  size: "454 MB", blurb: "Adapted Stable Diffusion · draws anything",
    hint: "This one draws anything you describe — no grammar, no vocabulary limits. It is a 454 MB download the first time, and slower than the others.",
    examples: ["two astronauts playing chess", "a red car beside a blue house", "an apple on a wooden table"] },
];

export const byId = (id: string) => MODELS.find((m) => m.id === id) ?? MODELS[0];
