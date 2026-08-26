import { describe, it, expect, vi, afterEach } from "vitest";

afterEach(() => {
  vi.doUnmock("@huggingface/transformers");
  vi.resetModules();
});

describe("ClipEmbeddingProvider (package unavailable)", () => {
  it("throws clean install hint when @huggingface/transformers is missing", async () => {
    vi.doMock("@huggingface/transformers");
    vi.resetModules();
    const { ClipEmbeddingProvider: Fresh } = await import(
      "../src/providers/embedding/clip.js"
    );
    await expect(new Fresh().embed("hello")).rejects.toThrow(
      "Install @huggingface/transformers for CLIP embeddings",
    );
  });
});

describe("ClipEmbeddingProvider (with loaded pipeline)", () => {
  function mockSuccessModule() {
    let lastBatchSize = 1;
    const textModel = vi.fn(async () => ({
      text_embeds: { tolist: () => Array.from({ length: lastBatchSize }, () => [0.1, 0.2]) },
    }));
    const tokenizer = vi.fn((texts: string[]) => {
      lastBatchSize = texts.length;
      return { input_ids: texts.map(() => [1, 2, 3]) };
    });
    const fromPretrainedText = vi.fn(async () => textModel);
    const fromPretrainedTokenizer = vi.fn(async () => tokenizer);
    const imageExtractor = vi.fn(async () => ({
      tolist: () => [[0.3, 0.4]],
      data: new Float32Array([0.3, 0.4]),
    }));
    const fromBlob = vi.fn(async () => ({}));
    const pipeline = vi.fn((task: string) => {
      if (task === "image-feature-extraction") return Promise.resolve(imageExtractor);
      return Promise.reject(new Error(`unmocked task: ${task}`));
    });
    vi.doMock("@huggingface/transformers", () => ({
      pipeline,
      AutoTokenizer: { from_pretrained: fromPretrainedTokenizer },
      CLIPTextModelWithProjection: { from_pretrained: fromPretrainedText },
      RawImage: { fromBlob },
    }));
    vi.resetModules();
    return {
      pipeline,
      textModel,
      tokenizer,
      fromPretrainedText,
      fromPretrainedTokenizer,
      imageExtractor,
      fromBlob,
    };
  }

  it("loads the CLIP text tower with dtype: q8 and returns a normalized Float32Array", async () => {
    const { pipeline, fromPretrainedText, fromPretrainedTokenizer } = mockSuccessModule();
    const { ClipEmbeddingProvider: Fresh } = await import(
      "../src/providers/embedding/clip.js"
    );
    const vec = await new Fresh().embed("hello");

    expect(fromPretrainedTokenizer).toHaveBeenCalledWith("Xenova/clip-vit-base-patch32");
    expect(fromPretrainedText).toHaveBeenCalledWith("Xenova/clip-vit-base-patch32", {
      dtype: "q8",
    });
    // Regression guard for #1249: the generic feature-extraction pipeline
    // instantiates the full dual-encoder and demands pixel_values.
    expect(pipeline).not.toHaveBeenCalledWith(
      "feature-extraction",
      expect.anything(),
      expect.anything(),
    );
    expect(vec).toBeInstanceOf(Float32Array);
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it("embedBatch returns one Float32Array per input", async () => {
    mockSuccessModule();
    const { ClipEmbeddingProvider: Fresh } = await import(
      "../src/providers/embedding/clip.js"
    );
    const vecs = await new Fresh().embedBatch(["a", "b"]);

    expect(vecs).toHaveLength(2);
    for (const v of vecs) expect(v).toBeInstanceOf(Float32Array);
  });

  it("embedImage loads image pipeline with dtype: q8 and decodes data: URL", async () => {
    const { pipeline, fromBlob } = mockSuccessModule();
    const { ClipEmbeddingProvider: Fresh } = await import(
      "../src/providers/embedding/clip.js"
    );
    const vec = await new Fresh().embedImage("data:image/png;base64,AAAA");

    expect(pipeline).toHaveBeenCalledWith(
      "image-feature-extraction",
      "Xenova/clip-vit-base-patch32",
      { dtype: "q8" },
    );
    expect(fromBlob).toHaveBeenCalled();
    expect(vec).toBeInstanceOf(Float32Array);
  });

  it("accepts custom model ID via constructor", async () => {
    const { fromPretrainedText, fromPretrainedTokenizer } = mockSuccessModule();
    const { ClipEmbeddingProvider: Fresh } = await import(
      "../src/providers/embedding/clip.js"
    );
    await new Fresh("Xenova/clip-vit-large-patch14").embed("hello");

    expect(fromPretrainedTokenizer).toHaveBeenCalledWith("Xenova/clip-vit-large-patch14");
    expect(fromPretrainedText).toHaveBeenCalledWith("Xenova/clip-vit-large-patch14", {
      dtype: "q8",
    });
  });
});
