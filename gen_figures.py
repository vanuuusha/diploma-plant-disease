#!/usr/bin/env python3
"""Generate missing figures for the diploma via OpenRouter + Gemini."""

import json, base64, sys, time, os, asyncio, httpx
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-3.1-flash-image-preview"
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path("диплом/figures")

FIGURES = {
    # Chapter 1 - architecture diagrams
    "fig_1": "Generate a clean academic technical diagram: Timeline of image classification architectures from 2012 to 2025. Top row shows CNN evolution: AlexNet(2012) → VGG(2014) → ResNet(2015) → EfficientNet(2019) → ConvNeXt(2022). Bottom row shows Transformers: ViT(2020) → DeiT(2021) → Swin(2021). Both rows converge into Hybrid models (MobileViT, CoAtNet). Use boxes connected by arrows on a horizontal timeline. White background, clean lines, no decorations, professional academic style.",

    "fig_2": "Generate a clean academic diagram of a generic Convolutional Neural Network (CNN) for image classification. Show: Input Image → [Conv + ReLU + Pooling] repeated 3 times with feature maps getting smaller but deeper → Flatten → Fully Connected layers → Softmax → Output classes. Label each component. White background, clean technical style, horizontal layout.",

    "fig_3": "Generate a clean technical diagram of a ResNet Residual Block. Show input x splitting into two paths: (1) Main path: Conv 3x3 → BatchNorm → ReLU → Conv 3x3 → BatchNorm, (2) Identity shortcut: straight arrow bypassing the main path. Both paths merge with element-wise addition (+) symbol, followed by ReLU. Label all components. White background, vertical layout, academic style.",

    "fig_4": "Generate a clean diagram showing EfficientNet compound scaling. Show 4 network variants as rectangular blocks: (a) Baseline - narrow and shallow, (b) Width scaling - wider blocks, (c) Depth scaling - more stacked blocks, (d) Compound scaling - simultaneously wider, deeper, and higher resolution. Label each variant. White background, academic technical style.",

    "fig_5": "Generate a clean academic diagram of Vision Transformer (ViT) architecture. Show: Input image → split into patches (grid) → Linear Projection of flattened patches → prepend [CLS] token and add Position Embeddings → stack of Transformer Encoder blocks (each with Multi-Head Self-Attention and MLP) → take [CLS] token → MLP Head → Class prediction. White background, clean technical illustration.",

    "fig_6": "Generate a clean technical diagram showing the Shifted Window mechanism in Swin Transformer. Show two feature maps side by side: (a) Layer l with regular non-overlapping window partition shown with solid borders, (b) Layer l+1 with shifted window partition offset by M/2 shown with dashed borders. Use arrows to indicate that elements from different windows in layer l end up in the same window in layer l+1. White background, academic style.",

    "fig_7": "Generate a clean tree/taxonomy diagram of object detection architectures. Root node: 'Object Detection'. Three main branches: (1) Two-stage: Faster R-CNN, Cascade R-CNN; (2) One-stage: SSD, YOLOv5, YOLOv8, YOLOv10, YOLOv12; (3) Transformer-based: DETR, RT-DETR. Additional branch: Foundation Models (SAM, DINOv2, Grounding DINO). Use boxes and connecting lines. White background, academic style.",

    "fig_8": "Generate a clean academic diagram of Faster R-CNN architecture. Show: Input Image → ResNet-50 Backbone → Feature Maps → two branches: (1) Region Proposal Network (RPN) generating proposals, (2) Feature Pyramid Network (FPN). Then RoI Align extracts features for each proposal → FC layers → two outputs: Classification (class) and BBox Regression. White background, block diagram style.",

    "fig_9": "Generate a clean academic diagram of YOLO architecture with three main components. Left: Backbone (CSPDarknet) extracting features at three scales P3, P4, P5 with decreasing spatial resolution. Center: Neck (FPN/PAN) with top-down arrows from P5 to P3 and bottom-up arrows from P3 to P5 for feature fusion. Right: Three Detection Heads at each scale predicting bounding boxes and classes. White background, technical style.",

    "fig_10": "Generate a clean technical diagram comparing Area Attention vs Full Self-Attention. Left panel: a feature map divided into 4 horizontal strips, with self-attention arrows only within each strip (efficient). Right panel: full self-attention where every position connects to every other position (expensive). Label complexity as O(HW^2/n) vs O((HW)^2). White background, academic style.",

    "fig_11": "Generate a clean academic diagram of DETR architecture. Show: Input Image → CNN Backbone → Flatten feature maps + Positional Encoding → Transformer Encoder (N layers) → Transformer Decoder with N learned Object Queries → N parallel predictions, each outputting (class, bounding box). Below: Hungarian Matching connecting predictions to Ground Truth objects. White background, technical block diagram.",

    "fig_12": "Generate a clean technical diagram of FiLM (Feature-wise Linear Modulation) mechanism. Two parallel branches: Left branch is the Main CNN with feature maps F at each layer. Right branch is the Conditioning Network that takes external input c and produces gamma and beta vectors. Arrows show gamma and beta being applied to each layer via channel-wise multiplication and addition: F' = gamma * F + beta. White background, academic style.",

    # Chapter 3
    "fig_3_1": "Generate a detailed YOLOv12 architecture diagram with three columns. Left column 'Backbone': sequence of Conv, C2f blocks with Area Attention modules, producing P3 (80x80), P4 (40x40), P5 (20x20) feature maps. Middle column 'Neck (PAN)': bidirectional feature fusion with top-down and bottom-up arrows connecting P3, P4, P5. Right column 'Detection Head': three independent heads for each scale. White background, technical style.",

    "fig_3_2": "Generate a clean diagram of Area Attention mechanism in YOLOv12. Show a feature map split into 4 horizontal strips. Within each strip, show self-attention connections (arrows between positions). Beside it, show the same map split into vertical strips with attention within each. Below for comparison, show full self-attention on the entire map. Label the complexity reduction. White background, academic style.",

    "fig_3_3": "Generate a clean block diagram of R-ELAN (Residual Efficient Layer Aggregation Network). Show input X splitting into main branch (Conv1x1 → Conv3x3 → Area Attention → Conv3x3, with intermediate output taps from each stage) and a skip connection. Intermediate outputs are concatenated, fused with Conv1x1, then added to the skip connection output. White background, technical diagram style.",

    "fig_3_4": "Generate a clean diagram of Path Aggregation Network (PAN). Show three horizontal levels labeled P3, P4, P5 from top to bottom. Left side: backbone outputs. Top-down pathway: arrows with 'Upsample + Concat' from P5 down to P4, then to P3. Bottom-up pathway: arrows with 'Downsample + Concat' from P3 up to P4, then to P5. Right side: neck outputs. White background, academic technical style.",

    "fig_3_5": "Generate a clean diagram of RT-DETR architecture. Show: Input Image → HGNetV2 Backbone producing S3, S4, S5 feature maps → Hybrid Encoder with two stages: 'Intra-scale Self-Attention (AIFI)' showing three parallel self-attention blocks, then 'Cross-scale Feature Fusion (CCFM)' → IoU-aware Query Selection → Transformer Decoder → N predictions (class + bbox). White background, academic style.",

    "fig_3_6": "Generate a clean diagram of the RT-DETR Hybrid Encoder with two stages. Top stage 'AIFI (Intra-scale)': three parallel streams S3, S4, S5, each going through MHSA+FFN independently. Bottom stage 'CCFM (Cross-scale)': the three outputs merge via upsample/downsample, concatenation, and convolution into unified features. Arrows connecting both stages. White background, technical illustration.",

    # Chapter 4
    "fig_4_1": "Generate a clean block diagram of FiLM layer. Top: 'Context vector c' feeds into two FC layers producing gamma and beta vectors. Bottom: Feature tensor F flows through channel-wise multiplication with gamma, then channel-wise addition with beta, producing output F'. Show the formula F' = gamma * F + beta. White background, academic technical style, clear labels.",

    "fig_4_2": "Generate a clean architecture diagram of CGFM module integrated into YOLOv12. Input image splits into two branches: (1) Main branch: YOLOv12 Backbone → PAN Neck → outputs P3, P4, P5, (2) Context branch: lightweight MobileNetV3-Small (224x224 input) → Projection Head → context vector c (256-dim). Vector c feeds into FiLM layers that modulate P3, P4, P5 outputs before Detection Head. White background, academic style.",

    "fig_4_3": "Generate a grouped bar chart comparing six detection configurations. X-axis labels: Baseline, Late Fusion, SE-Neck, CBAM-Neck, CGFM (P3+P4+P5), CGFM (P5-only). Two bars per group: darker bar for mAP@50 (values ~0.59-0.68), lighter bar for mAP@50-95 (values ~0.30-0.39). Horizontal dashed line at mAP@50=0.651 (baseline). CGFM P5-only should be highlighted as best. Academic matplotlib style.",

    "fig_4_4": "Generate a scatter plot (Pareto diagram). X-axis: number of parameters in millions (range 20-26). Y-axis: mAP@50 (range 0.58-0.69). Points: Baseline (gray, 20.1M, 0.651), Late Fusion (blue, ~21M, 0.588), SE-Neck (orange, ~21M, 0.643), CBAM-Neck (yellow, ~21M, 0.652), CGFM P3+P4+P5 (green, ~22.5M, 0.660), CGFM P5-only (dark green highlighted, ~22M, 0.678). Dashed Pareto frontier. Academic style.",

    # Chapter 5
    "fig_5_1": "Generate a clean three-tier web application architecture diagram. Top layer: Browser/Client (React/Next.js). Middle layer: API Service (FastAPI). Bottom layer: PostgreSQL Database on left, YOLO Detector model in center, Qwen LLM model on right. Arrows show HTTP/REST between client and server, ORM between server and DB, direct calls between server and models. All inside Docker containers. Green accent, minimalist technical style.",

    "fig_5_6": "Generate a clean Entity-Relationship diagram with two PostgreSQL tables. Table 'scans': columns id (PK), created_at, width, height, is_wheat, is_healthy, dominant_diagnosis, status. Table 'regions': columns id (PK), scan_id (FK), x1, y1, x2, y2, detector_probs (JSONB), top_class, description, recommendations (JSONB). One-to-many relationship arrow from scans to regions. Minimalist technical style, monochrome with green accent.",

    "fig_5_7": "Generate a clean horizontal pipeline diagram with 4 sequential steps for disease diagnosis. Step 1: 'Binary Filter' (wheat check). Step 2: 'YOLOv12 + CGFM Detector'. Step 3: 'Qwen-3 LLM (text generation)'. Step 4: 'Save & Return'. Arrows between steps. Below: 'Result saved to DB and returned to client'. Green accent color, minimalist academic style.",
}

sem = asyncio.Semaphore(3)  # max 3 concurrent requests

async def generate_one(client: httpx.AsyncClient, name: str, prompt: str) -> bool:
    out_path = OUT / f"{name}.png"
    if out_path.exists():
        print(f"  SKIP {name} (exists)")
        return True

    async with sem:
        for attempt in range(3):
            try:
                print(f"  GEN  {name} (attempt {attempt+1})...")
                resp = await client.post(
                    URL,
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=120,
                )
                data = resp.json()
                if "error" in data:
                    print(f"  ERR  {name}: {data['error']}")
                    await asyncio.sleep(5)
                    continue

                msg = data["choices"][0]["message"]
                images = msg.get("images", [])
                if not images:
                    print(f"  WARN {name}: no images in response")
                    await asyncio.sleep(3)
                    continue

                img_data = images[0]
                if isinstance(img_data, dict):
                    url = img_data.get("image_url", {}).get("url", "")
                else:
                    url = img_data

                if url.startswith("data:image/"):
                    _, b64 = url.split(",", 1)
                else:
                    b64 = url

                img_bytes = base64.b64decode(b64)
                out_path.write_bytes(img_bytes)
                print(f"  OK   {name} ({len(img_bytes)} bytes)")
                return True

            except Exception as e:
                print(f"  ERR  {name}: {e}")
                await asyncio.sleep(5)

    print(f"  FAIL {name}")
    return False


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        tasks = [generate_one(client, name, prompt) for name, prompt in FIGURES.items()]
        results = await asyncio.gather(*tasks)
    ok = sum(results)
    print(f"\nDone: {ok}/{len(FIGURES)} generated")


if __name__ == "__main__":
    asyncio.run(main())
