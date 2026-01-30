#!/bin/bash
# Run LongLive with your quantized models
# Place this in: /media/sid/Kingston/longlive/LongLive/

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║    LongLive Interactive Inference with Quantized Models          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if quantized models exist
if [ ! -f "../longlive_models/longlive_base_bfloat16.pt" ]; then
    echo "❌ Error: Quantized models not found!"
    echo ""
    echo "Please run the quantization script first:"
    echo "  cd /media/sid/Kingston/longlive"
    echo "  python longlive_3x3090.py"
    exit 1
fi

echo "✅ Quantized models found"
echo ""

# Create a simple prompt file for testing
cat > test_prompts.txt << 'EOF'
A serene garden with colorful butterflies, sunny day, photorealistic
The butterflies begin to glow with a magical light, gathering together
The garden transforms into an enchanted mystical forest at twilight
EOF

echo "📝 Created test prompts:"
cat test_prompts.txt
echo ""

# Run with quantized models
python interactive_inference_quantized.py \
    --config_path configs/longlive_interactive_inference.yaml \
    --use_quantized

echo ""
echo "🎉 Done! Check the output folder for your video."
