#!/usr/bin/env bash
# =============================================================================
#  download_models.sh: restore every model weight that is NOT stored in git.
#
#  Usage (from anywhere, with the venv activated):
#      ./scripts/download_models.sh            # everything
#      ./scripts/download_models.sh vlm        # GroundingDINO + EfficientSAM + BLIP + BERT
#      ./scripts/download_models.sh yoloe      # YOLOE-11s-seg + MobileCLIP text encoder
#      ./scripts/download_models.sh smolvlm    # SmolVLM 256M / 500M OpenVINO int8 exports
#
#  Files that already exist are skipped, so the script is safe to re-run.
# =============================================================================
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLM_MODULES="$WS_DIR/src/perception/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules"
YOLOE_DIR="$WS_DIR/src/perception/yoloe/yoloe"
SMOLVLM_DIR="$WS_DIR/tools/smolvlm"

# Google Drive file with the OpenVINO IRs (from the vlms_with_ros2_workshop README)
OV_IRS_GDRIVE_ID="1yCpUEWr3KR76uVWgtHNreIdlYh59z0dS"

log()  { printf '\033[1;34m[models]\033[0m %s\n' "$*"; }
skip() { printf '\033[0;32m[skip]\033[0m   %s already present\n' "$1"; }

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: '$1' not found. Activate the venv first: source $WS_DIR/venv/bin/activate" >&2
        exit 1
    }
}

# fetch <url> <destination>
fetch() {
    if [[ -s "$2" ]]; then skip "${2#"$WS_DIR"/}"; return; fi
    log "Downloading $(basename "$2")"
    mkdir -p "$(dirname "$2")"
    curl -L --fail --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

download_vlm() {
    log "=== ros2_vlm: GroundingDINO / EfficientSAM / BLIP / BERT ==="

    fetch "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
          "$VLM_MODULES/checkpoints/groundingdino_swint_ogc.pth"

    # Optional PyTorch EfficientSAM weights (the ROS node itself uses the OpenVINO IR)
    fetch "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt" \
          "$VLM_MODULES/EfficientSAM/weights/efficient_sam_vitt.pt"

    local irs="$VLM_MODULES/openvino_irs"
    if [[ -s "$irs/openvino_grounding_dino.bin" && -s "$irs/blip_vision_model.bin" ]]; then
        skip "openvino_irs/*"
    else
        need gdown
        log "Downloading OpenVINO IR zip from Google Drive"
        local tmp; tmp="$(mktemp -d)"
        gdown "$OV_IRS_GDRIVE_ID" -O "$tmp/openvino_irs.zip"
        unzip -q "$tmp/openvino_irs.zip" -d "$tmp/unzipped"
        # The zip may contain a sub-folder; copy the IR files flat into openvino_irs/
        find "$tmp/unzipped" -type f \( -name '*.xml' -o -name '*.bin' \) -exec cp -v {} "$irs/" \;
        rm -rf "$tmp"
    fi

    need hf
    if [[ -s "$VLM_MODULES/bert-base-uncased/model.safetensors" ]]; then
        skip "bert-base-uncased/model.safetensors"
    else
        log "Downloading bert-base-uncased (text encoder used by GroundingDINO)"
        hf download google-bert/bert-base-uncased model.safetensors config.json vocab.txt \
            --local-dir "$VLM_MODULES/bert-base-uncased"
    fi
    # GroundingDINO loads "bert-base-uncased" by name, so also warm the HF cache
    python3 -c "from transformers import BertModel, BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')" >/dev/null
}

download_yoloe() {
    log "=== YOLOE ==="
    fetch "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt" \
          "$YOLOE_DIR/models/yoloe-11s-seg.pt"
    fetch "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts" \
          "$YOLOE_DIR/mobileclip_blt.ts"
}

# export_smolvlm <hf_model_id> <output_dir>
export_smolvlm() {
    if [[ -s "$2/openvino_language_model.bin" ]]; then skip "${2#"$WS_DIR"/}"; return; fi
    need optimum-cli
    log "Exporting $1 -> OpenVINO int8 (this takes a few minutes)"
    optimum-cli export openvino --model "$1" --weight-format int8 "$2"
}

download_smolvlm() {
    log "=== SmolVLM (OpenVINO int8) ==="
    export_smolvlm "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" "$SMOLVLM_DIR/smolvlm_250m_int8"
    export_smolvlm "HuggingFaceTB/SmolVLM-500M-Instruct"        "$SMOLVLM_DIR/smolvlm_500m_int8"
}

case "${1:-all}" in
    vlm)     download_vlm ;;
    yoloe)   download_yoloe ;;
    smolvlm) download_smolvlm ;;
    all)     download_vlm; download_yoloe; download_smolvlm ;;
    *)       echo "Usage: $0 [all|vlm|yoloe|smolvlm]" >&2; exit 1 ;;
esac

log "Done."
