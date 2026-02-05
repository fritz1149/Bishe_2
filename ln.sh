#!/bin/bash

# 基础目录数组
BASE_DIRS=("datasets/rag-corpus" "datasets/rag-emb" )

# SAMPLE_NUM数组
SAMPLE_NUMS=("300" "400" "500" "600" "700" "800")

# 原始目录名称列表
SRC_DIRS=("browser" "CICMalAnal-Benign" "ISCX-VPN-service")

# 目标目录名称
DST_DIR="3mixed"

for BASE_DIR in "${BASE_DIRS[@]}"; do
    for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
        # 目标目录的实际路径
        DST_PATH="${BASE_DIR}/${SAMPLE_NUM}/${DST_DIR}"

        mkdir -p "$DST_PATH"

        for SRC in "${SRC_DIRS[@]}"; do
            # 源目录的路径
            SRC_PATH="${BASE_DIR}/${SAMPLE_NUM}/${SRC}"
            if [ ! -d "$SRC_PATH" ]; then
                echo "Warning: source directory ${SRC_PATH} does not exist, skipped."
                continue
            fi
            # 检查是否存在train子目录
            if [ -d "${SRC_PATH}/train" ]; then
                # 存在train/test结构
                for SPLIT in train test; do
                    SPLIT_SRC="${SRC_PATH}/${SPLIT}"
                    SPLIT_DST="${DST_PATH}/${SPLIT}"
                    mkdir -p "$SPLIT_DST"
                    if [ ! -d "$SPLIT_SRC" ]; then
                        echo "Warning: subdir ${SPLIT_SRC} not found, skipped."
                        continue
                    fi
                    for f in "$SPLIT_SRC"/*; do
                        [ -e "$f" ] || continue
                        basefile=$(basename "$f")
                        ln "$f" "${SPLIT_DST}/${SRC}_${basefile}"
                    done
                done
            else
                # 不存在train子目录，直接处理SRC_PATH
                for f in "$SRC_PATH"/*; do
                    [ -e "$f" ] || continue
                    basefile=$(basename "$f")
                    ln "$f" "${DST_PATH}/${SRC}_${basefile}"
                done
            fi
        done
    done
done

