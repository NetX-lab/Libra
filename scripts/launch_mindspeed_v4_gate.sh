set -eu
base=/data/qianzhirong/runtime_sources/Libra_MindSpeed_PP2_20261007
gate_image="${LIBRA_GATE_IMAGE:-libra-v4-cann91-ready:20261007}"
node_ip=$(hostname -I)
rank=-1
idx=0
for ip in 192.168.0.50 192.168.0.189 192.168.0.41 192.168.0.99 192.168.0.48 192.168.0.63 192.168.0.85 192.168.0.89 192.168.0.220 192.168.0.112 192.168.0.217 192.168.0.2 192.168.0.138 192.168.0.71 192.168.0.29 192.168.0.109; do
 case " $node_ip " in *" $ip "*) rank=$idx ;; esac
 idx=$((idx + 1))
done
[ "$rank" -ge 0 ] || exit 2
for attempt in $(seq 1 60); do
 docker image inspect deepseek-v4-dspark:v30-fixed >/dev/null 2>&1 && break
 sleep 5
done
[ "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8 ] || { echo 'NPU node busy'; exit 3; }
[ "$(docker image inspect deepseek-v4-dspark:v30-fixed --format '{{.Id}}')" = 'sha256:ef517fe815af8c8d68a3fb5c8e2432866843dd7690f163a28266e345f0b3aaf2' ] || exit 4
docker image inspect "$gate_image" >/dev/null 2>&1 || exit 4
container=libra-v4-full-gate-20261007
if docker container inspect "$container" >/dev/null 2>&1; then echo 'Existing gate container; inspect it before relaunch'; exit 5; fi
mkdir -p "$base/full_v4_gate/logs"
: > "$base/full_v4_gate/logs/node_${rank}.log"
rm -f "$base/full_v4_gate/logs/node_${rank}.exit"
docker run -d --name "$container" --network host --shm-size 64g \
 --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
 --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /data:/data:ro -v "$base":/libra-work \
 -e LIBRA_SOURCE_PARENT=/libra-work \
 -e LIBRA_CANN_PREINSTALLED="${LIBRA_CANN_PREINSTALLED:-1}" \
 -e NODE_RANK="$rank" -e OMP_NUM_THREADS=4 \
 --entrypoint /bin/bash "$gate_image" -c \
 'set -o pipefail; log="/libra-work/full_v4_gate/logs/node_${NODE_RANK}.log"; fail() { code=$?; echo "$code" > "/libra-work/full_v4_gate/logs/node_${NODE_RANK}.exit"; exit "$code"; }; if [ "${LIBRA_CANN_PREINSTALLED:-0}" != 1 ]; then ln -sfn /data/image/libra-cann91-toolkit-installed /tmp/cann91tk; bash /data/image/Ascend-cann-ops_9.1.0_linux-aarch64.run --quiet --install --install-path=/tmp/libra-cann91-ops --install-for-all >> "$log" 2>&1 || fail; python3 -m pip install --no-deps --force-reinstall /data/image/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl >> "$log" 2>&1 || fail; fi; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; source /libra-work/env.sh || fail; python3 /libra-work/scripts/check_mindspeed_v4_ops.py >> "$log" 2>&1 || fail; python3 /libra-work/scripts/patch_mindspeed_gloo_groups.py >> "$log" 2>&1 || fail; timeout 3600 python3 -m torch.distributed.run --master-addr=192.168.0.50 --master-port=29681 --nnodes=16 --nproc_per_node=8 --node_rank="$NODE_RANK" /libra-work/validate_v4.py >> "$log" 2>&1; code=$?; echo "$code" > "/libra-work/full_v4_gate/logs/node_${NODE_RANK}.exit"; exit "$code"'
