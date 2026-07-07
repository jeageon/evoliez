#!/usr/bin/env bash
# ===========================================================================
# check_cpu_quota.sh
#   공유 서버에서 특정 유저 계정에 걸린 systemd CPUQuota(코어 제한)를
#   진단하고, 원하면 해제한다. 관리자(root)가 실행하는 용도.
#
#   sudo bash check_cpu_quota.sh                # jglee 진단 (읽기 전용)
#   sudo bash check_cpu_quota.sh <user>         # 다른 유저 진단
#   sudo bash check_cpu_quota.sh <user> --fix   # 진단 후 제한 "해제"까지
#
# 읽기 전용(--fix 없으면 아무것도 바꾸지 않음). [4]번 실측은 해당 유저로
# CPU 무한루프 8개를 5초 돌려 실제로 몇 코어 받는지 측정한다.
# ===========================================================================
set -uo pipefail

USER_NAME="${1:-jglee}"
[ "$USER_NAME" = "--fix" ] && USER_NAME="jglee"
DO_FIX=0; [ "${1:-}" = "--fix" ] && DO_FIX=1; [ "${2:-}" = "--fix" ] && DO_FIX=1

UID_NUM="$(id -u "$USER_NAME" 2>/dev/null)" || { echo "ERROR: user '$USER_NAME' 없음"; exit 1; }
SLICE="user-${UID_NUM}.slice"

run_as_user() {   # 대상 유저로 실행 (이미 그 유저면 그대로, 아니면 sudo)
  if [ "$(id -un)" = "$USER_NAME" ]; then bash -c "$1"; else sudo -u "$USER_NAME" bash -c "$1"; fi
}

echo "######## CPU 쿼터 진단: $USER_NAME (uid $UID_NUM, $SLICE) ########"; echo

echo "===[1] systemd 가 보고하는 한도 ==="
systemctl show "$SLICE" -p CPUQuotaPerSecUSec -p CPUQuotaPeriodUSec -p CPUAccounting 2>/dev/null
echo "   ※ CPUQuotaPerSecUSec=200ms  ==  0.2코어 제한.  'infinity' 면 무제한."; echo

echo "===[2] 어디서 설정됐나 (출처) ==="
CONF="/etc/systemd/system.control/${SLICE}.d/50-CPUQuota.conf"
if [ -f "$CONF" ]; then
  echo ">>> $CONF   (마지막 수정: $(stat -c %y "$CONF" 2>/dev/null))"
  sed 's/^/      /' "$CONF" 2>/dev/null
  echo "   => 'systemctl set-property $SLICE CPUQuota=...' 로 이 계정에 직접 설정된 것"
  echo "      (글로벌 기본값이면 user-.slice.d 에 있어야 함 — 여긴 uid 전용 경로)"
else
  echo "   system.control 드롭인 없음. 다른 출처 확인:"
  systemctl show "$SLICE" -p DropInPaths 2>/dev/null | tr ' ' '\n' | grep -iv '^$' | sed 's/^/      /'
fi; echo

echo "===[3] 커널이 실제로 강제하는 값 (cgroup) ==="
FOUND=0
for CG in "/sys/fs/cgroup/cpu,cpuacct/user.slice/${SLICE}" "/sys/fs/cgroup/cpu/user.slice/${SLICE}"; do
  if [ -f "$CG/cpu.cfs_quota_us" ]; then
    FOUND=1; q=$(cat "$CG/cpu.cfs_quota_us"); p=$(cat "$CG/cpu.cfs_period_us")
    if [ "$q" = "-1" ]; then echo "  $CG : 무제한(-1)"
    else echo "  cpu.cfs_quota_us=$q / cpu.cfs_period_us=$p  =>  $(awk "BEGIN{printf \"%.2f\",$q/$p}") 코어"; fi
    awk '/nr_periods|nr_throttled|throttled_time/{printf "    %s\n",$0}' "$CG/cpu.stat" 2>/dev/null
  fi
done
CG2="/sys/fs/cgroup/user.slice/${SLICE}"
[ -f "$CG2/cpu.max" ] && { FOUND=1; echo "  cpu.max = $(cat "$CG2/cpu.max")   (cgroup v2)"; }
[ "$FOUND" = 0 ] && echo "  (cgroup cpu 파일 못 찾음)"; echo

echo "===[4] 실측 증명: 해당 유저로 CPU 무한루프 8개 × 5초 ==="
echo "   ※ 서버 한가할 때: 무제한이면 ~40 CPU초, 0.2코어 캡이면 ~1 CPU초"
USAGE=""
for U in "/sys/fs/cgroup/cpu,cpuacct/user.slice/${SLICE}/cpuacct.usage" "/sys/fs/cgroup/cpu/user.slice/${SLICE}/cpuacct.usage"; do
  [ -f "$U" ] && USAGE="$U"
done
if [ -n "$USAGE" ]; then
  u1=$(cat "$USAGE")
  run_as_user 'for i in $(seq 1 8); do timeout 5 bash -c "while :; do :; done" & done; wait' 2>/dev/null
  u2=$(cat "$USAGE")
  echo "   실제 소비 = $(awk "BEGIN{printf \"%.1f\",($u2-$u1)/1e9}") CPU초"
  echo "   (현재 load: $(cut -d' ' -f1-3 /proc/loadavg) / $(nproc)코어)"
else
  echo "   (cpuacct.usage 못 찾아 실측 생략)"
fi; echo

echo "===[5] 해제 / 상향 방법 (root) ==="
echo "   제거:  systemctl set-property $SLICE CPUQuota="
echo "   상향:  systemctl set-property $SLICE CPUQuota=2000%      # 예: 20코어"
echo "   영구化 시 드롭인 직접 편집/삭제 후:  systemctl daemon-reload"
if [ "$DO_FIX" = "1" ]; then
  echo; echo "   --fix 지정됨 → 제한 제거 실행..."
  if systemctl set-property "$SLICE" CPUQuota= 2>/dev/null; then
    echo "   ✔ 제거 완료. 재확인:"
    systemctl show "$SLICE" -p CPUQuotaPerSecUSec 2>/dev/null | sed 's/^/      /'
  else
    echo "   ✘ 실패 (root 권한 필요 / polkit 거부)"
  fi
fi
