#!/bin/bash
#
# GPU 监控脚本 - 在压力测试期间运行
# 每 10 秒记录一次 GPU 状态
#

LOG_FILE="gpu_monitor_$(date '+%Y%m%d_%H%M%S').log"

echo "开始 GPU 监控 - 日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo ""

# 写入日志头
{
    echo "================================================================================"
    echo "GPU 监控日志"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""
} > "$LOG_FILE"

# 监控循环
COUNTER=0
while true; do
    COUNTER=$((COUNTER + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    {
        echo "===== 记录 #$COUNTER - $TIMESTAMP ====="
        echo ""
        
        # GPU 状态摘要
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm \
            --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: Temp=%s°C, GPU利用率=%s%%, 显存=%s/%sMB, 功率=%sW, 时钟=%sMHz\n", $1, $2, $4, $6, $7, $8, $9}'
        
        echo ""
        
        # 每分钟输出详细信息
        if [ $((COUNTER % 6)) -eq 0 ]; then
            echo "--- 详细 GPU 状态 (每分钟) ---"
            nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem \
                --format=csv
            echo ""
        fi
        
    } | tee -a "$LOG_FILE"
    
    # 同时在终端显示摘要
    echo "[$(date '+%H:%M:%S')] 监控记录 #$COUNTER - GPU 温度和利用率已记录"
    
    sleep 10
done
