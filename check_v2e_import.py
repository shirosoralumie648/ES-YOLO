import sys
import traceback

try:
    print("开始诊断：尝试导入 v2e 模块...")
    
    # 尝试导入 v2e 的关键组件
    from v2e import v2e
    from v2e.renderer import EventRenderer
    
    print("成功：v2e 模块已成功导入。")
    print("结论：v2e 库本身似乎已正确安装。问题可能出在命令行执行或其依赖项的运行时行为上。")
    
except ImportError as e:
    print("错误：导入 v2e 模块失败。", file=sys.stderr)
    print("这表明 v2e 未正确安装或其依赖项缺失。", file=sys.stderr)
    print(f"具体错误信息: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print("错误：在导入过程中发生未知异常。", file=sys.stderr)
    print("这可能指向一个更深层次的依赖库冲突或环境问题。", file=sys.stderr)
    print(f"具体错误信息: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
