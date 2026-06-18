import os
import shutil
import tarfile
from datetime import datetime

def get_latest_log_dir(base_dir):
    """获取 base_dir 下最近修改的子目录"""
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    # 按最后修改时间排序
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir

def main():
    # 路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_base = os.path.join(project_root, "legged_gym/logs")
    export_dir = os.path.join(project_root, "docs/logs")
    
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # 1. 寻找最新日志
    latest_log = get_latest_log_dir(logs_base)
    if not latest_log:
        print(f"错误: 在 {logs_base} 中没有找到任何日志目录。")
        return

    log_name = os.path.basename(latest_log)
    # 我们使用固定名称以便在文档中引用，或者保留 log_name
    report_path = os.path.join(export_dir, log_name)

    # 如果目标文件夹已存在，先删除（保证最新）
    if os.path.exists(report_path):
        shutil.rmtree(report_path)

    print(f"🔍 检测到最新训练日志: {latest_log}")

    # 2. 复制日志到导出目录 (排除 .pt 文件以减小体积)
    def ignore_pt_files(dir, contents):
        return [c for c in contents if c.endswith('.pt')]

    shutil.copytree(latest_log, report_path, ignore=ignore_pt_files)
    
    # 3. 打包压缩
    timestamp = datetime.now().strftime("%Y%m%d")
    archive_path = os.path.join(export_dir, f"tensorboard_logs_{timestamp}.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(report_path, arcname=log_name)

    print("\n" + "="*50)
    print("✅ 报告整合完成！")
    print(f"📂 日志存放位置: \n   {report_path}")
    print(f"📦 压缩包路径 (发给别人): \n   {archive_path}")
    print("="*50)
    print("\n💡 如何在本地查看面板？")
    print(f"tensorboard --logdir={export_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
