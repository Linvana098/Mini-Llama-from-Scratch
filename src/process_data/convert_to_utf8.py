import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def convert_single_txt(file_path, overwrite=False, backup=True):
    """转换单个 TXT 文件（GB2312 → UTF-8）- 修复覆盖乱码问题"""
    try:

        with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
            content = f.read()  # 内容先存在内存里，和文件本身解耦

        # 第二步：处理备份（覆盖前先备份，避免读取时文件已被修改）
        output_path = file_path
        if overwrite:
            # 先备份原文件（读取完成后再备份，避免备份空文件）
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                # 避免重复备份
                if not os.path.exists(backup_path):
                    os.rename(file_path, backup_path)
        else:
            # 不覆盖：生成新文件
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_utf8{ext}"

        # 第三步：写入内容（此时不管是否覆盖，读取的都是原始内容）
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

        return (True, f"成功：{os.path.basename(file_path)}")
    except PermissionError:
        return (False, f"失败：{os.path.basename(file_path)} → 权限不足/文件被占用")
    except Exception as e:
        return (False, f"失败：{os.path.basename(file_path)} → {str(e)}")


def batch_convert_all_subdirs(root_dir, overwrite=False, backup=True):
    """递归遍历所有子文件夹，批量转换 TXT（修复覆盖逻辑）"""
    total_files = 0
    success = 0
    fail = 0

    print(f"📌 开始递归处理目录：{root_dir}（包括所有子文件夹）")
    print("-" * 50)

    for dirpath, _, filenames in os.walk(root_dir):
        txt_files = [os.path.join(dirpath, fn) for fn in filenames if fn.lower().endswith('.txt')]
        if not txt_files:
            continue

        print(f"\n📂 正在处理目录：{dirpath}")
        for txt_file in txt_files:
            total_files += 1
            res, msg = convert_single_txt(txt_file, overwrite, backup)
            print(f"  {msg}")
            success += 1 if res else 0
            fail += 1 if not res else 0

    print("-" * 50)
    print(f"\n📊 转换完成！")
    print(f"总计扫描到 TXT 文件：{total_files} 个")
    print(f"转换成功：{success} 个 | 转换失败：{fail} 个")


# ####################### 你只需要修改这里的配置 #######################
if __name__ == "__main__":

    target_dir = "./data/raw"

    # 执行批量转换
    batch_convert_all_subdirs(
        root_dir=target_dir,
        overwrite=False,
        backup=True
    )