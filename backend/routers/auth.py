import hashlib
import json
import logging
import os
from threading import Lock

from config.config import (
    ADVICE_RETRIEVAL_DIR,
    MINERU_OUTPUT_DIR,
    STORAGE_FILE,
    STUDENT_PDFS_PATH,
    STUDENT_RESULTS_PATH,
    SUB_LIMIT,
    USER_AIGC_RESULT_DIR,
    USER_MD_DIR,
    USER_RESULT_DIR,
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel


# 数据模型
class TeacherAnnotation(BaseModel):
    sectionScores: list
    totalScore: float
    adviceContent: str
    teacherComments: str = None


router = APIRouter()

lock = Lock()
registered_users = {}


# 加载用户数据
def load_users():
    """从存储文件中加载用户"""
    global registered_users
    if not os.path.isfile(STORAGE_FILE):
        return {}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
        logging.info(f"成功从 {STORAGE_FILE} 加载 {len(users)} 个用户")
        registered_users = users
        return users
    except Exception as e:
        logging.error(f"加载用户时出错: {str(e)}")
        return {}


# 保存用户数据
def save_users(users):
    """将用户保存到存储文件"""
    try:
        temp_file = f"{STORAGE_FILE}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, STORAGE_FILE)
        logging.info(f"成功将 {len(users)} 个用户保存到 {STORAGE_FILE}")
        return True
    except Exception as e:
        logging.error(f"保存用户时出错: {str(e)}")
    return False


# 密码哈希
def hash_password(password: str) -> str:
    """使用 SHA-256 对密码进行哈希处理"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# 工具函数
def load_json_file(folder: str, filename: str):
    """加载JSON文件"""
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_text_file(folder: str, filename: str):
    """加载文本文件"""
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def get_single_folders(student_id: str = None):
    """获取单个学生文件夹列表（递归搜索）"""
    folders = []

    # 搜索多个可能存储学生文件夹的目录
    search_dirs = [
        USER_RESULT_DIR,
    ]

    def recursive_search(path):
        if not os.path.exists(path):
            return

        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                # 检查是否有evaluation.txt、result.txt、metadata.json或evaluation_result.json文件
                if (
                    os.path.exists(os.path.join(item_path, "evaluation.txt"))
                    or os.path.exists(os.path.join(item_path, "result.txt"))
                    or os.path.exists(os.path.join(item_path, "metadata.json"))
                    or os.path.exists(os.path.join(item_path, "evaluation_result.json"))
                ):
                    # 如果提供了student_id，检查文件夹是否与该student_id相关
                    if student_id:
                        # 检查metadata.json中的student_id
                        metadata = load_json_file(item_path, "metadata.json")
                        if metadata and metadata.get("student_id") == student_id:
                            folders.append(item_path)
                        # 检查evaluation_result.json中的student_id
                        elif os.path.exists(
                            os.path.join(item_path, "evaluation_result.json")
                        ):
                            eval_data = load_json_file(
                                item_path, "evaluation_result.json"
                            )
                            if (
                                eval_data
                                and eval_data.get("metadata", {}).get("student_id")
                                == student_id
                            ):
                                folders.append(item_path)
                        # 或者检查文件夹名中是否包含student_id
                        elif student_id in item:
                            folders.append(item_path)
                    else:
                        # 如果没有提供student_id，添加所有符合条件的文件夹
                        folders.append(item_path)
                # 继续递归搜索
                recursive_search(item_path)

    for search_dir in search_dirs:
        recursive_search(search_dir)

    return folders


def get_joint_folders():
    """获取所有合作学生文件夹列表（递归搜索）"""
    if not os.path.exists(STUDENT_RESULTS_PATH):
        return []
    folders = []

    def recursive_search(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                # 检查是否有evaluation.txt、result.txt或metadata.json文件
                if (
                    os.path.exists(os.path.join(item_path, "evaluation.txt"))
                    or os.path.exists(os.path.join(item_path, "result.txt"))
                    or os.path.exists(os.path.join(item_path, "metadata.json"))
                ):
                    folders.append(item_path)
                # 继续递归搜索
                recursive_search(item_path)

    recursive_search(STUDENT_RESULTS_PATH)
    return folders


def get_student_folders(type: str, student_id: str = None):
    if type == "j":
        return get_joint_folders()
    elif type == "s":
        return get_single_folders(student_id)
    else:
        raise HTTPException(
            status_code=400, detail="Invalid type. Use 'j' for joint or 's' for single."
        )


def find_student_folder(student_id: str, type: str):
    """查找学生文件夹"""
    # 首先尝试使用student_id搜索
    folders = get_student_folders(type, student_id)
    logging.info(f"Searching for student_id: {student_id} in folders: {folders}")

    # 尝试匹配metadata.json中的student_id或file_name
    for folder in folders:
        metadata = load_json_file(folder, "metadata.json")
        if metadata:
            logging.info(
                f"Checking folder: {folder}, metadata student_id: {metadata.get('student_id')}, file_name: {metadata.get('file_name')}"
            )
            if metadata.get("student_id") == student_id:
                logging.info(f"Found folder via metadata student_id: {folder}")
                return folder
            elif metadata.get("file_name") and student_id in metadata.get("file_name"):
                logging.info(f"Found folder via metadata file_name: {folder}")
                return folder

    # 尝试在文件夹名中匹配学号
    for folder in folders:
        folder_name = os.path.basename(folder)
        logging.info(
            f"Checking folder name: {folder_name} for student_id: {student_id}"
        )
        if student_id in folder_name:
            logging.info(f"Found folder via folder name: {folder}")
            return folder

    # 如果没有找到，尝试获取所有文件夹并搜索
    if not folders:
        logging.info(
            f"No folders found with student_id: {student_id}, trying all folders"
        )
        all_folders = get_student_folders(type, None)
        logging.info(f"Found {len(all_folders)} total folders")

        # 再次尝试匹配
        for folder in all_folders:
            # 检查metadata.json中的student_id或file_name
            metadata = load_json_file(folder, "metadata.json")
            if metadata:
                logging.info(
                    f"Checking all folders - folder: {folder}, metadata student_id: {metadata.get('student_id')}, file_name: {metadata.get('file_name')}"
                )
                if metadata.get("student_id") == student_id:
                    logging.info(
                        f"Found folder via metadata student_id in all folders: {folder}"
                    )
                    return folder
                elif metadata.get("file_name") and student_id in metadata.get(
                    "file_name"
                ):
                    logging.info(
                        f"Found folder via metadata file_name in all folders: {folder}"
                    )
                    return folder

            # 检查文件夹名
            folder_name = os.path.basename(folder)
            if student_id in folder_name:
                logging.info(f"Found folder via folder name in all folders: {folder}")
                return folder

    logging.error(f"No folder found for student_id: {student_id}")
    return None


# 用户注册
@router.post("/register_user")
async def register_user(request: Request):
    """注册一个新用户，包含用户名和密码。"""
    global registered_users  # 在函数开始时声明为全局变量

    print(registered_users)  # 打印当前用户数据，用于调试

    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse({"error": "请求体过大"}), 413

        registered_users = load_users()

        data = await request.json()
        logging.info(f"收到来自 {request.client.host} 的注册请求: {data}")

        if not isinstance(data, dict):
            return JSONResponse({"error": "无效的 JSON 格式"}), 400

        username = data.get("username")
        logging.info(f"收到的用户名: {username}")  # 打印用户名，用于调试
        password = data.get("password")
        logging.info(f"收到的密码: {password}")  # 打印密码，用于调试
        role = data.get("role", "student")  # 默认角色为学生
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse({"error": "无效的用户名参数"}), 400
        if not password or not isinstance(password, str):
            logging.error("缺少或无效的密码")
            return JSONResponse({"error": "无效的密码参数"}), 400
        if role not in ["student", "teacher", "admin"]:
            logging.error("无效的角色")
            return JSONResponse({"error": "无效的角色参数"}), 400

        hashed_password = hash_password(password)

        with lock:
            if registered_users:  # 检查是否有用户数据
                logging.info(
                    f"当前已注册的用户: {list(registered_users.keys())}"
                )  # 打印当前用户列表，用于调试
                if username in registered_users:
                    logging.warning(f"用户名 '{username}' 已注册")
                    return JSONResponse({"error": "用户名已存在"}), 409

                temp_users = dict(registered_users)
                temp_users[username] = {
                    "password": hashed_password,
                    "subnum": 0,
                    "if_logged_in": 0,
                    "role": role,
                }
            else:
                temp_users = {
                    username: {
                        "password": hashed_password,
                        "subnum": 0,
                        "if_logged_in": 0,
                        "role": role,
                    }
                }
            if save_users(temp_users):
                registered_users = temp_users  # 直接在这里更新全局变量
                logging.info(
                    f"用户 '{username}' 注册成功，角色: {role} ({request.client.host})"
                )
            else:
                logging.error(f"无法持久化用户 '{username}'")
                return JSONResponse({"error": "服务器内部错误"}), 500

        return JSONResponse({"message": "用户注册成功", "role": role}, status_code=200)

    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse({"error": "无效的 JSON 格式"}), 400
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse({"error": "服务器内部错误"}), 500


# 用户登录
@router.post("/login_user")
async def login_user(request: Request):
    """用户登录"""
    global registered_users
    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse({"error": "请求体过大"}), 413

        registered_users = load_users()

        data = await request.json()
        logging.info(f"收到来自 {request.client.host} 的登录请求: {data}")
        if not isinstance(data, dict):
            return JSONResponse({"error": "无效的 JSON 格式"}), 400

        username = data.get("username")
        password = data.get("password")
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse({"error": "无效的用户名参数"}), 400
        if not password or not isinstance(password, str):
            logging.error("缺少或无效的密码")
            return JSONResponse({"error": "无效的密码参数"}), 400

        hashed_password = hash_password(password)
        logging.info(f"哈希密码{hashed_password}")  # 打印哈希后的密码，用于调试
        with lock:

            if username not in registered_users:
                logging.warning(f"用户 '{username}' 不存在 ({request.client.host})")
                return JSONResponse({"error": "用户名不存在"}), 404
            else:
                logging.info(registered_users[username])
                logging.info(registered_users[username]["if_logged_in"])
                if registered_users[username]["password"] == hashed_password:

                    if not registered_users[username]["if_logged_in"]:
                        temp_users = dict(registered_users)
                        temp_users[username]["if_logged_in"] = 1
                        if save_users(temp_users):
                            logging.info(
                                f"用户 '{username}' 登录成功 ({request.client.host})"
                            )
                        else:
                            logging.error(f"无法持久化用户 '{username}' 的登录状态")
                            return JSONResponse({"error": "服务器内部错误"}), 500
                        role = registered_users[username].get("role", "student")
                        return JSONResponse({"message": "登录成功", "role": role}), 200
                    else:
                        logging.warning(
                            f"用户 '{username}' 已登录 ({request.client.host})"
                        )
                        return JSONResponse({"message": "用户已登录"}), 403
                else:
                    logging.warning(
                        f"用户 '{username}' 登录失败 ({request.client.host})"
                    )
                    return JSONResponse({"error": "用户名或密码错误"}), 401

    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse({"error": "无效的 JSON 格式"}), 400
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse({"error": "服务器内部错误"}), 500


# 用户登出
@router.post("/logout")
async def logout(request: Request):
    """用户登出。"""
    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse({"error": "请求体过大"}), 413

        registered_users = load_users()

        data = await request.json()
        logging.info(f"收到来自 {request.client.host} 的登出请求: {data}")
        if not isinstance(data, dict):
            return JSONResponse({"error": "无效的 JSON 格式"}), 400

        username = data.get("username")
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse({"error": "无效的用户名参数"}), 401

        with lock:
            if username not in registered_users:
                logging.warning(f"用户 '{username}' 不存在 ({request.client.host})")
                return JSONResponse({"error": "用户名不存在"}), 404
            else:
                if registered_users[username]["if_logged_in"]:
                    temp_users = dict(registered_users)
                    role = registered_users[username].get("role", "student")
                    temp_users[username] = {
                        "password": registered_users[username]["password"],
                        "subnum": registered_users[username]["subnum"],
                        "if_logged_in": 0,
                        "role": role,
                    }
                    if save_users(temp_users):
                        logging.info(
                            f"用户 '{username}' 登出成功 ({request.client.host})"
                        )
                    else:
                        logging.error(f"无法持久化用户 '{username}' 的登出状态")
                        return JSONResponse({"error": "服务器内部错误"}), 500
                    return JSONResponse({"message": "登出成功"}), 200
                else:
                    logging.warning(f"用户 '{username}' 未登录 ({request.client.host})")
                    return JSONResponse({"message": "用户未登录"}), 201

    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse({"error": "无效的 JSON 格式"}), 400
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse({"error": "服务器内部错误"}), 500


@router.post("/sub_check")
async def sub_check(request: Request):
    """检查是否剩余提交次数"""
    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse({"error": "请求体过大"}), 413

        registered_users = load_users()

        data = await request.json()

        logging.info(f"收到来自 {request.client.host} 的提交检查请求: {data}")
        if not isinstance(data, dict):
            return JSONResponse({"error": "无效的 JSON 格式"}), 400

        username = data.get("username")
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse({"error": "无效的用户名参数"}), 401

        with lock:
            if username not in registered_users:
                logging.debug(f"用户 '{username}' 不存在 ({request.client.host})")
                return JSONResponse({"error": "用户名不存在"}), 404
            else:
                if registered_users[username]["subnum"] < SUB_LIMIT:
                    temp_users = dict(registered_users)
                    temp_users[username]["subnum"] += 1
                    # 确保角色信息被保留
                    if "role" not in temp_users[username]:
                        temp_users[username]["role"] = "student"
                    if save_users(temp_users):
                        logging.info(
                            f"用户 '{username}' 提交次数增加 ({request.client.host})"
                        )
                    else:
                        logging.error(f"无法持久化用户 '{username}' 的提交次数")
                        return JSONResponse({"error": "服务器内部错误"}), 500
                    logging.info(
                        f"用户 '{username}' 剩余提交次数: {SUB_LIMIT - temp_users[username]['subnum']}"
                    )
                    return (
                        JSONResponse(
                            {
                                "message": "剩余提交次数",
                                "subnum_remain": SUB_LIMIT
                                - temp_users[username]["subnum"],
                            }
                        ),
                        200,
                    )
                else:
                    logging.warning(
                        f"用户 '{username}' 提交次数已达上限 ({request.client.host})"
                    )
                    return JSONResponse({"error": "提交次数已达上限"}), 403
    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse({"error": "无效的 JSON 格式"}), 400
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse({"error": "服务器内部错误"}), 500


# 提交次数检查
@router.post("/check_sub")
async def check_sub(request: Request):
    """检查是否剩余提交次数"""
    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse({"error": "请求体过大"}), 413

        registered_users = load_users()
        logging.error(f"registered_users: {registered_users}")
        data = await request.json()
        logging.info(f"收到来自 {request.client.host} 的提交检查请求: {data}")
        if not isinstance(data, dict):
            return JSONResponse({"error": "无效的 JSON 格式"}), 400

        username = data.get("username")
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse({"error": "无效的用户名参数"}), 401

        with lock:
            if username not in registered_users:
                logging.error(f"用户 '{username}' 不存在 ({request.client.host})")
                return JSONResponse({"error": "用户名不存在"}), 404
            else:
                logging.error(
                    f"用户 '{username}' 剩余提交次数: {SUB_LIMIT - registered_users[username]['subnum']}"
                )
                return JSONResponse(
                    {
                        "message": "剩余可用提交次数",
                        "subnum_remain": SUB_LIMIT
                        - registered_users[username]["subnum"],
                    },
                    status_code=200,
                )
    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse({"error": "无效的 JSON 格式"}), 400
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse({"error": "服务器内部错误"}), 500


# 初始化用户数据
load_users()

# ============ 学生管理路由 (来自 result_display 和 teacher_review) ============


@router.get("/students", response_model=list[dict])
async def get_students():
    """
    获取所有学生列表
    整合自: result_display, teacher_review
    """
    try:
        folders = get_student_folders("j")
        logging.error(f"folders: {folders}")
        students = []

        for folder in folders:
            try:
                # 尝试从evaluation.txt或result.txt读取数据
                evaluation_content = load_text_file(folder, "evaluation.txt")
                result_content = load_text_file(folder, "result.txt")

                # 尝试从metadata.json读取学生信息
                metadata = load_json_file(folder, "metadata.json")

                # 从文件夹名提取学生信息
                folder_name = os.path.basename(folder)
                # 尝试解析学号和姓名
                student_id = ""
                student_name = ""
                paper_title = folder_name
                process_time = ""

                # 优先从metadata.json读取
                if metadata:
                    student_id = metadata.get("student_id", "")
                    student_name = metadata.get("student_name", "")
                    paper_title = metadata.get("paper_title", folder_name)
                    process_time = metadata.get("process_time", "")
                else:
                    # 从文件夹名解析
                    parts = folder_name.split(" ")
                    if len(parts) > 0:
                        # 假设第一部分是学号
                        if parts[0].isdigit():
                            student_id = parts[0]
                            # 尝试提取姓名
                            if len(parts) > 1:
                                student_name = parts[1]

                # 尝试从文件内容提取分数
                total_score = 0

                # 优先从teacher_evaluation.json中提取分数（教师保存的测评数据）
                teacher_evaluation = load_json_file(folder, "teacher_evaluation.json")
                if teacher_evaluation and teacher_evaluation.get("evaluation_data"):
                    score_list = teacher_evaluation.get("evaluation_data", {}).get(
                        "score_list", []
                    )
                    if score_list and len(score_list) > 18:
                        total_score = round(float(score_list[18]))
                        logging.info(
                            f"Score from teacher_evaluation.json: {total_score}"
                        )

                # 如果没有教师保存数据，尝试从evaluation_result.json中提取分数（系统评估数据）
                if total_score == 0:
                    evaluation_result = load_json_file(folder, "evaluation_result.json")
                    if evaluation_result and evaluation_result.get("total_score"):
                        total_score = round(float(evaluation_result.get("total_score")))
                        logging.info(
                            f"Score from evaluation_result.json: {total_score}"
                        )

                    # 如果没有系统评估数据，尝试从其他文件中提取分数
                    if total_score == 0 and evaluation_content:
                        # 简单的分数提取逻辑
                        if "得分" in evaluation_content:
                            import re

                            score_match = re.search(
                                r"得分:\s*([0-9.]+)", evaluation_content
                            )
                            if score_match:
                                total_score = round(float(score_match.group(1)))
                                logging.info(
                                    f"Score from evaluation.txt: {total_score}"
                                )

                    if total_score == 0 and result_content:
                        # 尝试从直接的得分字段提取
                        if "得分" in result_content:
                            import re

                            score_match = re.search(
                                r"得分:\s*([0-9.]+)", result_content
                            )
                            if score_match:
                                total_score = round(float(score_match.group(1)))
                                logging.info(f"Score from result.txt: {total_score}")
                        # 尝试从Summary Response中提取分数
                        elif "Summary Response:" in result_content:
                            # 简化处理：直接提取所有数字分数并计算平均分
                            import re

                            try:
                                # 提取所有数字分数
                                score_matches = re.findall(
                                    r'"?\d+"?\s*:\s*(\d+)', result_content
                                )
                                if score_matches:
                                    score_values = [
                                        float(score) for score in score_matches
                                    ]
                                    total_score = round(
                                        sum(score_values) / len(score_values)
                                    )
                                    logging.info(
                                        f"Score from result.txt scores: {total_score}"
                                    )
                            except Exception as e:
                                logging.error(
                                    f"Error extracting scores from result.txt: {e}"
                                )

                    if total_score == 0 and metadata and metadata.get("total_score"):
                        total_score = metadata.get("total_score")
                        logging.info(f"Score from metadata: {total_score}")

                students.append(
                    {
                        "id": folder,
                        "studentId": student_id,
                        "name": student_name,
                        "paperTitle": paper_title,
                        "processTime": process_time,
                        "totalScore": total_score,
                        "folder": folder,
                    }
                )
            except Exception as e:
                logging.error(f"Error reading data from {folder}: {e}")
                continue

        # 按学号排序
        students.sort(key=lambda x: x.get("studentId", ""))
        logging.error(f"students: {students}")
        return students
    except Exception as e:
        logging.error(f"Error in get_students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}")
async def get_student_detail(student_id: str, type: str):
    """
    获取单个学生详情
    整合自: result_display, teacher_review
    """
    try:
        logging.info(f"student_id: {student_id}, type: {type}")
        folder = find_student_folder(student_id, type)
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        metadata = load_json_file(folder, "metadata.json")
        evaluation = load_json_file(folder, "evaluation_result.json")
        teacher_evaluation = load_json_file(folder, "teacher_evaluation.json")
        workload = load_json_file(folder, "workload_evaluation.json")
        logging.error(f"teacher_evaluation: {teacher_evaluation}")
        if not metadata:
            raise HTTPException(status_code=404, detail="学生元数据未找到")

        # 优先从教师评估数据中读取
        if teacher_evaluation and teacher_evaluation.get("evaluation_data"):
            score_list = teacher_evaluation.get("evaluation_data", {}).get(
                "score_list", [0] * 19
            )
            advice_content = teacher_evaluation.get("evaluation_data", {}).get(
                "advice_content", ""
            )
            evaluation_time = teacher_evaluation.get("evaluation_data", {}).get(
                "teacher_evaluation_time", ""
            )
        else:
            # 否则从系统评估数据中读取
            score_list = (
                evaluation.get("evaluation_data", {}).get("score_list", [0] * 19)
                if evaluation
                else [0] * 19
            )
            advice_content = (
                evaluation.get("evaluation_data", {}).get("advice_content", "")
                if evaluation
                else ""
            )
            evaluation_time = (
                evaluation.get("evaluation_data", {}).get("evaluation_time", "")
                if evaluation
                else ""
            )

        return {
            "studentId": metadata.get("student_id", ""),
            "name": metadata.get("student_name", ""),
            "paperTitle": metadata.get("paper_title", ""),
            "processTime": metadata.get("process_time", ""),
            "totalScore": score_list[18] if len(score_list) > 18 else 0,
            "sectionScores": score_list[:18] if len(score_list) > 18 else score_list,
            "adviceContent": advice_content,
            "evaluationTime": evaluation_time,
            "structureScores": (
                workload.get("structure_evaluation", {}) if workload else {}
            ),
            "workloadAnalysis": (
                workload.get("workload_evaluation", {}).get("analysis", "")
                if workload
                else ""
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_student_detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/chapter-structure")
async def get_chapter_structure(student_id: str):
    """
    获取学生章节结构数据
    整合自: result_display
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        data = load_json_file(folder, "chapter_structure.json")
        if not data:
            raise HTTPException(status_code=404, detail="章节结构数据未找到")

        return data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_chapter_structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/chapter-info")
async def get_chapter_info(student_id: str):
    """
    获取学生章节详细信息
    整合自: result_display
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        data = load_json_file(folder, "chapter_information.json")
        if not data:
            raise HTTPException(status_code=404, detail="章节信息数据未找到")

        return data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_chapter_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/result-txt")
async def get_result_txt(student_id: str):
    """
    获取学生评估结果文本
    整合自: result_display
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        content = load_text_file(folder, "result.txt")
        if content is None:
            raise HTTPException(status_code=404, detail="结果文本未找到")

        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_result_txt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 数据导出路由 (来自 result_display) ============


@router.get("/students/export/csv")
async def export_students_csv():
    """
    导出学生评估数据为CSV
    整合自: result_display
    """
    try:
        folders = get_student_folders()
        students = []

        for folder in folders:
            try:
                metadata = load_json_file(folder, "metadata.json")
                evaluation = load_json_file(folder, "evaluation_result.json")

                if metadata and evaluation:
                    score_list = evaluation.get("evaluation_data", {}).get(
                        "score_list", [0] * 19
                    )
                    students.append(
                        {
                            "studentId": metadata.get("student_id", ""),
                            "name": metadata.get("student_name", ""),
                            "paperTitle": metadata.get("paper_title", ""),
                            "totalScore": score_list[18] if len(score_list) > 18 else 0,
                            "sectionScores": (
                                score_list[:18] if len(score_list) > 18 else score_list
                            ),
                            "processTime": metadata.get("process_time", ""),
                        }
                    )
            except Exception as e:
                logging.error(f"Error reading data from {folder}: {e}")
                continue

        # 生成CSV
        headers = [
            "学号",
            "姓名",
            "论文标题",
            "总分",
            "结构完整性",
            "摘要关键词",
            "目录规范性",
            "章节规范性",
            "参考文献格式",
            "致谢规范性",
            "选题契合度",
            "工作量适宜度",
            "学术价值",
            "文献检索能力",
            "知识综合应用",
            "专业方法运用",
            "专业技能实践",
            "技术应用能力",
            "创新性",
            "论证严谨性",
            "结构语言表达",
            "成果价值",
        ]

        csv_rows = [",".join(headers)]

        for s in students:
            row = [
                s["studentId"],
                s["name"],
                '"{}"'.format(s["paperTitle"].replace('"', '""')),
                str(s["totalScore"]),
            ] + [str(score) for score in s["sectionScores"]]
            csv_rows.append(",".join(row))

        csv_content = "\n".join(csv_rows)
        filename = f"学生评估数据_{int(time.time())}.csv"

        return Response(
            content="\ufeff" + csv_content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logging.error(f"Error in export_students_csv: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 教师评审路由 (来自 teacher_review) ============


@router.post("/students/{student_id}/evaluation")
async def save_teacher_evaluation(student_id: str, annotation: TeacherAnnotation):
    """
    保存教师评估数据
    整合自: teacher_review
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        # 读取原始评估数据
        evaluation = load_json_file(folder, "evaluation_result.json")
        if not evaluation:
            evaluation = {"evaluation_data": {}}

        # 更新分数和建议
        score_list = annotation.sectionScores + [annotation.totalScore]
        evaluation["evaluation_data"]["score_list"] = score_list
        evaluation["evaluation_data"]["advice_content"] = annotation.adviceContent
        evaluation["evaluation_data"]["teacher_comments"] = annotation.teacherComments
        evaluation["evaluation_data"]["teacher_evaluation_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 保存教师评估数据
        teacher_eval_path = os.path.join(
            STUDENT_RESULTS_PATH, folder, "teacher_evaluation.json"
        )
        with open(teacher_eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)

        return {"message": "评估数据保存成功"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in save_teacher_evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/evaluation")
async def get_teacher_evaluation(student_id: str):
    """
    获取教师评估数据
    整合自: teacher_review
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        # 优先读取教师评估数据
        teacher_eval = load_json_file(folder, "teacher_evaluation.json")
        if teacher_eval:
            return teacher_eval

        # 如果没有教师评估，返回系统评估
        evaluation = load_json_file(folder, "evaluation_result.json")
        if evaluation:
            return evaluation

        raise HTTPException(status_code=404, detail="评估数据未找到")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_teacher_evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/annotations")
async def get_student_annotations(student_id: str):
    """
    获取学生批注数据
    整合自: teacher_review
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        # 读取批注数据
        annotations = load_json_file(folder, "annotations.json")
        if not annotations:
            return {"annotations": []}

        return annotations
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_student_annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/visualization")
async def get_student_visualization(student_id: str):
    """
    获取学生可视化评分数据
    专门用于学生端读取可视化评分数据
    """
    try:
        logging.info(f"Getting visualization data for student_id: {student_id}")
        # 查找学生文件夹
        folder = find_student_folder(student_id, "s")
        if not folder:
            raise HTTPException(status_code=404, detail="学生文件夹未找到")

        # 读取评估数据
        metadata = load_json_file(folder, "metadata.json")
        evaluation = load_json_file(folder, "evaluation_result.json")
        workload = load_json_file(folder, "workload_evaluation.json")

        if not metadata:
            raise HTTPException(status_code=404, detail="学生元数据未找到")

        # 直接从系统评估数据中读取
        score_list = (
            evaluation.get("evaluation_data", {}).get("score_list", [0] * 19)
            if evaluation
            else [0] * 19
        )
        advice_content = (
            evaluation.get("evaluation_data", {}).get("advice_content", "")
            if evaluation
            else ""
        )
        evaluation_time = (
            evaluation.get("evaluation_data", {}).get("evaluation_time", "")
            if evaluation
            else ""
        )

        # 构建可视化评分数据
        visualization_data = {
            "studentId": metadata.get("student_id", ""),
            "name": metadata.get("student_name", ""),
            "paperTitle": metadata.get("paper_title", ""),
            "processTime": metadata.get("process_time", ""),
            "totalScore": score_list[18] if len(score_list) > 18 else 0,
            "sectionScores": score_list[:18] if len(score_list) > 18 else score_list,
            "adviceContent": advice_content,
            "evaluationTime": evaluation_time,
            "structureScores": (
                workload.get("structure_evaluation", {}) if workload else {}
            ),
            "workloadAnalysis": (
                workload.get("workload_evaluation", {}).get("analysis", "")
                if workload
                else ""
            ),
            "sectionNames": [
                "结构完整性",
                "摘要关键词",
                "目录规范性",
                "章节规范性",
                "参考文献格式",
                "致谢规范性",
                "选题契合度",
                "工作量适宜度",
                "学术价值",
                "文献检索能力",
                "知识综合应用",
                "专业方法运用",
                "专业技能实践",
                "技术应用能力",
                "创新性",
                "论证严谨性",
                "结构语言表达",
                "成果价值",
            ],
        }

        logging.info(f"Visualization data for {student_id}: {visualization_data}")
        return visualization_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_student_visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/students/{student_id}/annotations")
async def save_student_annotations(student_id: str, annotations: dict):
    """
    保存学生批注数据
    整合自: teacher_review
    """
    try:
        folder = find_student_folder(student_id, "j")
        if not folder:
            raise HTTPException(status_code=404, detail="学生未找到")

        # 保存批注数据
        annotations_path = os.path.join(
            STUDENT_RESULTS_PATH, folder, "annotations.json"
        )
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        return {"message": "批注数据保存成功"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in save_student_annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ PDF文件服务路由 (来自 teacher_review) ============


@router.get("/pdf/{student_id}")
async def get_student_pdf(student_id: str):
    """
    获取学生PDF文件
    整合自: teacher_review
    """
    try:
        logging.info(f"Requesting PDF for student_id: {student_id}")

        # 查找学生文件夹获取PDF文件名
        folder = find_student_folder(student_id, "j")
        logging.info(f"Found folder: {folder}")
        if not folder:
            logging.error(f"Student folder not found for student_id: {student_id}")
            raise HTTPException(status_code=404, detail="学生未找到")

        # 读取metadata.json获取PDF文件名
        metadata = load_json_file(folder, "metadata.json")
        logging.info(f"Metadata: {metadata}")
        if not metadata:
            logging.error(f"Metadata not found for folder: {folder}")
            # 尝试直接使用student_id作为PDF文件名
            pdf_file_name = f"{student_id}.pdf"
            logging.info(f"Trying PDF file name: {pdf_file_name}")
        else:
            if not metadata.get("file_name"):
                logging.error(f"File name not found in metadata for folder: {folder}")
                # 尝试直接使用student_id作为PDF文件名
                pdf_file_name = f"{student_id}.pdf"
                logging.info(f"Trying PDF file name: {pdf_file_name}")
            else:
                pdf_file_name = metadata.get("file_name")
                logging.info(f"PDF file name from metadata: {pdf_file_name}")

        # 从STUDENT_PDFS_PATH读取PDF
        pdf_path = os.path.join(STUDENT_PDFS_PATH, pdf_file_name)
        logging.info(f"PDF path: {pdf_path}")

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file not found at: {pdf_path}")
            # 尝试列出STUDENT_PDFS_PATH中的文件
            if os.path.exists(STUDENT_PDFS_PATH):
                files = os.listdir(STUDENT_PDFS_PATH)
                logging.info(f"Files in STUDENT_PDFS_PATH: {files}")
                # 尝试在文件名中查找包含student_id的文件
                for file in files:
                    if student_id in file:
                        pdf_file_name = file
                        pdf_path = os.path.join(STUDENT_PDFS_PATH, pdf_file_name)
                        logging.info(f"Found PDF file by student_id: {pdf_file_name}")
                        if os.path.exists(pdf_path):
                            return FileResponse(pdf_path, media_type="application/pdf")
            else:
                logging.error(f"STUDENT_PDFS_PATH does not exist: {STUDENT_PDFS_PATH}")
            raise HTTPException(status_code=404, detail="PDF文件未找到")

        return FileResponse(pdf_path, media_type="application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_student_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 统计信息路由 ============


@router.get("/statistics/dashboard")
async def get_dashboard_statistics():
    """
    获取仪表板统计数据
    整合自: result_display
    """
    try:
        folders = get_student_folders("j")
        total_students = len(folders)

        total_score = 0
        excellent_count = 0

        for folder in folders:
            try:
                evaluation = load_json_file(folder, "evaluation_result.json")
                if evaluation:
                    score_list = evaluation.get("evaluation_data", {}).get(
                        "score_list", [0] * 19
                    )
                    if len(score_list) > 18:
                        total = score_list[18]
                        total_score += total
                        if total >= 90:
                            excellent_count += 1
            except Exception:
                continue

        avg_score = total_score / total_students if total_students > 0 else 0
        excellent_rate = (
            (excellent_count / total_students * 100) if total_students > 0 else 0
        )

        return {
            "totalStudents": total_students,
            "averageScore": round(avg_score, 2),
            "excellentCount": excellent_count,
            "excellentRate": round(excellent_rate, 2),
        }
    except Exception as e:
        logging.error(f"Error in get_dashboard_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


import time
