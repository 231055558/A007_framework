import asyncio
import shutil
import uuid
import zipfile
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict
import os
from datetime import datetime
from PIL import Image
import io

from soloprocess import DeepLabV3Plus_Solo_Detect
from batchprocess import DeepLabV3Plus_Batch_Detect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"WebSocket连接已建立 - session_id: {session_id}")  # 新增日志
        print(f"当前活动连接数: {len(self.active_connections)}")  # 新增日志

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        print(f"WebSocket连接已断开 - session_id: {session_id}")  # 新增日志

    async def send_progress(self, session_id: str, progress: int, message: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json({
                    "progress": progress,
                    "message": message
                })
                print(f"进度消息已发送 - session_id: {session_id}, progress: {progress}, message: {message}")
            except Exception as e:
                print(f"发送进度时出错: {str(e)}")
        else:
            print(f"未找到WebSocket连接 - session_id: {session_id}")  # 新增日志

manager = ConnectionManager()
# 创建FastAPI应用
app = FastAPI()

# 创建静态文件目录
STATIC_DIR = "D://static//"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
# 确保临时文件夹存在
TEMP_DIR = os.path.join(IMAGES_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 配置CORS
app.add_middleware(    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/soloupload")
async def upload_files(left_eye: UploadFile, right_eye: UploadFile) -> Dict:
    try:
        # 验证文件格式
        for file in [left_eye, right_eye]:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是图像格式")
        
        # 读取并处理图像
        left_content = await left_eye.read()
        left_image = Image.open(io.BytesIO(left_content))
        right_content = await right_eye.read()
        right_image = Image.open(io.BytesIO(right_content))
        result = DeepLabV3Plus_Solo_Detect(left_image, right_image).solo_detect(path=TEMP_DIR)
        print(result)

        result_url = f"/static/images/temp/pred.png"
        # 返回包含诊断结果和图像URL的响应
        return {
            "diagnosis": result,
            "image_url": result_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_file")
async def process_file(file: UploadFile):
    try:
        # 生成会话ID
        session_id = str(uuid.uuid4())
        print(f"新建处理任务 - session_id: {session_id}")  # 添加日志
        # 等待一段时间，确保WebSocket连接建立
        #response = {"session_id": session_id, "status": "waiting_for_connection"}
        # 创建该会话的临时目录
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 保存上传的文件
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            comtent = await file.read()
            buffer.write(comtent)
            
        return {"session_id": session_id, "status": "Created"}

    except Exception as e:
        print(f"上传文件异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_process/{session_id}")
async def start_process(session_id: str):
    try:
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail="会话不存在")
            
        # 获取上传的文件路径
        files = os.listdir(session_dir)
        if not files:
            raise HTTPException(status_code=400, detail="未找到上传的文件")
            
        file_path = os.path.join(session_dir, files[0])
        
        # 启动异步处理任务
        asyncio.create_task(process_files_async(session_id, session_dir, file_path))
        
        return {"status": "Processing"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def process_files_async(session_id: str, session_dir: str, file_path: str):
    try:
        print(f"开始处理文件 - session_id: {session_id}")  # 添加日志      
        # 创建解压任务
        await manager.send_progress(session_id, 0, "开始解压文件...")
        await asyncio.create_task(process_zip_file(session_id, file_path))
        await manager.send_progress(session_id, 100, "解压完成")
        print (f"解压完成 - session_id: {session_id}")
        await asyncio.sleep(1)  # 等待一段时间，确保解压完成
        await manager.send_progress(session_id, 0, "开始处理图像...")
        #解压完毕后进入处理环节
        #查看根目录下的txt文件，返回文件名，若没有则返回错误
        txt_files = [f for f in os.listdir(session_dir) if f.endswith('.txt')]
        if not txt_files:
            await manager.send_progress(session_id, -1, "未找到txt文件")
            raise HTTPException(status_code=500, detail="未找到txt文件")
        if len(txt_files) > 1:
            await manager.send_progress(session_id, -1, "找到多个txt文件")
            raise HTTPException(status_code=500, detail="找到多个txt文件")
        txt_file = txt_files[0]
        pretrain_ckp = os.path.join(os.path.dirname(__file__), "best_model.pth")
        #await manager.send_progress(session_id, 0, "开始处理图像...")
        detector = DeepLabV3Plus_Batch_Detect(data_root=session_dir, txt_file=txt_file, threshold=0.5, pretrain_ckp=pretrain_ckp)
        
        #创建进度回调函数
        async def progress_callback(progress: int, message: str):
            await manager.send_progress(session_id, progress, message)

        print(f"开始推理，数据目录: {session_dir}, txt文件: {txt_file}")
        result = await detector.predict_model(progress_callback=progress_callback)
        print(f"推理结果: {result}")  # 添加调试信息
        await manager.send_progress(session_id, 100, "处理完成")

        return {"session_id": session_id, "result": result}

        
    except Exception as e:
        await manager.send_progress(session_id, -1, f"处理失败{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/progress/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    print(f"收到WebSocket连接请求 - session_id: {session_id}")  # 新增日志
    await manager.connect(websocket, session_id)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        print(f"WebSocket连接异常 - session_id: {session_id}, 错误: {str(e)}")  # 新增日志
        manager.disconnect(session_id)

# ... 现有代码 ...

@app.get("/process_status/{session_id}")
async def get_process_status(session_id: str):
    try:
        # 检查会话目录是否存在
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 检查结果文件是否存在
        result_file = os.path.join(session_dir, "output", "0.5.txt")
        if os.path.exists(result_file):
            return {"status": "completed", "message": "处理完成"}
        
        # 检查WebSocket连接状态
        if session_id in manager.active_connections:
            return {"status": "processing", "message": "正在处理中"}
        
        return {"status": "waiting", "message": "等待处理"}
        
    except Exception as e:
        print(f"获取处理状态失败: {str(e)}")  # 添加错误日志
        raise HTTPException(status_code=500, detail=str(e))

async def process_zip_file(session_id: str, file_path: str):
    try:
        # 解压文件
        extract_path = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            for index, file in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(file, extract_path)
                progress = int((index / total_files) * 100)
                await manager.send_progress(session_id, progress, f"正在处理第 {index}/{total_files} 个文件")
        #删除上传的zip文件
        os.remove(file_path)
        # 发送完成消息
        await manager.send_progress(session_id, 100, "解压完成")
        
    except Exception as e:
        await manager.send_progress(session_id, -1, f"解压失败: {str(e)}")

@app.get("/download/{session_id}")
async def download_result(session_id: str):
    file_path = os.path.join(UPLOAD_DIR, session_id, "output", "0.5.txt")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return FileResponse(file_path, filename="0.5.txt")

@app.get("/")
async def root():
    return {"message": "欢迎使用眼科诊断API"}

# 添加新的异步函数来处理后台任务
'''async def handle_background_task(task, session_id):
    try:
        await task
    except Exception as e:
        await manager.send_progress(session_id, -1, f"处理失败: {str(e)}")'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)