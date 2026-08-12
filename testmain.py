import toupcam
import cv2 as cv
import numpy as np
import keyboard
import time
import os
import ctypes  # 修改：使用可写的 ctypes buffer 接收 SDK 写入的图像数据


type1 = int(input("模式(0,1,2):"))
if type1 not in (0,1,2):
    print("错误")
    exit()

if type1 == 0:
    fps = 15
elif type1 == 1:
    fps = 50
elif type1 == 2:
    fps = 60
cam = toupcam.Toupcam.Open(None)
if cam is None:
    print("失败")
    exit()

cam.put_eSize(type1)
width, height = cam.get_Size()
speed_level = cam.get_Speed()  # 修改：get_Speed() 返回的是速度档位，不是真实 fps
print("width, height,speed_level=", width, height, speed_level)

buf_size = ((width * 24 + 31) // 32 * 4) * height
buf = ctypes.create_string_buffer(buf_size)

try:
    cam.put_Option(toupcam.TOUPCAM_OPTION_TRIGGER, 0)#####11111111
    cam.put_Option(toupcam.TOUPCAM_OPTION_RAW,0)
    cam.StartPullModeWithCallback(None, None)
except toupcam.HRESULTException as e:
    print(f"相机启动失败: {e}")
    cam.Close()
    exit()
a = 10  # 自动截图持续时间 (秒)
b = 2  # 自动截图间隔时间 (秒)
k1, k2 = 0, 0  # k1: 录制状态 (0=关, 1=开); k2: 自动截图状态 (0=关, 1=开)
s1, s2, s3 = 0, 0, 0  # s1: 自动截图会话计数; s2: 会话内截图计数; s3: 视频文件计数
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = None

SAVE_DIR = r'C:\Users\Origami\Desktop\111'


cv.namedWindow('frame', cv.WINDOW_NORMAL)


#python_fps 是本程序成功 PullImageV2 的帧率，sdk_fps 是 SDK 统计的相机实际帧率
fps_count = 0
fps_last_time = time.perf_counter()
python_fps = 0.0
sdk_fps = 0.0
sdk_total_frames = 0
real_exposure_us = 0


while True:
    try:
        cam.PullImageV2(buf, 24, None)
    except toupcam.HRESULTException as ex:
        hr = ex.hr & 0xFFFFFFFF
        if hr == 0x8001011f or hr == 0x8000000a:
            time.sleep(0.001)  # 休息一下，避免 CPU 占用过高
            continue  # 跳过本次循环，等待下一帧
        print(f"拉取图像失败: {ex} (hr=0x{hr:08X})")
        break


    raw_data = np.frombuffer(buf, dtype=np.uint8)
    frame = high_res_frame_rgb = raw_data.reshape((height, width, 3))

    # 修改：每秒刷新一次真实帧率，避免把 get_Speed() 的速度档位误认为 fps。
    fps_count += 1
    now = time.perf_counter()
    if now - fps_last_time >= 1.0:
        python_fps = fps_count / (now - fps_last_time)
        sdk_frame_count, sdk_time_ms, sdk_total_frames = cam.get_FrameRate()
        sdk_fps = sdk_frame_count * 1000.0 / sdk_time_ms if sdk_time_ms else 0.0
        try:
            real_exposure_us = cam.get_RealExpoTime()
        except toupcam.HRESULTException:
            real_exposure_us = 0

        try:
            cv.setWindowTitle(
                'frame',
                f"frame | python {python_fps:.2f} fps | sdk {sdk_fps:.2f} fps | exposure {real_exposure_us} us"
            )
        except cv.error:
            pass
        fps_count = 0
        fps_last_time = now



    cv.imshow('frame', frame)

    if keyboard.is_pressed('d') and k1 == 0:
        k1 = 1
        s3 += 1
        name = os.path.join(SAVE_DIR, f"视频-{int(s3)}.mp4")
        out = cv.VideoWriter(name, fourcc, fps, (width, height))
        print(f"开始录制: {name}")

    if keyboard.is_pressed('f') and k1 == 1:
        k1 = 0
        if out:
            out.release()
            print(f"停止录制: 视频-{int(s3)}.mp4")
        out = None

    if k1 == 1:
        if out:
            out.write(frame)

    if keyboard.is_pressed('a') and k2 != 1:
        s1 += 1
        s2 = 0
        k2 = 1
        t1 = time.time()
        t3 = t1
        while keyboard.is_pressed('a'):
            cv.waitKey(1)
        print(f"开始自动截图会话 {s1} (持续 {a}秒, 间隔 {b}秒)")

    if k2 == 1:
        t2 = time.time()
        if t2 - t3 >= b:
            s2 += 1
            t3 = t2
            c1 = [SAVE_DIR, f"{s1}-{s2}.jpg"]
            d = os.path.join(*c1)

            # 保存高分辨率帧
            if cv.imwrite(d, frame):
                print(f"截图保存: {d}")
            else:
                print(f"截图失败: 无法写入文件 {d}")

        if t2 - t1 >= a + 0.1:
            k2 = 0
            print(f"自动截图会话 {s1} 结束")


    if cv.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
        if k1 == 1:
            if out:
                out.release()
        break


if cam:
    cam.Stop()
    cam.Close()

cv.destroyAllWindows()
print("程序已退出并清理资源。")  
