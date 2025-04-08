from PIL import Image
import os


def split_gif_to_jpg(gif_path, output_folder, num_frames):

    os.makedirs(output_folder, exist_ok=True)
    gif = Image.open(gif_path)
    total_frames = gif.n_frames

    for i in range(num_frames):
        frame_num = i % total_frames  # 如果帧数少于 num_frames，就循环复用
        gif.seek(frame_num)  # 选取帧
        frame = gif.convert("RGBA")  # 处理透明背景

        # 创建白色背景，防止透明背景变黑
        white_bg = Image.new("RGBA", frame.size, (255, 255, 255, 255))
        combined = Image.alpha_composite(white_bg, frame)

        # 旋转（角度随 i 递增）
        # rotation_angle = (i + 1) % 360  # 防止角度超过 360
        # rotated_frame = combined.rotate(rotation_angle, expand=True)

        # 转换为 RGB 并保存为 JPG
        rgb_frame = combined.convert("RGB")
        output_path = os.path.join(output_folder, f"King_Slime_{i + 1}.jpg")
        rgb_frame.save(output_path, "JPEG")

    print(f"GIF 已分割并扩增 {num_frames} 张 JPG 图片，保存在 {output_folder} 文件夹。")

#"E:\\terraria_project\\Eye_of_Cthulhu_(Phase_1).gif"
#"E:\\terraria_project\\Eye_of_Cthulhu_(Second_Phase).gif"
#"E:\\terraria_project\\King_Slime.gif"
#"E:\\terraria_project\\BOSS_img"
gif_path = "E:\\terraria_project\\King_Slime.gif"
output_folder = "E:\\terraria_project\\BOSS_img"
num_frames = 360  # 分割帧数

split_gif_to_jpg(gif_path, output_folder, num_frames)


