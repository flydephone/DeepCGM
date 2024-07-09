from PIL import Image
import os,glob

# 这里替换为你的PNG文件列表
filedir = "figure/series_MC_base_prior13_01_convergence_fit_scratch"
filename_list = glob.glob(f"{filedir}/*.png")  # 确保这里只匹配 SVG 文件
filename_list = filename_list[0:500]
filename_list = [tpt for i,tpt in enumerate(filename_list) if i%3==0]
# 输出的GIF文件名
output_gif = 'DeepCGM.gif'

# 创建一个Image对象的列表
images = [Image.open(fn) for fn in filename_list]

# 保存为GIF
images[0].save(output_gif, save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)




